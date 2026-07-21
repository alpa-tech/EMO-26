import argparse
from contextlib import nullcontext
from pathlib import Path
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from dataset import build_datasets
from model import build_model
from preprocessing import build_eval_transform, build_train_transform
from utils import AverageMeter, append_csv, classification_metrics, count_parameters
from utils import get_device, load_json, save_checkpoint, save_json, seed_everything


def train_epoch(model, loader, criterion, optimizer, device, scaler, mixed_precision, gradient_clip):
    model.train()
    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    for images, landmarks, targets in loader:
        images = images.to(device, non_blocking=True)
        landmarks = landmarks.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        context = autocast(enabled=mixed_precision) if device.type == "cuda" else nullcontext()
        with context:
            logits = model(images, landmarks)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        predictions = logits.detach().argmax(dim=1)
        batch_accuracy = (predictions == targets).float().mean().item()
        batch_size = images.shape[0]
        loss_meter.update(loss.item(), batch_size)
        accuracy_meter.update(batch_accuracy, batch_size)
    return loss_meter.average, accuracy_meter.average


def evaluate_epoch(model, loader, criterion, device, mixed_precision):
    model.eval()
    loss_meter = AverageMeter()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device, non_blocking=True)
            landmarks = batch[1].to(device, non_blocking=True)
            targets = batch[2].to(device, non_blocking=True)
            context = autocast(enabled=mixed_precision) if device.type == "cuda" else nullcontext()
            with context:
                logits = model(images, landmarks)
                loss = criterion(logits, targets)
            predictions = logits.argmax(dim=1)
            batch_size = images.shape[0]
            loss_meter.update(loss.item(), batch_size)
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())
    return loss_meter.average, y_true, y_pred


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir")
    parser.add_argument("--landmark_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--landmark_checkpoint")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--se_levels", nargs="*")
    parser.add_argument("--train_landmark_encoder", action="store_true")
    parser.add_argument("--no_pretrained_backbone", action="store_true")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def merged_configuration(arguments):
    configuration = load_json(arguments.config)
    direct_values = {
        "data_dir": arguments.data_dir,
        "landmark_dir": arguments.landmark_dir,
        "output_dir": arguments.output_dir,
        "landmark_checkpoint": arguments.landmark_checkpoint,
        "device": arguments.device,
        "epochs": arguments.epochs,
        "batch_size": arguments.batch_size,
        "num_workers": arguments.num_workers,
        "seed": arguments.seed
    }
    for key, value in direct_values.items():
        if value is not None:
            configuration[key] = value
    if arguments.se_levels is not None:
        configuration["se_levels"] = arguments.se_levels
    if arguments.train_landmark_encoder:
        configuration["freeze_landmark_encoder"] = False
    if arguments.no_pretrained_backbone:
        configuration["pretrained_backbone"] = False
    if arguments.amp:
        configuration["amp"] = True
    return configuration


def main():
    arguments = parse_arguments()
    configuration = merged_configuration(arguments)
    required = ["data_dir", "landmark_dir", "output_dir"]
    missing = [key for key in required if not configuration.get(key)]
    if missing:
        raise ValueError(f"Missing configuration values: {missing}")
    seed_everything(configuration.get("seed", 42))
    device = get_device(configuration.get("device"))
    output_dir = Path(configuration["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    image_size = configuration.get("image_size", 224)
    train_transform = build_train_transform(image_size)
    eval_transform = build_eval_transform(image_size)
    train_set, val_set, test_set = build_datasets(
        configuration["data_dir"],
        configuration["landmark_dir"],
        train_transform,
        eval_transform
    )
    batch_size = configuration.get("batch_size", 32)
    num_workers = configuration.get("num_workers", 4)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0
    )
    model = build_model(
        num_classes=len(train_set.classes),
        pretrained_backbone=configuration.get("pretrained_backbone", True),
        embedding_dimension=configuration.get("embedding_dimension", 256),
        se_reduction=configuration.get("se_reduction", 16),
        se_levels=configuration.get("se_levels", ["fine", "mid", "global"]),
        token_size=configuration.get("token_size", 7),
        transformer_depth=configuration.get("transformer_depth", 4),
        transformer_heads=configuration.get("transformer_heads", 8),
        mlp_ratio=configuration.get("mlp_ratio", 4.0),
        dropout=configuration.get("dropout", 0.1),
        landmark_checkpoint=configuration.get("landmark_checkpoint"),
        freeze_landmark_encoder=configuration.get("freeze_landmark_encoder", False)
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=configuration.get("label_smoothing", 0.0))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=configuration.get("learning_rate", 1e-4),
        weight_decay=configuration.get("weight_decay", 1e-4)
    )
    epochs = configuration.get("epochs", 300)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mixed_precision = configuration.get("amp", False) and device.type == "cuda"
    scaler = GradScaler(enabled=mixed_precision)
    gradient_clip = configuration.get("gradient_clip", 5.0)
    best_accuracy = -1.0
    history_path = output_dir / "history.csv"
    save_json(output_dir / "configuration.json", configuration)
    save_json(
        output_dir / "model_information.json",
        {
            "backbone": "ResNet-18",
            "input_size": [image_size, image_size],
            "classes": train_set.classes,
            "trainable_parameters": count_parameters(model),
            "attention": "Transformer self-attention",
            "se_levels": configuration.get("se_levels", ["fine", "mid", "global"])
        }
    )
    header = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
        "validation_macro_f1"
    ]
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            mixed_precision,
            gradient_clip
        )
        validation_loss, y_true, y_pred = evaluate_epoch(
            model,
            val_loader,
            criterion,
            device,
            mixed_precision
        )
        metrics = classification_metrics(y_true, y_pred, train_set.classes)
        validation_accuracy = metrics["accuracy"]
        learning_rate = optimizer.param_groups[0]["lr"]
        append_csv(
            history_path,
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "validation_macro_f1": metrics["macro_f1"]
            },
            header
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_accuracy,
                train_set.classes,
                configuration
            )
            save_json(output_dir / "best_validation_metrics.json", metrics)
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_accuracy,
            train_set.classes,
            configuration
        )
        scheduler.step()
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.6f} "
            f"train_accuracy={train_accuracy:.6f} "
            f"validation_loss={validation_loss:.6f} "
            f"validation_accuracy={validation_accuracy:.6f} "
            f"best_accuracy={best_accuracy:.6f}"
        )
    save_json(
        output_dir / "summary.json",
        {
            "best_validation_accuracy": best_accuracy,
            "best_checkpoint": str(output_dir / "best.pt"),
            "test_available": test_set is not None
        }
    )


if __name__ == "__main__":
    main()
