import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from dataset import PairedFERDataset
from model import build_model
from preprocessing import build_eval_transform
from utils import classification_metrics, get_device, save_json, write_csv


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--landmark_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device")
    return parser.parse_args()


def model_from_checkpoint(checkpoint, device):
    configuration = checkpoint["configuration"]
    classes = checkpoint["classes"]
    model = build_model(
        num_classes=len(classes),
        pretrained_backbone=False,
        embedding_dimension=configuration.get("embedding_dimension", 256),
        se_reduction=configuration.get("se_reduction", 16),
        se_levels=configuration.get("se_levels", ["fine", "mid", "global"]),
        token_size=configuration.get("token_size", 7),
        transformer_depth=configuration.get("transformer_depth", 4),
        transformer_heads=configuration.get("transformer_heads", 8),
        mlp_ratio=configuration.get("mlp_ratio", 4.0),
        dropout=configuration.get("dropout", 0.1),
        landmark_checkpoint=None,
        freeze_landmark_encoder=False
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, configuration, classes


def main():
    arguments = parse_arguments()
    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(arguments.device)
    checkpoint = torch.load(arguments.checkpoint, map_location=device)
    model, configuration, classes = model_from_checkpoint(checkpoint, device)
    dataset = PairedFERDataset(
        arguments.image_dir,
        arguments.landmark_dir,
        build_eval_transform(configuration.get("image_size", 224)),
        return_path=True
    )
    if dataset.classes != classes:
        raise ValueError(f"Checkpoint classes {classes} do not match dataset classes {dataset.classes}")
    loader = DataLoader(
        dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.num_workers,
        pin_memory=device.type == "cuda"
    )
    model.eval()
    y_true = []
    y_pred = []
    rows = []
    with torch.no_grad():
        for images, landmarks, targets, paths in loader:
            images = images.to(device, non_blocking=True)
            landmarks = landmarks.to(device, non_blocking=True)
            logits = model(images, landmarks)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1).cpu()
            confidences = probabilities.max(dim=1).values.cpu()
            for path, target, prediction, confidence in zip(paths, targets, predictions, confidences):
                true_index = int(target)
                predicted_index = int(prediction)
                y_true.append(true_index)
                y_pred.append(predicted_index)
                rows.append(
                    [
                        path,
                        true_index,
                        classes[true_index],
                        predicted_index,
                        classes[predicted_index],
                        float(confidence)
                    ]
                )
    metrics = classification_metrics(y_true, y_pred, classes)
    save_json(output_dir / "metrics.json", metrics)
    write_csv(
        output_dir / "predictions.csv",
        rows,
        ["path", "true_index", "true_label", "pred_index", "pred_label", "confidence"]
    )
    print(f"accuracy={metrics['accuracy']:.6f}")
    print(f"macro_f1={metrics['macro_f1']:.6f}")
    print(f"balanced_accuracy={metrics['balanced_accuracy']:.6f}")


if __name__ == "__main__":
    main()
