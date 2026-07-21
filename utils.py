import csv
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0.0
        self.average = 0.0
        self.total = 0.0
        self.count = 0

    def update(self, value, count=1):
        self.value = float(value)
        self.total += float(value) * count
        self.count += count
        self.average = self.total / max(self.count, 1)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(device=None):
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_csv(path, rows, header):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def append_csv(path, row, header):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_accuracy, classes, configuration):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": int(epoch),
            "best_accuracy": float(best_accuracy),
            "classes": list(classes),
            "configuration": configuration
        },
        path
    )


def classification_metrics(y_true, y_pred, classes):
    labels = list(range(len(classes)))
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(precision_weighted),
        "weighted_recall": float(recall_weighted),
        "weighted_f1": float(f1_weighted),
        "classes": list(classes),
        "confusion_matrix": matrix.tolist(),
        "normalized_confusion_matrix": normalized.tolist()
    }
