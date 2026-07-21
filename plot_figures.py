import argparse
import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def save_figure(figure, output):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(path)


def plot_training(arguments):
    rows = read_csv(arguments.input)
    epochs = [int(row["epoch"]) for row in rows]
    train_accuracy = [float(row["train_accuracy"]) for row in rows]
    validation_accuracy = [float(row["validation_accuracy"]) for row in rows]
    figure = plt.figure(figsize=(7, 5))
    axis = figure.add_subplot(1, 1, 1)
    axis.plot(epochs, train_accuracy, label="Training")
    axis.plot(epochs, validation_accuracy, label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy")
    axis.legend()
    axis.grid(True, alpha=0.3)
    save_figure(figure, arguments.output)


def plot_confusion(arguments):
    with Path(arguments.input).open("r", encoding="utf-8") as file:
        metrics = json.load(file)
    matrix = np.asarray(metrics["normalized_confusion_matrix"]) * 100.0
    classes = metrics["classes"]
    figure = plt.figure(figsize=(8, 7))
    axis = figure.add_subplot(1, 1, 1)
    image = axis.imshow(matrix)
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    axis.set_yticks(range(len(classes)), classes)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row, column]:.1f}", ha="center", va="center")
    save_figure(figure, arguments.output)


def plot_comparison(arguments):
    rows = read_csv(arguments.input)
    datasets = sorted({row["dataset"] for row in rows})
    methods = sorted({row["method"] for row in rows})
    x_positions = np.arange(len(datasets))
    width = 0.8 / max(len(methods), 1)
    figure = plt.figure(figsize=(9, 5))
    axis = figure.add_subplot(1, 1, 1)
    for index, method in enumerate(methods):
        values = []
        for dataset in datasets:
            matching = [
                row for row in rows if row["dataset"] == dataset and row["method"] == method
            ]
            values.append(float(matching[0]["accuracy"]) if matching else np.nan)
        axis.bar(x_positions + index * width, values, width=width, label=method)
    axis.set_xticks(x_positions + width * (len(methods) - 1) / 2.0, datasets)
    axis.set_ylabel("Accuracy (%)")
    axis.legend()
    save_figure(figure, arguments.output)


def plot_ablation(arguments):
    rows = read_csv(arguments.input)
    names = [row["variant"] for row in rows]
    accuracy = [float(row["accuracy"]) for row in rows]
    macro_f1 = [float(row["macro_f1"]) for row in rows]
    x_positions = np.arange(len(names))
    width = 0.35
    figure = plt.figure(figsize=(10, 5))
    axis = figure.add_subplot(1, 1, 1)
    axis.bar(x_positions - width / 2, accuracy, width, label="Accuracy")
    axis.bar(x_positions + width / 2, macro_f1, width, label="Macro F1")
    axis.set_xticks(x_positions, names, rotation=25, ha="right")
    axis.set_ylabel("Score")
    axis.legend()
    save_figure(figure, arguments.output)


def parse_arguments():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("training", "confusion", "comparison", "ablation"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--input", required=True)
        subparser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    functions = {
        "training": plot_training,
        "confusion": plot_confusion,
        "comparison": plot_comparison,
        "ablation": plot_ablation
    }
    functions[arguments.command](arguments)


if __name__ == "__main__":
    main()
