import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path


RAF_LABELS = {
    1: "surprise",
    2: "fear",
    3: "disgust",
    4: "happiness",
    5: "sadness",
    6: "anger",
    7: "neutral"
}

AFFECTNET_LABELS = {
    0: "neutral",
    1: "happiness",
    2: "sadness",
    3: "surprise",
    4: "fear",
    5: "disgust",
    6: "anger",
    7: "contempt"
}


def transfer(source, destination, mode):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())
    else:
        raise ValueError(mode)


def stratified_split(items, validation_ratio, seed):
    grouped = defaultdict(list)
    for item in items:
        grouped[item[1]].append(item)
    generator = random.Random(seed)
    train_items = []
    validation_items = []
    for values in grouped.values():
        generator.shuffle(values)
        validation_count = max(1, int(round(len(values) * validation_ratio)))
        validation_items.extend(values[:validation_count])
        train_items.extend(values[validation_count:])
    return train_items, validation_items


def write_items(items, split, arguments):
    output_root = Path(arguments.output_root)
    for source, label in items:
        destination = output_root / split / label / source.name
        transfer(source, destination, arguments.mode)


def prepare_rafdb(arguments):
    image_root = Path(arguments.image_root)
    records = []
    with Path(arguments.partition_file).open("r", encoding="utf-8") as file:
        for line in file:
            filename, raw_label = line.strip().split()
            label = RAF_LABELS[int(raw_label)]
            split = "test" if filename.startswith("test") else "train"
            stem = Path(filename).stem
            candidates = [
                image_root / filename,
                image_root / f"{stem}_aligned.jpg",
                image_root / "Image" / "aligned" / f"{stem}_aligned.jpg",
                image_root / "aligned" / f"{stem}_aligned.jpg"
            ]
            source = next((candidate for candidate in candidates if candidate.exists()), None)
            if source is None:
                raise FileNotFoundError(filename)
            records.append((source, label, split))
    training = [(source, label) for source, label, split in records if split == "train"]
    testing = [(source, label) for source, label, split in records if split == "test"]
    training, validation = stratified_split(training, arguments.validation_ratio, arguments.seed)
    write_items(training, "train", arguments)
    write_items(validation, "val", arguments)
    write_items(testing, "test", arguments)


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def first_value(row, names):
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def ferplus_label(row):
    direct = first_value(row, ["emotion", "label", "Emotion"])
    if direct is not None:
        return int(float(direct))
    vote_columns = [
        "neutral",
        "happiness",
        "surprise",
        "sadness",
        "anger",
        "disgust",
        "fear",
        "contempt"
    ]
    votes = [float(row.get(column, 0) or 0) for column in vote_columns]
    return max(range(len(votes)), key=votes.__getitem__)


def prepare_ferplus(arguments):
    labels = [
        "neutral",
        "happiness",
        "surprise",
        "sadness",
        "anger",
        "disgust",
        "fear",
        "contempt"
    ]
    image_root = Path(arguments.image_root)
    items_by_split = defaultdict(list)
    for row in read_csv(arguments.metadata_csv):
        relative = first_value(row, ["path", "image", "Image name", "filename", "file"])
        usage = first_value(row, ["Usage", "usage", "split", "set"])
        if relative is None or usage is None:
            raise ValueError("FERPlus metadata requires image path and split columns")
        label_index = ferplus_label(row)
        if label_index < 0 or label_index >= len(labels):
            continue
        normalized_usage = usage.strip().lower()
        if "train" in normalized_usage:
            split = "train"
        elif "public" in normalized_usage or "val" in normalized_usage:
            split = "val"
        elif "private" in normalized_usage or "test" in normalized_usage:
            split = "test"
        else:
            continue
        source = image_root / relative
        if not source.exists():
            raise FileNotFoundError(source)
        items_by_split[split].append((source, labels[label_index]))
    for split in ("train", "val", "test"):
        write_items(items_by_split[split], split, arguments)


def prepare_affectnet(arguments):
    image_root = Path(arguments.image_root)
    class_limit = 7 if arguments.classes == 7 else 8
    training = []
    testing = []
    for row in read_csv(arguments.train_csv):
        relative = first_value(row, ["subDirectory_filePath", "path", "file", "image"])
        expression = first_value(row, ["expression", "label", "emotion"])
        if relative is None or expression is None:
            continue
        label_index = int(float(expression))
        if label_index not in AFFECTNET_LABELS or label_index >= class_limit:
            continue
        source = image_root / relative
        if source.exists():
            training.append((source, AFFECTNET_LABELS[label_index]))
    for row in read_csv(arguments.test_csv):
        relative = first_value(row, ["subDirectory_filePath", "path", "file", "image"])
        expression = first_value(row, ["expression", "label", "emotion"])
        if relative is None or expression is None:
            continue
        label_index = int(float(expression))
        if label_index not in AFFECTNET_LABELS or label_index >= class_limit:
            continue
        source = image_root / relative
        if source.exists():
            testing.append((source, AFFECTNET_LABELS[label_index]))
    training, validation = stratified_split(training, arguments.validation_ratio, arguments.seed)
    write_items(training, "train", arguments)
    write_items(validation, "val", arguments)
    write_items(testing, "test", arguments)


def parse_arguments():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="dataset", required=True)
    rafdb = subparsers.add_parser("rafdb")
    rafdb.add_argument("--image_root", required=True)
    rafdb.add_argument("--partition_file", required=True)
    rafdb.add_argument("--output_root", required=True)
    rafdb.add_argument("--validation_ratio", type=float, default=0.1)
    rafdb.add_argument("--seed", type=int, default=42)
    rafdb.add_argument("--mode", choices=["copy", "symlink"], default="copy")
    ferplus = subparsers.add_parser("ferplus")
    ferplus.add_argument("--image_root", required=True)
    ferplus.add_argument("--metadata_csv", required=True)
    ferplus.add_argument("--output_root", required=True)
    ferplus.add_argument("--mode", choices=["copy", "symlink"], default="copy")
    affectnet = subparsers.add_parser("affectnet")
    affectnet.add_argument("--image_root", required=True)
    affectnet.add_argument("--train_csv", required=True)
    affectnet.add_argument("--test_csv", required=True)
    affectnet.add_argument("--output_root", required=True)
    affectnet.add_argument("--classes", type=int, choices=[7, 8], default=7)
    affectnet.add_argument("--validation_ratio", type=float, default=0.05)
    affectnet.add_argument("--seed", type=int, default=42)
    affectnet.add_argument("--mode", choices=["copy", "symlink"], default="copy")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    if arguments.dataset == "rafdb":
        prepare_rafdb(arguments)
    elif arguments.dataset == "ferplus":
        prepare_ferplus(arguments)
    elif arguments.dataset == "affectnet":
        prepare_affectnet(arguments)


if __name__ == "__main__":
    main()
