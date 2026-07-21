from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class PairedFERDataset(Dataset):
    def __init__(self, image_root, landmark_root, transform, return_path=False):
        self.image_root = Path(image_root)
        self.landmark_root = Path(landmark_root)
        self.transform = transform
        self.return_path = return_path
        self.classes = sorted(
            directory.name for directory in self.image_root.iterdir() if directory.is_dir()
        )
        if not self.classes:
            raise ValueError(f"No class directories found in {self.image_root}")
        self.class_to_idx = {class_name: index for index, class_name in enumerate(self.classes)}
        self.samples = []
        for class_name in self.classes:
            class_directory = self.image_root / class_name
            for path in sorted(class_directory.rglob("*")):
                if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    relative_path = path.relative_to(self.image_root)
                    landmark_path = self._find_landmark(relative_path)
                    self.samples.append((path, landmark_path, self.class_to_idx[class_name]))
        if not self.samples:
            raise ValueError(f"No images found in {self.image_root}")

    def _find_landmark(self, relative_path):
        candidates = [
            self.landmark_root / relative_path,
            (self.landmark_root / relative_path).with_suffix(".png"),
            (self.landmark_root / relative_path).with_suffix(".jpg")
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No landmark heatmap found for {relative_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, landmark_path, target = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        landmark = Image.open(landmark_path).convert("L")
        image, landmark = self.transform(image, landmark)
        if self.return_path:
            return image, landmark, target, str(image_path)
        return image, landmark, target


def build_datasets(data_dir, landmark_dir, train_transform, eval_transform):
    data_dir = Path(data_dir)
    landmark_dir = Path(landmark_dir)
    train_set = PairedFERDataset(
        data_dir / "train",
        landmark_dir / "train",
        train_transform
    )
    val_set = PairedFERDataset(
        data_dir / "val",
        landmark_dir / "val",
        eval_transform
    )
    test_set = None
    if (data_dir / "test").exists():
        test_set = PairedFERDataset(
            data_dir / "test",
            landmark_dir / "test",
            eval_transform,
            return_path=True
        )
    if train_set.classes != val_set.classes:
        raise ValueError("Training and validation class names do not match")
    if test_set is not None and train_set.classes != test_set.classes:
        raise ValueError("Training and test class names do not match")
    return train_set, val_set, test_set
