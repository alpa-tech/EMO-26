import random
from PIL import Image
from torchvision.transforms import ColorJitter, InterpolationMode
from torchvision.transforms import functional as F


DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


class PairedTransform:
    def __init__(self, image_size=224, train=False, mean=DEFAULT_MEAN, std=DEFAULT_STD):
        self.image_size = image_size
        self.train = train
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.color_jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

    def __call__(self, image, landmark):
        image = image.convert("RGB")
        landmark = landmark.convert("L")
        image = F.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR
        )
        landmark = F.resize(
            landmark,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR
        )
        if self.train:
            if random.random() < 0.5:
                image = F.hflip(image)
                landmark = F.hflip(landmark)
            angle = random.uniform(-10.0, 10.0)
            translate = (
                int(random.uniform(-0.05, 0.05) * self.image_size),
                int(random.uniform(-0.05, 0.05) * self.image_size)
            )
            scale = random.uniform(0.95, 1.05)
            image = F.affine(
                image,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR
            )
            landmark = F.affine(
                landmark,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR
            )
            if random.random() < 0.5:
                image = self.color_jitter(image)
        image_tensor = F.to_tensor(image)
        image_tensor = F.normalize(image_tensor, self.mean, self.std)
        landmark_tensor = F.to_tensor(landmark).repeat(3, 1, 1)
        return image_tensor, landmark_tensor


def build_train_transform(image_size=224):
    return PairedTransform(image_size=image_size, train=True)


def build_eval_transform(image_size=224):
    return PairedTransform(image_size=image_size, train=False)


def denormalize(tensor, mean=DEFAULT_MEAN, std=DEFAULT_STD):
    mean_tensor = tensor.new_tensor(mean).view(-1, 1, 1)
    std_tensor = tensor.new_tensor(std).view(-1, 1, 1)
    return tensor * std_tensor + mean_tensor


def load_image(path):
    return Image.open(path).convert("RGB")
