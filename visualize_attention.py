import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
from evaluation import model_from_checkpoint
from preprocessing import build_eval_transform, denormalize
from utils import get_device


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--landmark", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=["fine", "mid", "global"], default="global")
    parser.add_argument("--target_class", type=int)
    parser.add_argument("--device")
    return parser.parse_args()


def normalize_map(value):
    value = value - value.min()
    maximum = value.max()
    if maximum > 0:
        value = value / maximum
    return value


def main():
    arguments = parse_arguments()
    device = get_device(arguments.device)
    checkpoint = torch.load(arguments.checkpoint, map_location=device)
    model, configuration, classes = model_from_checkpoint(checkpoint, device)
    model.eval()
    transform = build_eval_transform(configuration.get("image_size", 224))
    image_pil = Image.open(arguments.image).convert("RGB")
    landmark_pil = Image.open(arguments.landmark).convert("L")
    image, landmark = transform(image_pil, landmark_pil)
    image_batch = image.unsqueeze(0).to(device)
    landmark_batch = landmark.unsqueeze(0).to(device)
    stage_index = {"fine": 0, "mid": 1, "global": 2}[arguments.stage]
    activations = {}
    gradients = {}

    def forward_hook(module, inputs, output):
        activations["value"] = output
        output.register_hook(lambda gradient: gradients.__setitem__("value", gradient))

    handle = model.se_blocks[stage_index].register_forward_hook(forward_hook)
    logits = model(image_batch, landmark_batch)
    target_class = int(logits.argmax(dim=1).item()) if arguments.target_class is None else arguments.target_class
    model.zero_grad(set_to_none=True)
    logits[0, target_class].backward()
    handle.remove()
    activation = activations["value"][0]
    gradient = gradients["value"][0]
    weights = gradient.mean(dim=(1, 2), keepdim=True)
    cam = F.relu((weights * activation).sum(dim=0, keepdim=True))
    cam = F.interpolate(
        cam.unsqueeze(0),
        size=image.shape[-2:],
        mode="bilinear",
        align_corners=False
    )[0, 0]
    cam = normalize_map(cam).detach().cpu().numpy()
    display_image = denormalize(image).clamp(0, 1).permute(1, 2, 0).numpy()
    landmark_display = landmark[0].numpy()
    figure = plt.figure(figsize=(12, 4))
    axis1 = figure.add_subplot(1, 3, 1)
    axis1.imshow(display_image)
    axis1.set_title("Input")
    axis1.axis("off")
    axis2 = figure.add_subplot(1, 3, 2)
    axis2.imshow(landmark_display, cmap="gray")
    axis2.set_title("Landmark heatmap")
    axis2.axis("off")
    axis3 = figure.add_subplot(1, 3, 3)
    axis3.imshow(display_image)
    axis3.imshow(cam, cmap="jet", alpha=0.45)
    axis3.set_title(f"{arguments.stage} Grad-CAM: {classes[target_class]}")
    axis3.axis("off")
    figure.tight_layout()
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
