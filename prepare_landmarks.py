import argparse
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def gaussian_heatmap(height, width, points, sigma):
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    heatmap = np.zeros((height, width), dtype=np.float32)
    denominator = 2.0 * sigma * sigma
    for x, y in points:
        distance = (x_grid - x) ** 2 + (y_grid - y) ** 2
        heatmap = np.maximum(heatmap, np.exp(-distance / denominator))
    return np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)


def process_image(face_mesh, source_path, output_path, sigma):
    image = cv2.imread(str(source_path))
    if image is None:
        raise ValueError(f"Unable to read {source_path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    height, width = image.shape[:2]
    points = []
    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark
        points = [
            (
                min(max(int(point.x * width), 0), width - 1),
                min(max(int(point.y * height), 0), height - 1)
            )
            for point in landmarks
        ]
    heatmap = gaussian_heatmap(height, width, points, sigma)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(heatmap).save(output_path)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    image_root = Path(arguments.image_root)
    output_root = Path(arguments.output_root)
    paths = [path for path in image_root.rglob("*") if path.suffix.lower() in SUPPORTED_EXTENSIONS]
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=arguments.min_detection_confidence
    ) as face_mesh:
        for index, source_path in enumerate(paths, start=1):
            relative_path = source_path.relative_to(image_root).with_suffix(".png")
            output_path = output_root / relative_path
            process_image(face_mesh, source_path, output_path, arguments.sigma)
            print(f"{index}/{len(paths)} {relative_path}")


if __name__ == "__main__":
    main()
