"""Create deterministic road scenes and controlled camera degradations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SEVERITY_LEVELS = range(5)


def _vertical_gradient(height: int, width: int, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> np.ndarray:
    alpha = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    top_color = np.array(top, dtype=np.float32)[None, None, :]
    bottom_color = np.array(bottom, dtype=np.float32)[None, None, :]
    return np.repeat(top_color * (1.0 - alpha) + bottom_color * alpha, width, axis=1).astype(np.uint8)


def create_road_scene(seed: int, night: bool = False, size: tuple[int, int] = (640, 360)) -> np.ndarray:
    """Draw a reproducible, textured road scene without external assets."""
    width, height = size
    rng = np.random.default_rng(seed)
    horizon = int(height * 0.48)
    if night:
        image = _vertical_gradient(height, width, (28, 16, 8), (75, 52, 35))
        sky_bottom = (62, 42, 28)
        ground_color = (24, 24, 28)
    else:
        image = _vertical_gradient(height, width, (235, 185, 105), (245, 225, 190))
        sky_bottom = (235, 205, 150)
        ground_color = (55, 95, 65)
    image[:horizon] = _vertical_gradient(horizon, width, tuple(int(x) for x in image[0, 0]), sky_bottom)
    image[horizon:] = ground_color

    # Distant skyline and trees add spatial frequencies useful for sharpness tests.
    for x in range(-20, width + 30, 38):
        tree_height = int(rng.integers(28, 62))
        color = (18, 48, 24) if not night else (10, 18, 13)
        cv2.rectangle(image, (x + 13, horizon - tree_height // 2), (x + 18, horizon + 8), color, -1)
        cv2.circle(image, (x + 15, horizon - tree_height // 2), tree_height // 2, color, -1, cv2.LINE_AA)

    road = np.array(
        [[int(width * 0.43), horizon], [int(width * 0.57), horizon], [width, height], [0, height]],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [road], (48, 50, 55) if not night else (22, 24, 29))
    cv2.line(image, (0, height - 1), (int(width * 0.43), horizon), (180, 180, 180), 3)
    cv2.line(image, (width - 1, height - 1), (int(width * 0.57), horizon), (180, 180, 180), 3)

    # Perspective lane markings.
    for y0, y1 in [(horizon + 10, horizon + 23), (horizon + 38, horizon + 62), (horizon + 85, horizon + 130)]:
        center = width // 2 + (seed % 3 - 1) * 12
        half0 = max(1, int((y0 - horizon) * 0.018))
        half1 = max(2, int((y1 - horizon) * 0.026))
        pts = np.array([[center - half0, y0], [center + half0, y0], [center + half1, y1], [center - half1, y1]])
        cv2.fillPoly(image, [pts], (225, 225, 220))

    # A lead vehicle and a roadside sign provide recognizable perception targets.
    car_x, car_y = int(width * 0.56), int(height * 0.61)
    cv2.rectangle(image, (car_x - 28, car_y - 13), (car_x + 28, car_y + 17), (35, 45, 160), -1)
    cv2.rectangle(image, (car_x - 17, car_y - 10), (car_x + 17, car_y), (120, 165, 190), -1)
    cv2.circle(image, (car_x - 19, car_y + 17), 6, (8, 8, 8), -1)
    cv2.circle(image, (car_x + 19, car_y + 17), 6, (8, 8, 8), -1)
    sign_x = int(width * 0.78)
    cv2.line(image, (sign_x, horizon + 4), (sign_x, horizon + 72), (160, 160, 160), 4)
    cv2.circle(image, (sign_x, horizon), 20, (235, 235, 235), -1, cv2.LINE_AA)
    cv2.circle(image, (sign_x, horizon), 20, (30, 45, 210), 3, cv2.LINE_AA)
    cv2.putText(image, "50", (sign_x - 13, horizon + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (20, 20, 20), 1, cv2.LINE_AA)

    if night:
        cv2.circle(image, (int(width * 0.15), int(height * 0.17)), 18, (220, 220, 185), -1, cv2.LINE_AA)
        for x, y in [(car_x - 18, car_y + 4), (car_x + 18, car_y + 4)]:
            cv2.circle(image, (x, y), 4, (180, 230, 255), -1, cv2.LINE_AA)

    texture = rng.normal(0.0, 2.2 if not night else 1.2, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + texture, 0, 255).astype(np.uint8)


def apply_degradation(image: np.ndarray, kind: str, level: int, rng: np.random.Generator) -> np.ndarray:
    if level not in SEVERITY_LEVELS:
        raise ValueError("level must be between 0 and 4")
    if level == 0:
        return image.copy()
    if kind == "blur":
        sigma = [0.0, 0.8, 1.6, 2.8, 4.2][level]
        return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if kind == "underexposure":
        factor = [1.0, 0.76, 0.53, 0.34, 0.18][level]
        darkened = image.astype(np.float32) * factor
        # Real low-light capture also introduces read/shot noise as gain increases.
        noise_sigma = [0.0, 2.0, 5.0, 10.0, 18.0][level]
        noise = rng.normal(0.0, noise_sigma, image.shape).astype(np.float32)
        return np.clip(darkened + noise, 0, 255).astype(np.uint8)
    if kind == "overexposure":
        factor = [1.0, 1.40, 1.85, 2.45, 3.20][level]
        lift = [0.0, 8.0, 20.0, 40.0, 65.0][level]
        lifted = image.astype(np.float32) * factor + lift
        return np.clip(lifted, 0, 255).astype(np.uint8)
    if kind == "noise":
        sigma = [0.0, 4.0, 10.0, 20.0, 35.0][level]
        noise = rng.normal(0.0, sigma, image.shape).astype(np.float32)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if kind == "glare":
        height, width = image.shape[:2]
        yy, xx = np.ogrid[:height, :width]
        center_x, center_y = int(width * 0.72), int(height * 0.25)
        radius = [1, 90, 135, 180, 230][level]
        distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        alpha = np.clip(1.0 - distance / radius, 0.0, 1.0) ** 1.6
        alpha = alpha[..., None] * [0.0, 0.72, 0.82, 0.91, 1.00][level]
        glare_color = np.full_like(image, (225, 245, 255), dtype=np.float32)
        result = image.astype(np.float32) * (1.0 - alpha) + glare_color * alpha
        return np.clip(result, 0, 255).astype(np.uint8)
    if kind == "fog":
        alpha = [0.0, 0.12, 0.26, 0.43, 0.62][level]
        haze = np.full_like(image, (205, 205, 205), dtype=np.float32)
        result = image.astype(np.float32) * (1.0 - alpha) + haze * alpha
        if level >= 2:
            result = cv2.GaussianBlur(result, (0, 0), sigmaX=0.28 * level)
        return np.clip(result, 0, 255).astype(np.uint8)
    raise ValueError(f"unknown degradation: {kind}")


def ensure_demo_inputs(input_dir: Path) -> list[Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if existing:
        return existing
    scenes = [
        ("day_road_01.png", create_road_scene(7, night=False)),
        ("day_road_02.png", create_road_scene(19, night=False)),
        ("night_road_01.png", create_road_scene(31, night=True)),
    ]
    for name, image in scenes:
        cv2.imwrite(str(input_dir / name), image)
    return [input_dir / name for name, _ in scenes]


def standardize_frame(image: np.ndarray, size: tuple[int, int] = (640, 360)) -> np.ndarray:
    """Resize-to-cover and center-crop so metrics are comparable across cameras."""
    target_width, target_height = size
    height, width = image.shape[:2]
    scale = max(target_width / width, target_height / height)
    resized_width = max(target_width, int(round(width * scale)))
    resized_height = max(target_height, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    x0 = (resized_width - target_width) // 2
    y0 = (resized_height - target_height) // 2
    return resized[y0:y0 + target_height, x0:x0 + target_width].copy()


def generate_dataset(input_dir: Path, output_dir: Path, synthetic_fallback: bool = False) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    input_paths = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not input_paths and synthetic_fallback:
        input_paths = ensure_demo_inputs(input_dir)
    if not input_paths:
        raise FileNotFoundError(
            f"No real input images found in {input_dir}. "
            "Run: python -m src.fetch_real_data"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    rows: list[dict[str, str | int]] = []
    kinds = ["blur", "underexposure", "overexposure", "noise", "fog"]
    for source_index, source_path in enumerate(input_paths):
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = standardize_frame(image)
        scene_mode = "night" if "night" in source_path.stem.lower() else "day"
        for kind_index, kind in enumerate(kinds):
            kind_dir = output_dir / kind
            kind_dir.mkdir(parents=True, exist_ok=True)
            for level in SEVERITY_LEVELS:
                rng = np.random.default_rng(10_000 * source_index + 100 * kind_index + level)
                degraded = apply_degradation(image, kind, level, rng)
                destination = kind_dir / f"{source_path.stem}_level_{level}.png"
                cv2.imwrite(str(destination), degraded)
                rows.append(
                    {
                        "source": source_path.name,
                        "scene_mode": scene_mode,
                        "degradation": kind,
                        "severity": level,
                        "image_path": destination.as_posix(),
                    }
                )
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "scene_mode", "degradation", "severity", "image_path"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate controlled camera degradations")
    parser.add_argument("--input-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--synthetic-fallback", action="store_true")
    args = parser.parse_args()
    manifest = generate_dataset(args.input_dir, args.output_dir, args.synthetic_fallback)
    print(f"Generated dataset manifest: {manifest}")


if __name__ == "__main__":
    main()
