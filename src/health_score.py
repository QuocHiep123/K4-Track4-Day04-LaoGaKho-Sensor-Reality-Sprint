"""Explainable, reference-free camera health metrics.

The constants in this module are demo calibration values for 640x360 imagery.
Real cameras should be calibrated on representative clean and failed frames.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

SceneMode = Literal["auto", "day", "night"]


@dataclass(frozen=True)
class HealthReport:
    health_score: float
    status: str
    scene_mode: str
    reasons: list[str]
    action: str
    components: dict[str, float]
    raw_metrics: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _quality_high_is_good(value: float, bad: float, good: float) -> float:
    return _clip01((value - bad) / (good - bad))


def _quality_low_is_good(value: float, good: float, bad: float) -> float:
    return _clip01((bad - value) / (bad - good))


def _infer_scene_mode(gray: np.ndarray) -> Literal["day", "night"]:
    median = float(np.median(gray))
    p90 = float(np.percentile(gray, 90))
    return "night" if median < 62.0 and p90 < 155.0 else "day"


def _noise_sigma(gray: np.ndarray) -> float:
    """Estimate additive noise using a robust 3x3 high-pass response."""
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    response = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kernel)
    if gray.shape[0] <= 2 or gray.shape[1] <= 2:
        return 0.0
    core = np.abs(response[1:-1, 1:-1])
    return float(np.sqrt(np.pi / 2.0) * np.mean(core) / 6.0)


def _entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probabilities = hist / max(float(hist.sum()), 1.0)
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def evaluate_frame(image: np.ndarray, mode: SceneMode = "auto") -> HealthReport:
    """Evaluate a BGR or grayscale uint8 image and return an explainable report."""
    if image is None or image.size == 0:
        raise ValueError("image must be a non-empty numpy array")
    if image.ndim == 2:
        gray = image.astype(np.uint8, copy=False)
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        conversion = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        gray = cv2.cvtColor(image, conversion)
    else:
        raise ValueError("image must be grayscale, BGR, or BGRA")
    if min(gray.shape[:2]) < 16:
        raise ValueError("image must be at least 16x16 pixels")
    if mode not in {"auto", "day", "night"}:
        raise ValueError("mode must be one of: auto, day, night")

    resolved_mode = _infer_scene_mode(gray) if mode == "auto" else mode
    denoised = cv2.GaussianBlur(gray, (3, 3), 0.6)
    laplacian_variance = float(cv2.Laplacian(denoised, cv2.CV_64F).var())
    median = float(np.median(gray))
    p05, p95 = (float(x) for x in np.percentile(gray, [5, 95]))
    contrast_span = p95 - p05
    dark_clip_ratio = float(np.mean(gray <= 5))
    bright_clip_ratio = float(np.mean(gray >= 250))
    # Bloom is often near-white without reaching literal 255 clipping.
    highlight_ratio = float(np.mean(gray >= 200))
    noise_sigma = _noise_sigma(gray)
    entropy = _entropy(gray)

    sharpness_q = _quality_high_is_good(
        math.log1p(laplacian_variance), math.log1p(7.0), math.log1p(180.0)
    )

    if resolved_mode == "night":
        low_luma, high_luma = 12.0, 125.0
        allowed_dark_clip = 0.78
        contrast_bad, contrast_good = 6.0, 42.0
    else:
        low_luma, high_luma = 58.0, 205.0
        allowed_dark_clip = 0.42
        contrast_bad, contrast_good = 18.0, 105.0

    if median < low_luma:
        brightness_q = _clip01(median / low_luma)
    elif median > high_luma:
        brightness_q = _clip01((255.0 - median) / (255.0 - high_luma))
    else:
        brightness_q = 1.0

    dark_penalty = _clip01(dark_clip_ratio / allowed_dark_clip)
    bright_penalty = _clip01(bright_clip_ratio / 0.22)
    # Do not penalize merely bright pixels: real road images often contain a large sky.
    # The near-white ratio is still reported for glare/overexposure analysis.
    clipping_q = _clip01(1.0 - 0.65 * dark_penalty - 0.70 * bright_penalty)
    exposure_q = math.sqrt(max(brightness_q * clipping_q, 0.0))
    contrast_q = _quality_high_is_good(contrast_span, contrast_bad, contrast_good)
    noise_q = _quality_low_is_good(noise_sigma, 2.0, 22.0)
    # Correct the classic Laplacian failure where sensor noise looks like detail.
    sharpness_q *= 0.45 + 0.55 * noise_q

    components = {
        "sharpness": round(sharpness_q, 4),
        "exposure": round(exposure_q, 4),
        "contrast": round(contrast_q, 4),
        "noise": round(noise_q, 4),
    }
    weights = {"sharpness": 0.40, "exposure": 0.30, "contrast": 0.15, "noise": 0.15}
    health = 100.0 * math.exp(
        sum(weights[name] * math.log(max(components[name], 0.02)) for name in weights)
    )
    health = round(float(np.clip(health, 0.0, 100.0)), 2)

    reasons: list[str] = []
    if sharpness_q < 0.55:
        reasons.append("blur_or_low_texture")
    if exposure_q < 0.55:
        reasons.append("underexposure" if median < low_luma else "overexposure")
    if contrast_q < 0.45:
        reasons.append("low_contrast")
    if noise_q < 0.50:
        reasons.append("high_noise")

    if health >= 70.0:
        status, action = "HEALTHY", "use_camera_normally"
    elif health >= 40.0:
        status, action = "DEGRADED", "down_weight_camera"
    else:
        status, action = "CRITICAL", "fallback_or_safe_mode"

    raw_metrics = {
        "laplacian_variance": round(laplacian_variance, 4),
        "median_luminance": round(median, 4),
        "p05_luminance": round(p05, 4),
        "p95_luminance": round(p95, 4),
        "contrast_span": round(contrast_span, 4),
        "dark_clip_ratio": round(dark_clip_ratio, 6),
        "bright_clip_ratio": round(bright_clip_ratio, 6),
        "highlight_ratio": round(highlight_ratio, 6),
        "estimated_noise_sigma": round(noise_sigma, 4),
        "entropy_bits": round(entropy, 4),
    }
    return HealthReport(
        health_score=health,
        status=status,
        scene_mode=resolved_mode,
        reasons=reasons,
        action=action,
        components=components,
        raw_metrics=raw_metrics,
    )


class TemporalHealthSmoother:
    """Median smoother for video scores and stable state transitions."""

    def __init__(self, window_size: int = 7):
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self._scores: deque[float] = deque(maxlen=window_size)

    def update(self, score: float) -> float:
        self._scores.append(float(np.clip(score, 0.0, 100.0)))
        return round(float(np.median(self._scores)), 2)

    def reset(self) -> None:
        self._scores.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate camera degradation health")
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument("--mode", choices=["auto", "day", "night"], default="auto")
    parser.add_argument("--compact", action="store_true", help="Print compact JSON")
    args = parser.parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    report = evaluate_frame(image, args.mode)
    print(json.dumps(report.to_dict(), indent=None if args.compact else 2, ensure_ascii=False))


if __name__ == "__main__":
    main()
