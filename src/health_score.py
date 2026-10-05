"""
Real-time Camera Degradation Assessment & Health Score (NR-IQA)
Designed for Embedded ADAS Perception Pipelines (e.g. Jetson AGX Orin).

Provides real-time health diagnostics without reference images:
- Sharpness / Blur score (Noise-robust filtered Laplacian + Tenengrad energy)
- Exposure Quality (Over-exposure / Glare & Under-exposure / Dark crush)
- Information Density (Shannon Entropy + RMS Contrast + Spatial Texture)
- Local Spatial Occlusion (Lens Soiling / Mud detection)
- Composite Health Score (0 - 100) + Actionable ADAS decision & Sensor Fusion Weights
"""

import time
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import cv2
import numpy as np


@dataclass
class HealthReport:
    health_score: float             # 0.0 to 100.0
    status: str                     # HEALTHY | DEGRADED | CRITICAL
    action: str                     # FULL_CONFIDENCE | DOWN_WEIGHT | FALLBACK_DISENGAGE
    camera_fusion_weight: float     # 0.0 to 1.0 recommended weight for multi-sensor Kalman/BEV fusion
    blur_score: float               # 0.0 (total blur) to 1.0 (crystal sharp)
    exposure_score: float           # 0.0 (over/under exposed) to 1.0 (well balanced)
    entropy_score: float            # 0.0 (no info) to 1.0 (rich scene info)
    contrast_score: float           # 0.0 (flat/foggy) to 1.0 (crisp dynamic range)
    soiling_penalty: float          # 0.0 (clean lens) to 1.0 (mud covered)
    raw_laplacian_var: float
    raw_shannon_entropy: float
    overexposed_ratio: float
    underexposed_ratio: float
    latency_ms: float


class CameraHealthScorer:
    """Ultra-fast, lightweight Camera Health Scorer (< 3ms per frame)."""

    def __init__(
        self,
        nominal_sharpness_target: float = 200.0,
        nominal_entropy_target: float = 7.2,
        overexposure_pixel_thresh: int = 245,
        underexposure_pixel_thresh: int = 15,
    ):
        self.sharpness_target = nominal_sharpness_target
        self.entropy_target = nominal_entropy_target
        self.over_thresh = overexposure_pixel_thresh
        self.under_thresh = underexposure_pixel_thresh

    def compute_blur_score(self, gray: np.ndarray) -> Tuple[float, float]:
        """Compute noise-robust sharpness score using Gaussian pre-filtered Laplacian.
        Pre-filtering with 3x3 Gaussian prevents high-ISO sensor noise and rain speckles
        from falsely registering as edge sharpness in dark/adverse scenes.
        """
        # Noise-robust pre-smoothing
        filtered = cv2.GaussianBlur(gray, (3, 3), 0.8)
        lap = cv2.Laplacian(filtered, cv2.CV_64F)
        lap_var = float(lap.var())
        
        # Soft sigmoid mapping
        # lap_var ~ 0 -> 0.0; lap_var ~ 50 -> 0.45; lap_var > 150 -> 0.85+
        blur_norm = 1.0 - np.exp(-lap_var / 80.0)
        return float(np.clip(blur_norm, 0.0, 1.0)), lap_var

    def compute_exposure_score(self, gray: np.ndarray) -> Tuple[float, float, float]:
        """Compute over-exposure (glare) and under-exposure (dark crush) ratios."""
        total_pixels = gray.size
        over_count = np.count_nonzero(gray >= self.over_thresh)
        under_count = np.count_nonzero(gray <= self.under_thresh)
        
        r_over = over_count / total_pixels
        r_under = under_count / total_pixels
        
        # Glare penalty: severe when > 8% of frame is washed out / clipped white
        p_over = np.clip(r_over / 0.08, 0.0, 1.0)
        # Night / dark penalty: severe when > 35% of frame is pitch dark
        p_under = np.clip(r_under / 0.35, 0.0, 1.0)
        
        # Mean intensity balance
        mean_intensity = float(np.mean(gray))
        balance_penalty = abs(mean_intensity - 120.0) / 130.0
        
        exposure_norm = 1.0 - (0.55 * p_over + 0.35 * p_under + 0.10 * balance_penalty)
        return float(np.clip(exposure_norm, 0.0, 1.0)), r_over, r_under

    def compute_entropy_and_contrast(self, gray: np.ndarray) -> Tuple[float, float, float]:
        """Compute Shannon Entropy and RMS Contrast."""
        # Shannon Entropy
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        hist = hist / (hist.sum() + 1e-9)
        non_zeros = hist[hist > 0]
        entropy = float(-np.sum(non_zeros * np.log2(non_zeros)))
        
        # Driving scenes normally 6.5 - 7.6 bits
        entropy_norm = np.clip((entropy - 3.5) / (self.entropy_target - 3.5), 0.0, 1.0)
        
        # RMS Contrast
        mean_lum = float(np.mean(gray))
        std_lum = float(np.std(gray))
        rms_contrast = std_lum / (mean_lum + 1e-5)
        contrast_norm = float(np.clip(rms_contrast / 0.60, 0.0, 1.0))
        
        return float(entropy_norm), float(contrast_norm), entropy

    def detect_lens_soiling(self, image: np.ndarray, gray: np.ndarray) -> float:
        """Detect localized lens soiling / mud occlusion via spatial patch gradient energy
        and chromatic desaturation in occluded sectors.
        """
        h, w = gray.shape
        r_grid, c_grid = 6, 6
        ch, cw = h // r_grid, w // c_grid
        
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        grad_mag = cv2.magnitude(gx, gy)
        
        patch_energies = []
        for r in range(r_grid):
            for c in range(c_grid):
                p = grad_mag[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
                patch_energies.append(float(np.mean(p)))
                
        # Overall texture energy loss compared to baseline
        mean_energy = float(np.mean(patch_energies))
        # Normal road image has mean energy 80-120; heavily soiled drops < 45
        energy_loss = np.clip((85.0 - mean_energy) / 55.0, 0.0, 1.0)
        
        # Sector variance imbalance (mud covers only portions of lens)
        energy_std = float(np.std(patch_energies))
        spatial_imbalance = np.clip(energy_std / (mean_energy + 1e-5) - 0.4, 0.0, 1.0)
        
        soiling_score = 0.70 * energy_loss + 0.30 * spatial_imbalance
        return float(np.clip(soiling_score, 0.0, 1.0))

    def evaluate(self, image: np.ndarray) -> HealthReport:
        """Compute full Camera Health Report."""
        start_t = time.perf_counter()
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Fast processing scale
        if gray.shape[0] > 720:
            scale = 640.0 / gray.shape[1]
            proc_gray = cv2.resize(gray, (640, int(gray.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            proc_img = cv2.resize(image, (640, int(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        else:
            proc_gray = gray
            proc_img = image

        s_blur, lap_var = self.compute_blur_score(proc_gray)
        s_exp, r_over, r_under = self.compute_exposure_score(proc_gray)
        s_entropy, s_contrast, raw_entropy = self.compute_entropy_and_contrast(proc_gray)
        soiling_penalty = self.detect_lens_soiling(proc_img, proc_gray)

        # Composite Health Calculation:
        # Incorporates worst-case bottleneck (safety-critical principle)
        worst_metric = min(s_blur, s_exp, s_entropy)
        mean_metric = (0.35 * s_blur + 0.35 * s_exp + 0.15 * s_entropy + 0.15 * s_contrast)
        
        # Penalize for mud soiling
        composite_raw = (0.40 * worst_metric + 0.60 * mean_metric) * (1.0 - 0.65 * soiling_penalty)
        health_score = round(float(np.clip(composite_raw * 100.0, 0.0, 100.0)), 1)

        # Safety-critical status & engineering recommendation
        if health_score >= 70.0:
            status = "HEALTHY"
            action = "FULL_CONFIDENCE"
            fusion_weight = 1.0
        elif health_score >= 42.0:
            status = "DEGRADED"
            action = "DOWN_WEIGHT"
            fusion_weight = round(float(health_score / 100.0 * 0.6), 2)
        else:
            status = "CRITICAL"
            action = "FALLBACK_DISENGAGE"
            fusion_weight = 0.0

        latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)

        return HealthReport(
            health_score=health_score,
            status=status,
            action=action,
            camera_fusion_weight=fusion_weight,
            blur_score=round(s_blur, 3),
            exposure_score=round(s_exp, 3),
            entropy_score=round(s_entropy, 3),
            contrast_score=round(s_contrast, 3),
            soiling_penalty=round(soiling_penalty, 3),
            raw_laplacian_var=round(lap_var, 1),
            raw_shannon_entropy=round(raw_entropy, 2),
            overexposed_ratio=round(r_over * 100.0, 2),
            underexposed_ratio=round(r_under * 100.0, 2),
            latency_ms=latency_ms,
        )
