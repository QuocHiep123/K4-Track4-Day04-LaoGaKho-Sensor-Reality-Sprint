"""
Camera Degradation Simulator for Autonomous Driving (nuScenes-C / KITTI-C standard)
Supports 5 corruption types at 5 severity levels:
1. Motion/Defocus Blur (blur)
2. Sun Glare / Over-exposure (glare)
3. Night / Low-light Underexposure + Sensor Noise (darkness)
4. Heavy Rain Streaks + Fog Veil (rain)
5. Lens Soiling / Mud & Dirt Splatters (soiling)
"""

import cv2
import numpy as np


def apply_blur(image: np.ndarray, severity: int = 3) -> np.ndarray:
    """Apply motion and defocus blur with increasing severity (1 to 5)."""
    severity = max(1, min(5, severity))
    kernel_sizes = [5, 11, 21, 35, 51]
    k = kernel_sizes[severity - 1]
    
    # Combined motion blur kernel + gaussian defocus
    motion_kernel = np.zeros((k, k), dtype=np.float32)
    # 45 degree motion angle (vehicle moving forward or camera vibration)
    np.fill_diagonal(motion_kernel, 1.0)
    motion_kernel = cv2.GaussianBlur(motion_kernel, (3, 3), 0)
    motion_kernel /= motion_kernel.sum()
    
    blurred = cv2.filter2D(image, -1, motion_kernel)
    if severity >= 3:
        blurred = cv2.GaussianBlur(blurred, (k // 2 * 2 + 1, k // 2 * 2 + 1), sigmaX=severity * 1.5)
    return blurred


def apply_glare(image: np.ndarray, severity: int = 3) -> np.ndarray:
    """Apply direct sun glare / high-beam glare with radial blooming and washout."""
    severity = max(1, min(5, severity))
    h, w = image.shape[:2]
    
    # Center of glare located in upper-center (typical sun / oncoming truck headlight)
    cx, cy = int(w * 0.55), int(h * 0.3)
    
    y, x = np.ogrid[:h, :w]
    dist_sq = (x - cx) ** 2 + (y - cy) ** 2
    max_radius = (h + w) * 0.35 * (severity / 5.0)
    
    glare_mask = np.exp(-dist_sq / (2 * (max_radius ** 2)))
    glare_mask = np.clip(glare_mask * (0.4 + 0.15 * severity), 0, 1)[..., np.newaxis]
    
    # Ambient contrast washout + intense white core
    washed = image.astype(np.float32) * (1.0 + 0.15 * severity) + 30 * severity
    glared = washed * (1 - glare_mask) + 255.0 * glare_mask
    return np.clip(glared, 0, 255).astype(np.uint8)


def apply_darkness(image: np.ndarray, severity: int = 3) -> np.ndarray:
    """Apply severe underexposure (night/tunnel) + Poisson-Gaussian sensor shot noise."""
    severity = max(1, min(5, severity))
    # Attenuation factors: 1 -> 0.7, 5 -> 0.08
    attenuation = [0.75, 0.5, 0.3, 0.18, 0.08][severity - 1]
    
    img_f = image.astype(np.float32) * attenuation
    
    # High ISO sensor noise under low light
    noise_sigma = severity * 6.0
    noise = np.random.normal(0, noise_sigma, image.shape).astype(np.float32)
    darkened = np.clip(img_f + noise, 0, 255).astype(np.uint8)
    return darkened


def apply_rain(image: np.ndarray, severity: int = 3) -> np.ndarray:
    """Apply heavy directional rain streaks and foggy atmospheric veil."""
    severity = max(1, min(5, severity))
    h, w = image.shape[:2]
    
    # 1. Atmospheric fog veil (reduced contrast, increased gray baseline)
    veil_intensity = 0.08 * severity
    fogged = cv2.addWeighted(image, 1.0 - veil_intensity, np.full_like(image, 200), veil_intensity, 0)
    
    # 2. Rain streak generation
    streak_count = severity * 400
    rain_layer = np.zeros((h, w), dtype=np.uint8)
    
    for _ in range(streak_count):
        x1 = np.random.randint(0, w)
        y1 = np.random.randint(0, h)
        length = np.random.randint(15, 30 + severity * 10)
        angle = np.deg2rad(75 + np.random.uniform(-5, 5))
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        thickness = 1 if severity < 4 else 2
        cv2.line(rain_layer, (x1, y1), (x2, y2), 255, thickness)
    
    rain_blur = cv2.GaussianBlur(rain_layer, (3, 3), 0)
    rain_bgr = cv2.cvtColor(rain_blur, cv2.COLOR_GRAY2BGR)
    
    rainy = cv2.addWeighted(fogged, 1.0, rain_bgr, 0.3 + 0.08 * severity, 0)
    return np.clip(rainy, 0, 255).astype(np.uint8)


def apply_soiling(image: np.ndarray, severity: int = 3) -> np.ndarray:
    """Apply lens soiling: mud drops, dirt splatters, water streaks on camera lens glass."""
    severity = max(1, min(5, severity))
    h, w = image.shape[:2]
    
    num_blobs = severity * 10
    mask = np.zeros((h, w), dtype=np.float32)
    np.random.seed(100 + severity)
    
    for _ in range(num_blobs):
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        rx = np.random.randint(25, 40 + severity * 25)
        ry = np.random.randint(20, 35 + severity * 20)
        angle = np.random.randint(0, 180)
        cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360, 1.0, -1)
        
    # Gaussian blur for soft out-of-focus optics on lens element
    blur_k = 21 + severity * 4
    if blur_k % 2 == 0:
        blur_k += 1
    mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
    opacity = min(0.98, 0.55 + 0.08 * severity)
    mask = np.clip(mask * opacity, 0, 1.0)[..., np.newaxis]
    
    # Dark brownish mud color (BGR)
    mud_bgr = np.array([40, 55, 70], dtype=np.float32)
    soiled = (image.astype(np.float32) * (1.0 - mask) + mud_bgr * mask)
    return np.clip(soiled, 0, 255).astype(np.uint8)


DEGRADATION_DISPATCH = {
    'blur': apply_blur,
    'glare': apply_glare,
    'darkness': apply_darkness,
    'rain': apply_rain,
    'soiling': apply_soiling,
}


def apply_degradation(image: np.ndarray, deg_type: str, severity: int = 3) -> np.ndarray:
    """Convenience wrapper to apply any degradation type at severity 1-5."""
    if deg_type not in DEGRADATION_DISPATCH:
        raise ValueError(f"Unknown degradation: {deg_type}. Choices: {list(DEGRADATION_DISPATCH.keys())}")
    return DEGRADATION_DISPATCH[deg_type](image, severity)
