"""
Camera Degradation Health Score calculation for ADAS perception.
Reference-free, fast, interpretable metrics.
"""
import numpy as np
import cv2


def compute_sharpness_score(gray_img):
    """
    Sharpness via Variance of Laplacian.
    Lower variance = more blur.
    Returns: raw_var, q_sharpness in [0, 1]
    """
    laplacian_var = float(cv2.Laplacian(gray_img, cv2.CV_64F).var())
    # Sigmoid-like soft clipping mapped to [0, 1]
    # Calibration target: 100-500 var is typically sharp for driving camera
    # var < 20 is severe blur, var > 200 is healthy
    q_sharp = float(np.clip((laplacian_var - 15.0) / (250.0 - 15.0), 0.0, 1.0))
    return laplacian_var, q_sharp


def compute_exposure_score(gray_img):
    """
    Exposure quality via under/over-exposure pixel clipping ratios.
    Returns: under_ratio, over_ratio, q_exposure in [0, 1]
    """
    total_pixels = float(gray_img.size)
    under_ratio = float(np.sum(gray_img < 15)) / total_pixels
    over_ratio = float(np.sum(gray_img > 240)) / total_pixels

    # Excessive dark or bright pixels penalize the score
    # Normal driving scene tolerates up to 10% dark/bright without big issue
    penalty = (under_ratio * 1.5) + (over_ratio * 2.0)
    q_exposure = float(np.clip(1.0 - penalty, 0.0, 1.0))
    return under_ratio, over_ratio, q_exposure


def compute_contrast_score(gray_img):
    """
    Contrast quality via percentile spread P95 - P5.
    Fog/rain/flat lens yields small range.
    Returns: contrast_range, q_contrast in [0, 1]
    """
    p5 = float(np.percentile(gray_img, 5))
    p95 = float(np.percentile(gray_img, 95))
    contrast_range = p95 - p5

    # Healthy scene has dynamic range > 120. Range < 40 indicates severe loss.
    q_contrast = float(np.clip((contrast_range - 30.0) / (130.0 - 30.0), 0.0, 1.0))
    return contrast_range, q_contrast


def compute_noise_proxy(gray_img):
    """
    Noise level estimation via residual of mild Gaussian smoothing.
    High residual with low structural contrast = sensor noise.
    """
    denoised = cv2.GaussianBlur(gray_img, (3, 3), 0)
    residual = cv2.absdiff(gray_img, denoised)
    noise_sigma_est = float(np.mean(residual))
    # Typical noise: < 2.0 is clean, > 8.0 is heavy sensor noise
    q_noise = float(np.clip(1.0 - (noise_sigma_est - 1.5) / 8.0, 0.0, 1.0))
    return noise_sigma_est, q_noise


def evaluate_camera_health(image_bgr):
    """
    Main evaluation pipeline.
    Input: image_bgr (H, W, 3) or grayscale (H, W)
    Output: dict with score, status, components, reasons, action
    """
    if len(image_bgr.shape) == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr

    lap_var, q_blur = compute_sharpness_score(gray)
    under_r, over_r, q_exp = compute_exposure_score(gray)
    c_range, q_contrast = compute_contrast_score(gray)
    noise_est, q_noise = compute_noise_proxy(gray)

    # Weighted blend (40% sharpness, 30% exposure, 20% contrast, 10% noise)
    raw_health = (
        0.40 * q_blur
        + 0.30 * q_exp
        + 0.20 * q_contrast
        + 0.10 * q_noise
    ) * 100.0

    health_score = round(float(np.clip(raw_health, 0.0, 100.0)), 1)

    reasons = []
    if q_blur < 0.4:
        reasons.append("blur_or_defocus")
    if under_r > 0.35:
        reasons.append("severe_underexposure")
    elif under_r > 0.18:
        reasons.append("mild_underexposure")
    if over_r > 0.25:
        reasons.append("glare_or_overexposure")
    if q_contrast < 0.4:
        reasons.append("low_contrast_fog_rain")
    if q_noise < 0.4:
        reasons.append("high_sensor_noise")

    if not reasons:
        reasons.append("nominal")

    # Status & engineering decision mapping
    if health_score >= 70.0:
        status = "HEALTHY"
        action = "use_camera_full_confidence"
    elif health_score >= 40.0:
        status = "DEGRADED"
        action = "down_weight_camera_increase_lidar_radar"
    else:
        status = "CRITICAL"
        action = "fallback_drop_camera_trigger_alert"

    return {
        "health_score": health_score,
        "status": status,
        "action": action,
        "reasons": reasons,
        "components": {
            "q_sharpness": round(q_blur, 3),
            "q_exposure": round(q_exp, 3),
            "q_contrast": round(q_contrast, 3),
            "q_noise": round(q_noise, 3),
        },
        "raw_measurements": {
            "laplacian_variance": round(lap_var, 2),
            "underexposed_ratio": round(under_r, 4),
            "overexposed_ratio": round(over_r, 4),
            "contrast_p95_p5": round(c_range, 2),
            "noise_sigma_est": round(noise_est, 2),
        }
    }
