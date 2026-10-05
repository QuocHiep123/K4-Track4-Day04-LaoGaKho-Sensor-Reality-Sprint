"""
Generate realistic synthetic sensor degradations:
- Gaussian / Motion Blur (optical defocus, vehicle vibration)
- Underexposure (night / tunnel entry without auto-gain)
- Overexposure / Glare (sun glare, high beams, tunnel exit)
- Gaussian Noise (low-light CMOS sensor noise)
- Fog / Low-contrast (scattering, lens condensation)
"""
import numpy as np
import cv2


def apply_blur(image, sigma):
    """Gaussian blur with kernel size proportional to sigma."""
    if sigma <= 0:
        return image.copy()
    ksize = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def apply_exposure(image, factor):
    """Multiply pixel values by factor, clipping to [0, 255]."""
    img_float = image.astype(np.float32) * factor
    return np.clip(img_float, 0, 255).astype(np.uint8)


def apply_noise(image, sigma):
    """Add zero-mean Gaussian noise."""
    if sigma <= 0:
        return image.copy()
    noise = np.random.normal(0, sigma, image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_fog_low_contrast(image, intensity):
    """Blend image with uniform light gray to simulate atmospheric scattering / fog."""
    if intensity <= 0:
        return image.copy()
    fog_layer = np.full_like(image, 210, dtype=np.uint8)
    return cv2.addWeighted(image, 1.0 - intensity, fog_layer, intensity, 0)


def create_synthetic_road_scene():
    """
    Generate a deterministic, photo-style synthetic dashcam road scene.
    """
    height, width = 480, 640
    horizon = 205
    rng = np.random.default_rng(24)

    # A soft sky gradient with low-frequency cloud texture.
    y = np.linspace(0.0, 1.0, horizon, dtype=np.float32)[:, None, None]
    sky_top = np.array([188, 151, 104], dtype=np.float32)
    sky_horizon = np.array([222, 203, 171], dtype=np.float32)
    sky = sky_top + (sky_horizon - sky_top) * y
    sky = np.broadcast_to(sky, (horizon, width, 3)).copy()
    cloud_noise = rng.normal(0, 1, (30, 40)).astype(np.float32)
    cloud_noise = cv2.resize(cloud_noise, (width, horizon), interpolation=cv2.INTER_CUBIC)
    cloud_noise = cv2.GaussianBlur(cloud_noise, (0, 0), 13)
    sky += cloud_noise[:, :, None] * np.array([2.0, 2.8, 3.2], dtype=np.float32)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:horizon] = np.clip(sky, 0, 255).astype(np.uint8)

    # Distant tree-covered hills soften the horizon.
    hill_points = np.array(
        [[0, 210], [0, 176], [65, 160], [125, 183], [200, 153],
         [280, 179], [370, 157], [470, 184], [555, 159], [639, 180], [639, 220]],
        dtype=np.int32,
    )
    cv2.fillPoly(img, [hill_points], (76, 105, 72))
    cv2.line(img, (0, horizon), (width - 1, horizon), (143, 153, 130), 2)

    # Grass verges and tree masses on both sides of the road.
    img[horizon:] = (62, 104, 64)
    left_woods = np.array([[0, 182], [170, 198], [238, 267], [165, 291], [0, 300]], dtype=np.int32)
    right_woods = np.array([[465, 201], [639, 177], [639, 305], [500, 285]], dtype=np.int32)
    cv2.fillPoly(img, [left_woods], (48, 84, 48))
    cv2.fillPoly(img, [right_woods], (52, 91, 52))
    for side in (0, 1):
        for _ in range(95):
            if side == 0:
                x = int(rng.integers(-15, 222))
            else:
                x = int(rng.integers(430, 655))
            cy = int(rng.integers(183, 273))
            radius = int(rng.integers(5, 20))
            shade = int(rng.integers(-14, 15))
            green = (max(0, 47 + shade), max(0, 84 + shade), max(0, 45 + shade))
            cv2.circle(img, (x, cy), radius, green, -1, cv2.LINE_AA)

    # Asphalt road with a perspective mask and subtle grain.
    road_mask = np.zeros((height, width), dtype=np.uint8)
    road_polygon = np.array([[269, horizon + 5], [371, horizon + 5], [width, height], [0, height]], dtype=np.int32)
    cv2.fillPoly(road_mask, [road_polygon], 255)
    road_noise = rng.normal(0, 5, (height, width, 1)).astype(np.float32)
    road_base = np.full((height, width, 3), (57, 61, 64), dtype=np.float32)
    road = np.clip(road_base + road_noise, 0, 255).astype(np.uint8)
    img[road_mask > 0] = road[road_mask > 0]

    # Shoulder strips follow the road edges toward the camera.
    cv2.line(img, (266, 211), (0, 479), (111, 112, 91), 5, cv2.LINE_AA)
    cv2.line(img, (374, 211), (639, 479), (111, 112, 91), 5, cv2.LINE_AA)
    cv2.line(img, (273, 211), (35, 479), (211, 205, 176), 2, cv2.LINE_AA)
    cv2.line(img, (367, 211), (605, 479), (211, 205, 176), 2, cv2.LINE_AA)

    # Broken center line; dash length and width increase toward the camera.
    for top_y, bottom_y, line_width in [(220, 230, 2), (244, 263, 3), (282, 313, 5),
                                        (340, 385, 8), (423, 479, 12)]:
        top_offset = int((top_y - horizon) * 0.04)
        bottom_offset = int((bottom_y - horizon) * 0.04)
        dash = np.array(
            [[320 - top_offset, top_y], [320 + top_offset, top_y],
             [320 + bottom_offset, bottom_y], [320 - bottom_offset, bottom_y]],
            dtype=np.int32,
        )
        cv2.fillPoly(img, [dash], (206, 207, 194), cv2.LINE_AA)

    # Small sedan ahead, shaded to read as a vehicle rather than a flat rectangle.
    cv2.ellipse(img, (320, 295), (55, 12), 0, 0, 360, (27, 29, 30), -1, cv2.LINE_AA)
    car_body = np.array([[279, 267], [291, 248], [305, 239], [337, 239],
                         [351, 250], [361, 273], [355, 294], [285, 294]], dtype=np.int32)
    cv2.fillPoly(img, [car_body], (53, 74, 104), cv2.LINE_AA)
    cv2.line(img, (284, 278), (357, 278), (104, 120, 139), 2, cv2.LINE_AA)
    rear_window = np.array([[301, 250], [309, 242], [334, 242], [344, 252], [346, 267], [298, 267]], dtype=np.int32)
    cv2.fillPoly(img, [rear_window], (57, 72, 78), cv2.LINE_AA)
    cv2.line(img, (305, 250), (340, 250), (119, 130, 129), 1, cv2.LINE_AA)
    cv2.ellipse(img, (291, 281), (5, 4), 0, 0, 360, (38, 45, 207), -1, cv2.LINE_AA)
    cv2.ellipse(img, (349, 281), (5, 4), 0, 0, 360, (38, 45, 207), -1, cv2.LINE_AA)

    # A roadside sign and post, kept small and distant in the scene.
    cv2.line(img, (504, 202), (504, 271), (96, 95, 83), 3, cv2.LINE_AA)
    cv2.rectangle(img, (493, 171), (515, 196), (180, 191, 174), -1, cv2.LINE_AA)
    cv2.rectangle(img, (496, 174), (512, 193), (62, 104, 68), -1, cv2.LINE_AA)

    # Gentle lens vignette and restrained sensor grain.
    yy, xx = np.mgrid[0:height, 0:width]
    radius = ((xx - width / 2) ** 2 + (yy - height / 2) ** 2) / (width * height / 2)
    vignette = np.clip(1.0 - 0.12 * radius, 0.78, 1.0).astype(np.float32)
    img = np.clip(img.astype(np.float32) * vignette[:, :, None], 0, 255)
    img += rng.normal(0, 1.2, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)
