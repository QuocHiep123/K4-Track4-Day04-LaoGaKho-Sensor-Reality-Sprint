from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.generate_degradation import apply_degradation, create_road_scene
from src.health_score import TemporalHealthSmoother, evaluate_frame


class HealthScoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = create_road_scene(seed=123, night=False)

    def test_blur_lowers_sharpness_and_health(self) -> None:
        rng = np.random.default_rng(1)
        clean = evaluate_frame(self.image, "day")
        blurred = evaluate_frame(apply_degradation(self.image, "blur", 4, rng), "day")
        self.assertLess(blurred.components["sharpness"], clean.components["sharpness"])
        self.assertLess(blurred.health_score, clean.health_score)

    def test_underexposure_lowers_exposure_quality(self) -> None:
        rng = np.random.default_rng(2)
        clean = evaluate_frame(self.image, "day")
        dark = evaluate_frame(apply_degradation(self.image, "underexposure", 4, rng), "day")
        self.assertLess(dark.components["exposure"], clean.components["exposure"])
        self.assertIn("underexposure", dark.reasons)

    def test_noise_lowers_noise_quality(self) -> None:
        rng = np.random.default_rng(3)
        clean = evaluate_frame(self.image, "day")
        noisy = evaluate_frame(apply_degradation(self.image, "noise", 4, rng), "day")
        self.assertLess(noisy.components["noise"], clean.components["noise"])
        self.assertIn("high_noise", noisy.reasons)

    def test_night_context_improves_valid_night_exposure(self) -> None:
        night_image = create_road_scene(seed=31, night=True)
        fixed_day = evaluate_frame(night_image, "day")
        context_aware = evaluate_frame(night_image, "night")
        self.assertGreater(context_aware.components["exposure"], fixed_day.components["exposure"])
        self.assertGreater(context_aware.health_score, fixed_day.health_score)

    def test_grayscale_and_invalid_mode(self) -> None:
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.assertGreaterEqual(evaluate_frame(gray, "auto").health_score, 0)
        with self.assertRaises(ValueError):
            evaluate_frame(self.image, "sunset")  # type: ignore[arg-type]

    def test_temporal_smoother_uses_median(self) -> None:
        smoother = TemporalHealthSmoother(window_size=3)
        smoother.update(90)
        smoother.update(88)
        value = smoother.update(10)
        self.assertEqual(value, 88.0)


if __name__ == "__main__":
    unittest.main()
