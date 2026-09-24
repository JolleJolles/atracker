"""Focused regression checks; no video files or GUI are needed."""

import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from atracker.helpers.detection_settings import aspect_limits, gamma_correct
from atracker.helpers.detection import ProcessImage
from atracker.helpers.filters import estimate_flicker_baseline, filter_contour_shape


class MemoryCapture:
    def __init__(self, frames, position=1):
        self.frames = frames
        self.position = position

    def get(self, key):
        return len(self.frames) if key == cv2.CAP_PROP_FRAME_COUNT else self.position

    def set(self, key, value):
        self.position = int(value)

    def read(self):
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position].copy()
        self.position += 1
        return True, frame


class DetectionFiltersTest(unittest.TestCase):
    def test_aspect_precedence_and_validation(self):
        config = SimpleNamespace(min_aspect_ratio=1.1, max_aspect_ratio=8)
        self.assertEqual(aspect_limits({}), (1.4, 10))
        self.assertEqual(aspect_limits({}, config), (1.1, 8))
        self.assertEqual(aspect_limits({"min_aspect_ratio": 2}, config), (2, 8))
        for settings in ({"min_aspect_ratio": 11}, {"max_aspect_ratio": float("nan")}):
            with self.assertRaises(ValueError):
                aspect_limits(settings)

    def test_gamma_default_and_source_immutability(self):
        source = np.array([[[0, 25, 255]]], dtype=np.uint8)
        original = source.copy()
        np.testing.assert_array_equal(gamma_correct(source), original)
        corrected = gamma_correct(source, 2)
        self.assertGreater(corrected[0, 0, 1], original[0, 0, 1])
        self.assertEqual(corrected[0, 0, 0], 0)
        self.assertEqual(corrected[0, 0, 2], 255)
        np.testing.assert_array_equal(source, original)

    def test_flicker_uses_detection_gamma_and_preserves_position(self):
        background = np.full((8, 8, 3), (30, 60, 100), dtype=np.uint8)
        frame = np.full((8, 8, 3), (25, 50, 90), dtype=np.uint8)
        for gamma in (1.0, 2.5):
            cap = MemoryCapture([frame, frame, frame])
            cutoff = estimate_flicker_baseline(cap, background, (0, 0), (8, 8),
                                              multiplier=2, gamma=gamma)
            expected = cv2.cvtColor(cv2.subtract(gamma_correct(background, gamma),
                                                 gamma_correct(frame, gamma)), cv2.COLOR_BGR2GRAY).mean() * 2
            self.assertAlmostEqual(cutoff, expected)
            self.assertEqual(cap.position, 1)
            normal = ProcessImage(frame, background, gamma=gamma, flicker_threshold=cutoff)
            normal.preprocess_bw_mode()
            self.assertFalse(normal.flicker)
            dark = ProcessImage(np.zeros_like(frame), background, gamma=gamma, flicker_threshold=cutoff)
            dark.preprocess_bw_mode()
            self.assertTrue(dark.flicker)
            self.assertFalse(dark.img_thresh.any())

    def test_size_filter_warmup_and_relative_limits(self):
        self.assertEqual(filter_contour_shape([0], [1000], 1, {0: [100] * 9}), [True])
        for area, expected in ((24, False), (25, True), (400, True), (401, False)):
            self.assertEqual(filter_contour_shape([0], [area], 1, {0: [100] * 10}), [expected])


if __name__ == "__main__":
    unittest.main()
