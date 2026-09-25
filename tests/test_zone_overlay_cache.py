"""Zone raster caching must preserve geometry changes and frame contents."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from atracker.visualise import Visualiser


class ZoneOverlayCacheTest(unittest.TestCase):
    def setUp(self):
        self.zones = {
            1: [[(-5, -5), (50, -5), (50, 50), (-5, 50)],
                [(70, 70), (90, 70), (90, 90), (70, 90)]],
            2: [(30, 30), (60, 30), (60, 60), (30, 60)],
        }
        self.config = SimpleNamespace(vis=SimpleNamespace(draw_zones=True))
        self.vis = Visualiser(zone_coords=self.zones, config=self.config)

    def assert_matches_fresh(self, shape=(100, 100, 3), brightness=50):
        actual = np.full(shape, brightness, np.uint8)
        expected = actual.copy()
        self.vis.draw_scene_overlays(actual)
        Visualiser(zone_coords=self.zones, config=self.config).draw_scene_overlays(expected)
        np.testing.assert_array_equal(actual, expected)

    def test_cached_frames_do_not_repeat_distance_transform(self):
        self.assert_matches_fresh()
        with patch("atracker.visualise.cv2.distanceTransform", side_effect=AssertionError):
            image = np.full((100, 100, 3), 180, np.uint8)
            self.vis.draw_scene_overlays(image)
        expected = np.full((100, 100, 3), 180, np.uint8)
        Visualiser(zone_coords=self.zones, config=self.config).draw_scene_overlays(expected)
        np.testing.assert_array_equal(image, expected)

    def test_invalidates_on_geometry_order_and_size_changes(self):
        self.assert_matches_fresh()
        self.zones[1][0][1] = (65, -5)
        self.assert_matches_fresh()
        self.zones[1] = self.zones.pop(1)
        self.assert_matches_fresh()
        self.assert_matches_fresh(shape=(70, 80, 3))
        self.zones.clear()
        self.assert_matches_fresh()

    def test_overlap_has_single_opacity_and_last_zone_colour(self):
        image = np.full((100, 100, 3), 100, np.uint8)
        self.vis.draw_scene_overlays(image)
        # Away from outlines/text: zone 2 overlays zone 1 once.
        expected = cv2.addWeighted(np.array([[[255, 255, 0]]], np.uint8), .18,
                                   np.full((1, 1, 3), 100, np.uint8), .82, 0)
        np.testing.assert_array_equal(image[35, 35], expected[0, 0])


if __name__ == "__main__":
    unittest.main()
