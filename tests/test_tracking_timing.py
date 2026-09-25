"""Timing accounting and history-independent live orientation regression checks."""

import copy
import unittest
from unittest.mock import patch

import numpy as np

from atracker.helpers.timing import TrackingTiming
from atracker.visualise import Visualiser


class NoScanList(list):
    def __iter__(self):
        raise AssertionError("Live drawing scanned the accumulated history")


class TrackingTimingTest(unittest.TestCase):
    def test_window_accounting_and_reset(self):
        messages = []
        timer = TrackingTiming(2, emit=messages.append)
        with patch("atracker.helpers.timing.perf_counter", side_effect=[0, .01, .03, 1, 1.01, 1.03]):
            for frame in (10, 11):
                timer.start()
                timer.mark("read")
                timer.mark("queue")
                timer.frame(frame)
        self.assertEqual(len(messages), 1)
        self.assertIn("2 processed, 33.3fps", messages[0])
        self.assertIn("queue=20.0ms (67%)", messages[0])
        self.assertEqual(dict(timer.seconds), {})
        self.assertEqual(timer.frames, 0)

    def test_disabled_does_not_read_clock(self):
        with patch("atracker.helpers.timing.perf_counter", side_effect=AssertionError):
            timer = TrackingTiming()
            timer.start()
            timer.mark("read")
            timer.frame(1)
            timer.report(1)

    def test_live_orientation_matches_history_lookup_without_scanning(self):
        vis = Visualiser.__new__(Visualiser)
        vis.orient_lwidth = 1
        history = 10000
        data = dict(frame=[1] * history + [2, 2, 2, 2],
                    id=[0] * history + [0, 1, "red", "blue"],
                    cx=[10] * history + [20, 60, 25, 55],
                    cy=[10] * history + [20, 60, 25, 55],
                    angle=[None] * (history + 4))
        expected = copy.deepcopy(data)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        expected_image = image.copy()
        vis.draw_orient_link(expected_image, expected, 2)
        data["frame"] = NoScanList(data["frame"])
        data["id"] = NoScanList(data["id"])
        vis.draw_orient_link(image, data, 2, frame_data_start=history)
        self.assertEqual(data["angle"], expected["angle"])
        np.testing.assert_array_equal(image, expected_image)
        # No detections in the next frame must not reuse previous detections.
        vis.draw_orient_link(image, data, 3, frame_data_start=history + 4)
        self.assertEqual(data["angle"], expected["angle"])


if __name__ == "__main__":
    unittest.main()
