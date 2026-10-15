"""Known and interactive calibration must produce reusable measurement CSVs."""
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

measure = importlib.import_module("atracker.measure")


class MeasureImagesTest(unittest.TestCase):
    def test_known_scale_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "a.png").touch()
            output = root / "measurements.csv"
            with patch.object(measure, "annotation_gui", return_value=("polygon", [(0, 0), (30, 0)], 30)) as gui:
                result = measure.measure_images(d, px_per_mm=10, variables=["length"], out_csv=str(output))
                self.assertEqual(result.loc[0, "length_length_mm"], 3)
                measure.measure_images(d, px_per_mm=10, variables=["length"], out_csv=str(output))
                self.assertEqual(gui.call_count, 1)
            self.assertEqual(len(pd.read_csv(output)), 1)

    def test_first_image_calibration_and_measurement(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.png").touch()
            replies = [("polygon", [(0, 0), (100, 0)], 100),
                       ("polygon", [(0, 0), (40, 0)], 40)]
            with patch.object(measure, "annotation_gui", side_effect=replies):
                result = measure.measure_images(d, conversion_image_mm=10, skip_first=False, variables=["width"])
            self.assertEqual(result.loc[0, "width_length_mm"], 4)


if __name__ == "__main__":
    unittest.main()
