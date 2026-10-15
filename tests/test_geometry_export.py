"""Geometry exports preserve raster components, holes, calibration and regions."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Point, shape

from atracker import ATracker
from atracker.geometry import export_geometry


class GeometryExportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dirs = {key: str(self.root/key) for key in ("originals", "tracked", "processed")}
        for directory in self.dirs.values():
            Path(directory).mkdir()
        mask = np.full((80, 100), 255, np.uint8)
        cv2.rectangle(mask, (10, 10), (50, 60), 0, -1)
        cv2.rectangle(mask, (20, 20), (40, 50), 255, -1)
        cv2.rectangle(mask, (25, 25), (30, 30), 0, -1)
        cv2.rectangle(mask, (60, 10), (70, 20), 0, -1)
        cv2.imwrite(str(Path(self.dirs['originals'])/'mask.png'), mask)
        zone = np.full((80, 100, 3), 255, np.uint8)
        cv2.rectangle(zone, (5, 5), (35, 50), (0, 0, 255), -1)
        cv2.rectangle(zone, (15, 15), (25, 35), (255, 255, 255), -1)
        cv2.imwrite(str(Path(self.dirs['originals'])/'zones.png'), zone)
        self.row = dict(video="video", resolution="(100, 80)", roi="((3, 4), (90, 75))",
                        conv=.25, fps=25, frame_start=1, frame_stop=100, objects=1,
                        maskimg="mask.png", wallimg=None, zoneimg="zones.png")
        self.project = object.__new__(ATracker)
        self.project.dirs = self.dirs
        self.project.overview = pd.DataFrame([self.row])

    def tearDown(self):
        self.tmp.cleanup()

    def test_holes_islands_and_full_frame_coordinates(self):
        files = self.project.export_geometry(names="video", zone_names={1: "refuge"})
        data = json.loads(Path(files[0]).read_text())
        self.assertEqual(data['calibration']['mm_per_pixel'], .25)
        self.assertEqual(data['coordinate_system']['reference'], 'full_frame')
        features = {f['properties']['kind']: f for f in data['features']}
        mask = shape(features['mask']['geometry'])
        self.assertEqual(len(mask.geoms), 3)
        for xy, inside in [((15,15),True), ((22,22),False), ((27,27),True), ((65,15),True)]:
            self.assertEqual(mask.contains(Point(*xy)), inside)
        self.assertEqual(features['zone']['properties']['name'], 'refuge')
        self.assertFalse(shape(features['zone']['geometry']).contains(Point(20,20)))
        self.assertEqual(data['associated_files']['tracked_csv'], '../tracked/video.csv')
        self.assertEqual(features['roi']['geometry']['coordinates'][0][0], [3,4])

    def test_regions_and_non_overwriting(self):
        self.project.overview = pd.DataFrame([dict(self.row, region=1), dict(self.row, region=2)])
        files = self.project.export_geometry(inds=[1])
        self.assertTrue(files[0].endswith('video_R2.geometry.json'))
        data = Path(files[0]).read_text()
        self.project.overview.loc[1, 'conv'] = .5
        self.project.export_geometry(inds=[1])
        self.assertEqual(Path(files[0]).read_text(), data)
        self.project.export_geometry(inds=[1], overwrite=True)
        self.assertEqual(json.loads(Path(files[0]).read_text())['calibration']['mm_per_pixel'], .5)

    def test_missing_calibration_and_bad_reference(self):
        self.project.overview.loc[0, 'conv'] = np.nan
        files = self.project.export_geometry()
        self.assertIsNone(json.loads(Path(files[0]).read_text())['calibration']['mm_per_pixel'])
        self.project.overview.loc[0, 'maskimg'] = 'missing.png'
        with self.assertRaisesRegex(ValueError, 'Cannot read mask'):
            self.project.export_geometry(overwrite=True)


if __name__ == '__main__':
    unittest.main()
