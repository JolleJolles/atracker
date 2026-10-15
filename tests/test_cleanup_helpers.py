import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from atracker.editor._project import editor_files, save_editor_data
from atracker.helpers.frames import frame_to_index, index_to_frame
from atracker.helpers.media import bg_extract
from atracker.helpers.naming import output_basename


class CleanupHelpersTest(unittest.TestCase):
    def test_editor_and_tracker_share_region_names(self):
        overview = pd.DataFrame([dict(video='fish', region=1, roi='((0, 0), (80, 60))'),
                                 dict(video='fish', region=2, roi='((0, 0), (80, 60))')])
        project = SimpleNamespace(overview=overview, dirs=dict(originals='/videos', tracked='/tracked'),
                                  threshinfo={})
        infos = editor_files(project, [1], None, (1.4, 10))
        self.assertEqual(output_basename(overview, 1), 'fish_R2')
        self.assertEqual(infos[0]['tracked_csv'], '/tracked/fish_R2.csv')
        self.assertEqual(output_basename(overview.iloc[[1]], 1), 'fish')
        overview.loc[1, 'region'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Missing region'):
            output_basename(overview, 1)

    def test_invalid_roi_is_reported(self):
        project = SimpleNamespace(overview=pd.DataFrame([dict(video='fish', roi='invalid')]),
                                  dirs=dict(originals='/videos', tracked='/tracked'), threshinfo={})
        with self.assertRaisesRegex(ValueError, 'Invalid ROI for overview row 0'):
            editor_files(project, [0], None, (1.4, 10))

    def test_editor_saves_frame_limits_and_threshold_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            project = SimpleNamespace(overview=pd.DataFrame([dict(video='fish')]),
                dirs={'originals':folder}, save=Mock(), threshinfo={},
                cfiles={'threshinfo':str(Path(folder)/'threshold.yml')})
            infos = [dict(output_basename='fish', threshold_type='bw', threshold_options={'bw':{}})]
            save_editor_data(project, infos, 0, 0, 'framelimits', (5, 20))
            project.save.assert_called_once()
            self.assertEqual(project.overview.loc[0, 'frame_stop'], 20)
            save_editor_data(project, infos, 0, 0, 'thresholding', {'threshold':50})
            self.assertEqual(infos[0]['threshold_options']['bw'], {'threshold':50})
            self.assertTrue(Path(project.cfiles['threshinfo']).exists())

    def test_frame_boundaries_and_roundtrip(self):
        self.assertEqual(frame_to_index(1), 0)
        self.assertEqual(index_to_frame(0), 1)
        self.assertEqual(index_to_frame(frame_to_index(1200)), 1200)

    def test_background_failure_releases_video(self):
        cap = Mock()
        cap.isOpened.return_value = True
        cap.read.return_value = (False, None)
        with patch('atracker.helpers.media.cv2.VideoCapture', return_value=cap):
            with self.assertRaisesRegex(ValueError, 'Cannot read video'):
                bg_extract('unreadable.mp4')
        cap.release.assert_called_once()
