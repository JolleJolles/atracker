import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from atracker.helpers.watch import watch_folder


class FolderWatchTest(unittest.TestCase):
    def test_stable_video_inherits_setup_and_is_only_tracked_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            incoming, originals, tracked = [root / name for name in ('incoming', 'originals', 'tracked')]
            for folder in (incoming, originals, tracked):
                folder.mkdir()
            (incoming / 'new.mp4').write_bytes(b'video')
            overview = pd.DataFrame([dict(video='template', roi='((0, 0), (80, 60))',
                resolution='(80, 60)', bgimg='template_bg.jpg', maskimg='shared_mask.png',
                frame_start=10, frame_stop=20, objects=1)])
            project = SimpleNamespace(dirs=dict(originals=str(originals), tracked=str(tracked)),
                overview=overview, save=Mock(), track=Mock(return_value=True))
            cap = Mock()
            cap.get.side_effect = lambda prop: {5:25, 3:80, 4:60, 7:100}[prop]
            cap.read.return_value = (True, object())
            with patch('atracker.helpers.watch.cv2.VideoCapture', return_value=cap), \
                    patch('atracker.helpers.watch.bg_extract', return_value=object()) as bg, \
                    patch('atracker.helpers.watch.cv2.imwrite', return_value=True), \
                    patch('atracker.helpers.watch.time.sleep', side_effect=[None, None, KeyboardInterrupt]):
                watch_folder(project, incoming, 0, settle_seconds=0)
            project.track.assert_called_once_with(inds=[1], folder='originals', overwrite=False)
            bg.assert_called_once_with(str((originals / 'new.mp4').resolve()), 1, 100, 25)
            row = project.overview.loc[1]
            self.assertEqual(row.maskimg, 'shared_mask.png')
            self.assertEqual(row.frame_start, 1)
            self.assertEqual(row.frame_stop, 100)
            self.assertEqual(row.bgimg, 'new_bg.jpg')
            self.assertEqual((incoming / 'new.mp4').read_bytes(), b'video')
            self.assertEqual((originals / 'new.mp4').read_bytes(), b'video')
            cap.release.assert_called_once()

    def test_project_folder_cannot_watch_itself(self):
        with tempfile.TemporaryDirectory() as folder:
            project = SimpleNamespace(dirs={'originals': folder})
            with self.assertRaises(ValueError):
                watch_folder(project, folder, 0)
