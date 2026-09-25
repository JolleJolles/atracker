import io
import unittest
from unittest.mock import patch

from atracker.helpers.logging import TrackingProgress
from atracker.helpers.media import framechecks


class TrackingProgressTest(unittest.TestCase):
    def test_pooled_output_is_rate_limited_and_identified(self):
        output = io.StringIO()
        with patch('sys.stdout', output), patch('atracker.helpers.logging.time.monotonic', side_effect=[0, 9, 10, 11]):
            progress = TrackingProgress('{P2}[15] ', pooled=True)
            progress.update(9, 100, 9)
            progress.update(10, 100, 10)
            progress.update(11, 100, 11)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertIn('{P2}[15] Frame 10/100; 10 processed — 1.0 FPS', output.getvalue())
        self.assertNotIn('\r', output.getvalue())

    def test_terminal_progress_is_cleared(self):
        output = io.StringIO()
        with patch('sys.stdout', output), patch.object(output, 'isatty', return_value=True), patch('atracker.helpers.logging.time.monotonic', side_effect=[0, 10]):
            progress = TrackingProgress('[15] ')
            progress.update(10, 100, 10)
            progress.clear()
        self.assertIn('\r[15]', output.getvalue())
        self.assertTrue(output.getvalue().endswith('\r'))
        self.assertEqual(progress.width, 0)

    def test_framechecks_can_be_silent_at_stop(self):
        class Capture:
            def get(self, key):
                return 1001
        output = io.StringIO()
        with patch('sys.stdout', output):
            stop, skip, frame = framechecks(Capture(), True, stopframe=1000, displaystep=None)
        self.assertTrue(stop)
        self.assertEqual(output.getvalue(), '')


if __name__ == '__main__':
    unittest.main()
