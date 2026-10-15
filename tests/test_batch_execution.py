"""Batches must expose failures and protect source videos and completed CSVs."""
import io
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
from box import Box

from atracker.helpers.pool import run_pool, BatchInterrupted
from atracker.helpers.output import write_csv_atomic
from atracker.track import Tracker, recover_tracking_sources


def process_item(path, fail):
    if fail:
        raise ValueError("bad input")
    Path(path).write_text("done")


class BatchExecutionTest(unittest.TestCase):
    def test_sequential_and_thread_failures_are_visible_and_other_items_run(self):
        for pools in (1, 2):
            done = []
            def work(item):
                if item == "bad":
                    raise ValueError("bad input")
                done.append(item)
            log = io.StringIO()
            with redirect_stdout(log):
                success = run_pool(work, ["first", "bad", "last"], pools=pools)
            self.assertFalse(success)
            self.assertCountEqual(done, ["first", "last"])
            self.assertIn("'bad'", log.getvalue())
            self.assertIn("1 failed", log.getvalue())

    def test_process_failure_keeps_other_files(self):
        with tempfile.TemporaryDirectory() as d, patch("atracker.helpers.pool.notebook", return_value=False):
            items = [(str(Path(d)/name), name == "bad") for name in ("first", "bad", "last")]
            self.assertFalse(run_pool(process_item, items, pools=2, mode="process"))
            self.assertEqual((Path(d)/"first").read_text(), "done")
            self.assertEqual((Path(d)/"last").read_text(), "done")
            self.assertFalse((Path(d)/"bad").exists())

    def test_interrupt_stops_sequential_batch(self):
        calls = []
        def work(item):
            calls.append(item)
            raise BatchInterrupted()
        self.assertFalse(run_pool(work, [1, 2, 3]))
        self.assertEqual(calls, [1])

    def test_failed_csv_write_keeps_previous_output(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d)/"result.csv"
            output.write_text("old output")
            def fail(path, **kwargs):
                Path(path).write_text("incomplete")
                raise OSError("disk error")
            with self.assertRaises(OSError):
                write_csv_atomic(Mock(to_csv=fail), output, index=False)
            self.assertEqual(output.read_text(), "old output")
            self.assertEqual(list(Path(d).glob("*.tmp")), [])
            write_csv_atomic(pd.DataFrame({"frame": [1]}), output, index=False)
            self.assertEqual(pd.read_csv(output).frame.tolist(), [1])

    def test_tracking_failure_closes_resources_and_restores_source(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root/"source.mp4"; temporary = root/"moved.mp4"
            source.write_bytes(b"original video")
            overlay = root/"partial_TR.mp4"
            tracker = object.__new__(Tracker)
            tracker.config = Box(vis=dict(show_tracking=False))
            capture, writer = Mock(), Mock()
            def fail(ind, path):
                tracker.trackfile = str(source); tracker.tempfile = str(temporary)
                tracker.trackedfile = str(overlay)
                shutil.move(source, temporary); tracker._source_moved = True
                tracker.cap = capture; tracker.vidout = writer
                overlay.write_bytes(b"partial overlay")
                raise ValueError("bad frame")
            tracker._setuptracking = fail
            with self.assertRaises(ValueError):
                tracker.setuptracking(0, str(source))
            self.assertEqual(source.read_bytes(), b"original video")
            self.assertFalse(temporary.exists())
            self.assertFalse(overlay.exists())
            capture.release.assert_called_once()
            writer.close.assert_called_once()

    def test_parent_recovers_terminated_worker_source_without_overwriting(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); temporary = root/"temp"; temporary.mkdir()
            source = root/"video.mp4"; moved = temporary/source.name
            moved.write_bytes(b"recover")
            recover_tracking_sources([(0, str(source)), (1, str(source))], {"temp": str(temporary)})
            self.assertEqual(source.read_bytes(), b"recover")
            moved.write_bytes(b"leave intact")
            recover_tracking_sources([(0, str(source))], {"temp": str(temporary)})
            self.assertEqual(moved.read_bytes(), b"leave intact")

    def test_link_distance_resets_between_files(self):
        tracker = object.__new__(Tracker)
        tracker.config = Box(track=dict(link_dist=100), vis=dict(show_tracking=False))
        tracker.overview = pd.DataFrame(dict(video=["a","b"],conv=[2,4],roi=[None,None]))
        tracker.inds = [0,1]; tracker.dirs = {"originals": "/tmp", "temp": "/tmp", "tracked": "/tmp"}
        tracker.suffix = ""
        tracker.setuptracking(0, "/tmp/a.mp4")
        self.assertEqual(tracker.linkdisthreshold, 50)
        tracker.setuptracking(1, "/tmp/b.mp4")
        self.assertEqual(tracker.linkdisthreshold, 25)


if __name__ == '__main__':
    unittest.main()
