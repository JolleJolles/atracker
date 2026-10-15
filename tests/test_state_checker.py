"""State CSV round-trips and navigation use real, short synthetic videos."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import importlib
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QMessageBox

from atracker import ATracker
from atracker.editor._state_candidates import StateCandidates
from atracker.editor._state_checker import StateCheckerWindow, event_checker


class StateCheckerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name in ("a", "b"):
            writer = cv2.VideoWriter(str(self.root / (name + ".avi")), cv2.VideoWriter_fourcc(*"MJPG"), 20, (80, 60))
            self.assertTrue(writer.isOpened())
            for frame in range(60):
                writer.write(np.full((60, 80, 3), frame * 3, np.uint8))
            writer.release()
            cap = cv2.VideoCapture(str(self.root / (name + ".avi")))
            self.assertTrue(cap.isOpened())
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 60)
            self.assertTrue(cap.read()[0])
            cap.release()
        self.csv = self.root / "states.csv"
        pd.DataFrame(dict(video=["a", "a", "b"], start_frame=[5, 25, 10],
            end_frame=[15, 35, 20], state=["state_jump", "state_slow", "state_jump"],
            review=["", "", ""], note=["keep", "", "also keep"])).to_csv(self.csv, index=False)
        self.window = None
        self.prefs = patch.object(importlib.import_module("atracker.editor._window"), "save_prefs")
        self.prefs.start()
        self.loadprefs = patch.object(importlib.import_module("atracker.editor._window"), "load_prefs", return_value={})
        self.loadprefs.start()

    def tearDown(self):
        if self.window:
            with patch.object(QMessageBox, "question", return_value=QMessageBox.Discard):
                self.window.close()
            self.window.deleteLater()
            self.app.processEvents()
        self.prefs.stop()
        self.loadprefs.stop()
        self.tmp.cleanup()

    def open_window(self, frame_base=1):
        self.model = StateCandidates(self.csv, self.root, frame_base=frame_base)
        self.window = StateCheckerWindow(self.model)
        self.window.advance_after_review.setChecked(False)
        self.window.show()
        self.app.processEvents()
        self.app.processEvents()
        return self.window

    def test_csv_roundtrip_and_external_changes(self):
        model = StateCandidates(self.csv, self.root)
        model.update(1, 26, 34, "denied")
        model.save()
        saved = pd.read_csv(self.csv, keep_default_na=False)
        self.assertEqual(list(saved.review), ["", "denied", ""])
        self.assertEqual(saved.loc[1, "start_frame"], 26)
        self.assertEqual(saved.loc[0, "note"], "keep")
        self.assertFalse(model.dirty)
        self.csv.write_text(self.csv.read_text() + "\n")
        with self.assertRaisesRegex(RuntimeError, "outside"):
            model.save()

    def test_missing_review_and_zero_based(self):
        data = pd.read_csv(self.csv).drop(columns="review")
        data.loc[0, ["start_frame", "end_frame"]] = [0, 4]
        data.to_csv(self.csv, index=False)
        w = self.open_window(frame_base=0)
        self.assertEqual(w.current_frame_idx, 0)
        self.assertEqual(w.flim_start_spin.value(), 1)
        w.flim_stop_spin.setValue(6)
        w.review("confirmed")
        saved = pd.read_csv(self.csv)
        self.assertEqual(saved.loc[0, "end_frame"], 5)
        self.assertEqual(saved.loc[0, "review"], "confirmed")

    def test_navigation_bounds_decisions_and_timeline(self):
        w = self.open_window()
        self.assertEqual(w.current_frame_idx, 4)
        self.assertTrue(w.timeline_group.isVisible())
        w.flim_start_spin.setValue(6)
        w.flim_stop_spin.setValue(14)
        w._navigate_to(1)
        self.app.processEvents()
        self.assertEqual(w.current_frame_idx, 24)
        w.review("denied")
        w._navigate_to(0)
        self.app.processEvents()
        self.assertEqual(w.flim_start_spin.value(), 6)
        self.assertEqual(w.flim_stop_spin.value(), 14)
        w.review("confirmed")
        w.navigate_video(1)
        self.app.processEvents()
        self.assertEqual(w._file_idx, 2)
        self.assertEqual(w.current_frame_idx, 9)
        w.navigate_video(-1)
        self.app.processEvents()
        self.assertEqual(w._file_idx, 1)
        # Select first event by clicking its state row on the timeline.
        x = w.timeline.LEFT + int(9 * w.timeline._width() / 60)
        QTest.mouseClick(w.timeline, Qt.LeftButton, pos=QPoint(x, w.timeline.TOP + 8))
        self.app.processEvents()
        self.assertEqual(w._file_idx, 0)
        self.assertEqual(w.current_frame_idx, 5)
        saved = pd.read_csv(self.csv, keep_default_na=False)
        self.assertEqual(list(saved.review), ["confirmed", "denied", ""])
        self.assertEqual(list(saved.note), ["keep", "", "also keep"])
        w.grab().save(str(self.root / "state-checker.png"))

    def test_invalid_bounds_and_missing_media(self):
        data = pd.read_csv(self.csv)
        data.loc[0, "end_frame"] = 100
        data.to_csv(self.csv, index=False)
        with self.assertRaisesRegex(ValueError, "exceeds video"):
            event_checker(self.csv, self.root)
        data.loc[0, "end_frame"] = 1
        data.to_csv(self.csv, index=False)
        with self.assertRaisesRegex(ValueError, "precede"):
            StateCandidates(self.csv, self.root)
        data.loc[0, "end_frame"] = 10
        data.loc[0, "video"] = "absent"
        data.to_csv(self.csv, index=False)
        with self.assertRaisesRegex(ValueError, "Expected one"):
            StateCandidates(self.csv, self.root)

    def test_pending_edits_close_and_invalid_interval(self):
        w = self.open_window()
        w.flim_start_spin.setValue(16)
        with patch.object(QMessageBox, "warning") as warning:
            w._navigate_to(1)
            self.assertEqual(w._file_idx, 0)
            warning.assert_called_once()
        w.flim_stop_spin.setValue(18)
        self.assertTrue(self.model.dirty)
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Cancel):
            self.assertFalse(w.close())
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Save):
            self.assertTrue(w.close())
        saved = pd.read_csv(self.csv)
        self.assertEqual(saved.loc[0, "start_frame"], 16)
        self.assertEqual(saved.loc[0, "end_frame"], 18)

    def test_modifier_values_follow_events_and_survive_csv_save(self):
        model = StateCandidates(self.csv, self.root, modifier="complete")
        self.model = model
        self.window = StateCheckerWindow(model)
        w = self.window
        w.advance_after_review.setChecked(False)
        w.show()
        self.app.processEvents()
        w.modifier_checkbox.setChecked(True)
        w.review("confirmed")
        w._navigate_to(1)
        self.app.processEvents()
        self.assertFalse(w.modifier_checkbox.isChecked())
        w.review("confirmed")
        w._navigate_to(0)
        self.app.processEvents()
        self.assertTrue(w.modifier_checkbox.isChecked())
        saved = pd.read_csv(self.csv, dtype=str, keep_default_na=False)
        self.assertEqual(list(saved.complete), ["true", "false", ""])
        reloaded = StateCandidates(self.csv, self.root, modifier="complete")
        self.assertEqual(reloaded.data.loc[0, "complete"], "true")
        with self.assertRaisesRegex(ValueError, "required"):
            StateCandidates(self.csv, self.root, modifier="review")

    def test_project_dispatch(self):
        project = object.__new__(ATracker)
        project.dirs = {"originals": str(self.root)}
        with patch.object(importlib.import_module("atracker.editor"), "event_checker", return_value="opened") as launch:
            self.assertEqual(project.editor(purpose="event_checker", candidates=str(self.csv)), "opened")
            launch.assert_called_once()
            args, kwargs = launch.call_args
            self.assertEqual(args, (str(self.csv), str(self.root)))
            self.assertEqual(kwargs["frame_base"], 1)
            self.assertIsNone(kwargs["modifier"])


if __name__ == "__main__":
    unittest.main()
