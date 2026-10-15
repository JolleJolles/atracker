"""Review candidate state intervals using the existing video editor controls."""
import sys

import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QMessageBox, QPushButton

from ._state_candidates import StateCandidates
from ._timeline import TrackingTimeline
from ._window import PyQt5ShapeDrawerWindow


class StateTimeline(TrackingTimeline):
    LEFT = 125

    def wheelEvent(self, event):
        step = self.editor.fps_spin.value() if event.modifiers() & Qt.ShiftModifier else 1
        direction = -1 if event.angleDelta().y() > 0 else 1
        self.editor.onVideoSliderChanged(self.editor.current_frame_idx + direction * step)
        event.accept()

    def sync(self):
        rows = self.editor.video_candidates()
        self._ids = list(dict.fromkeys(rows.state))
        self.setMinimumHeight(self.TOP + self.ROW * len(self._ids) + 25)
        self.update()

    def _seek(self, event):
        frame = self._frame_at(event.x()) + self.editor.candidates.frame_base
        row_number = (event.y() - self.TOP) // self.ROW
        if 0 <= row_number < len(self._ids):
            rows = self.editor.video_candidates()
            hits = rows[(rows.state == self._ids[row_number]) &
                        (rows.start_frame <= frame) & (rows.end_frame >= frame)]
            if len(hits) and self.editor._file_idx not in hits.index:
                self.editor._navigate_to(int(hits.index[0]))
                return
        super()._seek(event)

    def paintEvent(self, event):
        editor = self.editor
        first, last = self._bounds()
        width = self._width()
        span = last - first + 1
        offset = 1 - editor.candidates.frame_base
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().base())
        painter.setPen(self.palette().text().color())
        painter.drawText(8, 17, "Candidates — gold: unreviewed   green: confirmed   red: denied")
        for line, state in enumerate(self._ids):
            y = self.TOP + line * self.ROW
            painter.setPen(self.palette().text().color())
            painter.drawText(4, y + 15, state)
            painter.fillRect(self.LEFT, y, width, self.ROW - 3, QColor(225, 225, 225))
            for index, row in editor.video_candidates().iterrows():
                if row.state != state:
                    continue
                start, stop = row.start_frame + offset, row.end_frame + offset
                if stop < first or start > last:
                    continue
                x = self.LEFT + int((max(start, first) - first) * width / span)
                end = self.LEFT + int((min(stop, last) - first + 1) * width / span)
                colour = {"confirmed": "#309969", "denied": "#c74141"}.get(row.review, "#d99a32")
                painter.fillRect(x, y, max(2, end-x), self.ROW - 3, QColor(colour))
                if index == editor._file_idx:
                    painter.setPen(QPen(QColor("black"), 2))
                    painter.drawRect(x, y, max(2, end-x), self.ROW - 3)
        bottom = self.TOP + len(self._ids) * self.ROW
        painter.setPen(self.palette().text().color())
        painter.drawText(self.LEFT, bottom + 17, str(first))
        painter.drawText(self.LEFT + width - 65, bottom + 3, 65, 18, Qt.AlignRight, str(last))
        x = self.LEFT + int((editor.current_frame_idx + 1 - first) * width / span)
        painter.setPen(QPen(QColor(35, 100, 235), 2))
        painter.drawLine(x, self.TOP - 4, x, bottom)
        painter.end()


class StateCheckerWindow(PyQt5ShapeDrawerWindow):
    def __init__(self, candidates):
        self.candidates = candidates
        self._loading_candidate = True
        super().__init__(file_infos=candidates.file_infos())
        self.opmode_combo.setCurrentText("Frame limits")
        self.opmode_group.hide()
        self.mode_group.hide()
        self.func_group.hide()
        self.flim_group.setTitle("Candidate start / end (inclusive)")
        self.nav_group.setTitle("State checker — candidate navigation")
        self.review_label = QLabel()
        self.nav_group.layout().addWidget(self.review_label)
        controls = QHBoxLayout()
        self.confirm_button = QPushButton("Confirm")
        self.deny_button = QPushButton("Deny")
        self.reset_button = QPushButton("Unreviewed")
        for button, status in ((self.confirm_button, "confirmed"), (self.deny_button, "denied"),
                               (self.reset_button, "")):
            button.clicked.connect(lambda checked=False, value=status: self.review(value))
            controls.addWidget(button)
        self.nav_group.layout().addLayout(controls)
        file_controls = QHBoxLayout()
        for label, direction in (("← Video", -1), ("Video →", 1)):
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, step=direction: self.navigate_video(step))
            file_controls.addWidget(button)
        self.nav_group.layout().addLayout(file_controls)
        old_timeline = self.timeline_scroll.takeWidget()
        self.timeline = StateTimeline(self)
        self.timeline_scroll.setWidget(self.timeline)
        self.timeline_scroll.setMinimumHeight(120)
        # Reconnect updates that were bound to the old timeline instance.
        self.timeline_use_range.toggled.connect(self.timeline.update)
        self.range_slider.valueChanged.connect(self.timeline.update)
        for button in (self.rb_range_all, self.rb_range_past, self.rb_range_future):
            button.toggled.connect(self.timeline.update)
        old_timeline.hide()
        self.timeline_group.setTitle("State candidates — click an interval or drag to seek")
        self.flim_start_spin.valueChanged.connect(self._bounds_changed)
        self.flim_stop_spin.valueChanged.connect(self._bounds_changed)
        self._loading_candidate = False
        self.fps_spin.setValue(max(1, round(self.fps)))
        self._show_candidate()
        QTimer.singleShot(0, self._seek_candidate_start)

    def video_candidates(self):
        video = self.candidates.data.loc[self._file_idx, "video"]
        return self.candidates.data[self.candidates.data.video == video]

    def syncTimeline(self):
        if hasattr(self, "timeline"):
            self.timeline_group.setVisible(True)
            self.timeline.sync()

    def _remember_current(self):
        # Candidate edits belong to the CSV, never to overview or coordinate drafts.
        pass

    def _bounds_changed(self):
        if self._loading_candidate:
            return
        first, last = self.flim_start_spin.value(), self.flim_stop_spin.value()
        if first > last:
            self.statusBar().showMessage("Start must not exceed end; adjust the bounds before continuing.")
            return
        delta = 1 - self.candidates.frame_base
        self.candidates.update(self._file_idx, first-delta, last-delta)
        self._file_infos[self._file_idx].update(frame_start=first, frame_stop=last)
        self.timeline.update()
        self.statusBar().showMessage("Candidate bounds changed — Confirm, Deny or S saves the CSV.")

    def _valid_bounds(self):
        if self.flim_start_spin.value() > self.flim_stop_spin.value():
            QMessageBox.warning(self, "Invalid interval", "Start must not exceed end.")
            return False
        return True

    def _show_candidate(self):
        row = self.candidates.data.loc[self._file_idx]
        self.setWindowTitle(f"ATracker — State checker — {row.video}")
        self.review_label.setText(f"{row.state} — {row.review or 'unreviewed'}")
        self.mode_group.hide()
        self.func_group.hide()
        self.syncTimeline()

    def _seek_candidate_start(self):
        self.onVideoSliderChanged(self.flim_start_spin.value() - 1)

    def _navigate_to(self, new_idx):
        if not 0 <= new_idx < len(self._file_infos) or new_idx == self._file_idx:
            return
        if not self._valid_bounds():
            return
        self._loading_candidate = True
        super()._navigate_to(new_idx)
        self._loading_candidate = False
        self._show_candidate()
        # Run after the parent's queued first-frame read.
        QTimer.singleShot(0, self._seek_candidate_start)

    def navigate_video(self, direction):
        video = self.candidates.data.loc[self._file_idx, "video"]
        indices = range(self._file_idx + direction, len(self._file_infos) if direction > 0 else -1, direction)
        for index in indices:
            if self.candidates.data.loc[index, "video"] != video:
                self._navigate_to(index)
                break

    def _store_current(self):
        if not self._valid_bounds():
            return False
        try:
            self.candidates.save()
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
            return False
        self.statusBar().showMessage(f"Saved state reviews to {self.candidates.path.name}")
        return True

    def review(self, status):
        if not self._valid_bounds():
            return
        self.candidates.data.loc[self._file_idx, "review"] = status
        self._show_candidate()
        self._store_current()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Left, Qt.Key_Right):
            self._navigate_to(self._file_idx + (-1 if event.key() == Qt.Key_Left else 1))
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        if self.candidates.dirty:
            answer = QMessageBox.question(self, "Unsaved state reviews", "Save pending CSV edits before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
            if answer == QMessageBox.Cancel or (answer == QMessageBox.Save and not self._store_current()):
                event.ignore()
                return
        super().closeEvent(event)
        if event.isAccepted():
            if self.timer:
                self.timer.stop()
            if self.cap:
                self.cap.release()


def state_checker(csv_file, video_dir, frame_base=1):
    """Review a MoveR candidate CSV; save inclusive bounds and confirmed/denied review labels.

    UI frames are one-based. Set frame_base=0 explicitly for zero-based input CSVs.
    Extra CSV columns and unselected rows are preserved. Confirm/Deny and S save
    all pending edits to the original CSV; navigation alone does not write it.
    """
    candidates = StateCandidates(csv_file, video_dir, frame_base=frame_base)
    # Check bounds before Qt can silently clamp them to the available video frames.
    for video, path in candidates.paths.items():
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        opened = cap.isOpened()
        cap.release()
        if not opened or total < 1:
            raise ValueError(f"Cannot open video: {path}")
        if (candidates.data.loc[candidates.data.video == video, "end_frame"] >= total + frame_base).any():
            raise ValueError(f"Candidate frame exceeds video length ({total} frames): {video}")
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    window = StateCheckerWindow(candidates)
    window.show()
    window.raise_()
    window.activateWindow()
    app.exec_()
    return candidates.data.copy()
