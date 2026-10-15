"""Review candidate state intervals using the existing video editor controls."""
import sys

import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QApplication, QCheckBox, QHBoxLayout, QLabel, QMessageBox, QPushButton

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
        old_bounds = self.flim_group.layout().takeAt(0).layout()
        bounds_widgets = (self.flim_start_spin, self.btn_update_start, self.flim_stop_spin, self.btn_update_stop)
        while old_bounds.count():
            item = old_bounds.takeAt(0)
            if item.widget() is not None and item.widget() not in bounds_widgets:
                item.widget().deleteLater()
        old_bounds.deleteLater()
        for label, spin, button in (("Start:", self.flim_start_spin, self.btn_update_start),
                                    ("End:", self.flim_stop_spin, self.btn_update_stop)):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            spin.setFixedWidth(120)
            row.addWidget(spin)
            row.addWidget(button)
            self.flim_group.layout().addLayout(row)
        frame_steps = QHBoxLayout()
        for label, step in (("−1 s", "second_back"), ("−1 frame", -1),
                            ("+1 frame", 1), ("+1 s", "second_forward")):
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, amount=step: self.onVideoSliderChanged(
                self.current_frame_idx + (-self.fps_spin.value() if amount == "second_back" else
                    self.fps_spin.value() if amount == "second_forward" else amount)))
            frame_steps.addWidget(button)
        self.video_group.layout().addLayout(frame_steps)
        self.frame_spin.valueChanged.connect(lambda frame: self.onVideoSliderChanged(frame - 1)
            if frame - 1 != self.current_frame_idx else None)
        self.nav_group.setTitle("Event checker — candidate navigation")
        self.review_label = QLabel()
        self.nav_group.layout().addWidget(self.review_label)
        self.modifier_checkbox = None
        if candidates.modifier:
            self.modifier_checkbox = QCheckBox(candidates.modifier.replace("_", " ").capitalize())
            self.modifier_checkbox.setToolTip(f"Save true/false in the '{candidates.modifier}' CSV column")
            self.modifier_checkbox.toggled.connect(self._modifier_changed)
            self.nav_group.layout().addWidget(self.modifier_checkbox)
        self.advance_after_review = QCheckBox("Advance after review")
        self.advance_after_review.setChecked(True)
        self.nav_group.layout().addWidget(self.advance_after_review)
        controls = QHBoxLayout()
        self.confirm_button = QPushButton("Confirm")
        self.deny_button = QPushButton("Deny")
        self.reset_button = QPushButton("Unreviewed")
        for button, status, colour in ((self.confirm_button, "confirmed", "#d9efdf"),
                                       (self.deny_button, "denied", "#f5d9d9"),
                                       (self.reset_button, "", "#f7e4c8")):
            button.setStyleSheet(f"QPushButton {{ background-color: {colour}; color: #202020; padding: 5px; }}")
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
        self.timeline_use_range.setChecked(False)
        self.timeline_group.toggled.disconnect(self.timeline_use_range.setVisible)
        self.timeline_use_range.hide()
        old_timeline.hide()
        self.timeline_group.setTitle("Event candidates — click an interval or drag to seek")
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

    def _modifier_changed(self, checked):
        self.candidates.data.loc[self._file_idx, self.candidates.modifier] = "true" if checked else "false"

    def _show_candidate(self):
        row = self.candidates.data.loc[self._file_idx]
        self.setWindowTitle(f"ATracker — Event checker — {row.video}")
        self.review_label.setText(f"{row.state} — {row.review or 'unreviewed'}")
        if self.modifier_checkbox is not None:
            self.modifier_checkbox.blockSignals(True)
            self.modifier_checkbox.setChecked(str(row[self.candidates.modifier]).lower() in ("true", "1"))
            self.modifier_checkbox.blockSignals(False)
        self.mode_group.hide()
        self.func_group.hide()
        self.syncTimeline()

    def _seek_candidate_start(self, resume=False):
        self.onVideoSliderChanged(self.flim_start_spin.value() - 1)
        if resume:
            self.onPlayClicked()

    def _navigate_to(self, new_idx):
        if not 0 <= new_idx < len(self._file_infos) or new_idx == self._file_idx:
            return
        if not self._valid_bounds():
            return
        was_playing = self.timer is not None and self.timer.isActive()
        self._loading_candidate = True
        super()._navigate_to(new_idx)
        self._loading_candidate = False
        self._show_candidate()
        # Seek after the parent's queued first-frame read, then restore playback.
        QTimer.singleShot(0, lambda: self._seek_candidate_start(resume=was_playing))

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
        self.statusBar().showMessage(f"Saved event reviews to {self.candidates.path.name}")
        return True

    def review(self, status):
        if not self._valid_bounds():
            return
        self.candidates.data.loc[self._file_idx, "review"] = status
        if self.modifier_checkbox is not None and status:
            self._modifier_changed(self.modifier_checkbox.isChecked())
        self._show_candidate()
        if self._store_current() and self.advance_after_review.isChecked():
            self._navigate_to(self._file_idx + 1)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Left, Qt.Key_Right):
            self._navigate_to(self._file_idx + (-1 if event.key() == Qt.Key_Left else 1))
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        if self.candidates.dirty:
            answer = QMessageBox.question(self, "Unsaved event reviews", "Save pending CSV edits before closing?",
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


def event_checker(csv_file, video_dir, frame_base=1, modifier=None):
    """Review a MoveR candidate CSV; save inclusive bounds and confirmed/denied review labels.

    UI frames are one-based. Set frame_base=0 explicitly for zero-based input CSVs.
    modifier optionally adds a named checkbox saved as true/false in that CSV column.
    Extra CSV columns and unselected rows are preserved. Confirm/Deny and S save
    all pending edits to the original CSV; navigation alone does not write it.
    """
    candidates = StateCandidates(csv_file, video_dir, frame_base=frame_base, modifier=modifier)
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
