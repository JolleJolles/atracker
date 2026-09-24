"""Display-only coverage of the editor's one-based coordinate frames."""

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPen
from PyQt5.QtWidgets import QWidget


class TrackingTimeline(QWidget):
    LEFT = 85
    ROW = 22
    TOP = 26

    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor
        self._cache_key = None
        self._image = None
        self._ids = []
        self.setMouseTracking(True)
        self.setMinimumHeight(76)

    def invalidate(self):
        self._cache_key = None
        self.update()

    def sync(self):
        editor = self.editor
        ids = sorted(set(range(editor.tp_total_ids)) |
                     set(editor.drawing_widget.points_by_frame))
        if ids != self._ids:
            self._ids = ids
            self.setMinimumHeight(self.TOP + self.ROW * len(ids) + 25)
        self.update()

    def _width(self):
        return max(1, self.width() - self.LEFT - 16)

    def _frame_at(self, x):
        total = max(1, self.editor.total_frames)
        return min(total - 1, max(0, int((x - self.LEFT) * total / self._width())))

    def _seek(self, event):
        if self.editor.total_frames <= 1:
            return
        self.editor.onVideoSliderChanged(self._frame_at(event.x()))
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._seek(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self._seek(event)
        self.setToolTip(f"Frame {self._frame_at(event.x()) + 1} — click or drag to seek")

    def paintEvent(self, event):
        editor = self.editor
        total = max(1, editor.total_frames)
        width = self._width()
        start = max(1, editor.flim_start_spin.value())
        stop = min(total, editor.flim_stop_spin.value())
        ptype = editor.current_ptype
        data = editor.drawing_widget.points_by_frame
        # Identity/count changes catch newly loaded data; edits explicitly invalidate.
        key = (total, width, start, stop, ptype, tuple(self._ids),
               tuple((id(data.get(i, {}).get(ptype)),
                      len(data.get(i, {}).get(ptype, {}))) for i in self._ids))
        if key != self._cache_key:
            pixels = np.full((max(1, len(self._ids)), width, 3), 155, dtype=np.uint8)
            # Every frame belongs to one pixel bin. A bin is red if ANY expected
            # frame is absent, so short gaps remain visible in long recordings.
            edges = (np.arange(width + 1, dtype=np.int64) * total + width - 1) // width + 1
            if width > total:
                first = np.arange(width, dtype=np.int64) * total // width + 1
                last = first + 1
            else:
                first, last = edges[:-1], edges[1:]
            lo = np.maximum(first, start)
            hi = np.minimum(last, stop + 1)
            expected = np.maximum(0, hi - lo)
            for row, id_ in enumerate(self._ids):
                frames = np.array(sorted(int(f) for f, point in
                    data.get(id_, {}).get(ptype, {}).items() if point is not None), dtype=np.int64)
                counts = np.searchsorted(frames, hi) - np.searchsorted(frames, lo)
                pixels[row, expected > 0] = (199, 65, 65)
                pixels[row, (expected > 0) & (counts == expected)] = (48, 153, 105)
            self._image = QImage(pixels.data, width, pixels.shape[0], width * 3,
                                 QImage.Format_RGB888).copy()
            self._cache_key = key

        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().base())
        painter.setPen(self.palette().text().color())
        painter.drawText(8, 17, f"{editor.ptype_dropdown.currentText()} — green: present   red: missing   grey: outside interval")
        for row, id_ in enumerate(self._ids):
            y = self.TOP + row * self.ROW
            painter.drawText(8, y + 15, f"ID {id_ + 1}")
            painter.drawImage(self.LEFT, y, self._image.copy(0, row, width, 1).scaled(width, self.ROW - 3))
        bottom = self.TOP + len(self._ids) * self.ROW
        painter.drawText(self.LEFT, bottom + 17, "1")
        painter.drawText(self.LEFT + width - 65, bottom + 3, 65, 18, Qt.AlignRight, str(total))
        x = self.LEFT + min(width - 1, int((editor.current_frame_idx + 0.5) * width / total))
        painter.setPen(QPen(QColor(35, 100, 235), 2))
        painter.drawLine(x, self.TOP - 4, x, bottom)
        painter.end()
