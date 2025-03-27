import os
import sys
from pythutils.sysutils import get_google_drive_path

# Load atracker
drive_path = get_google_drive_path()
atracker_folder = os.path.join(drive_path, "ABEClab", "atracker")
sys.path.append(atracker_folder)
from atracker.imgprocessor import ImgProcessor

# Suppress unwanted macOS IMKClient messages
os.environ["QT_MAC_DISABLE_FOREIGN_WINDOWS"] = "1"

import sys, math, cv2, numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QHBoxLayout, QVBoxLayout,
    QGroupBox, QComboBox, QCheckBox, QPushButton, QSlider, QSpinBox, QSizePolicy
)
from PyQt5.QtGui import QPainter, QPen, QColor, QPolygon, QImage, QPixmap
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal, QTimer


# ---------- Styling Constant ----------
GROUPBOX_STYLE = """
QGroupBox {
    background-color: #eeeeee;
    border: 1px solid #999;
    border-radius: 5px;
    margin-top: 0.5em;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 3px;
    margin-left: 0px;
    margin-top: 0px;
}
"""

# ---------- Utility Functions ----------
def toGrayscale(qimg):
    """Convert a QImage to grayscale using OpenCV."""
    qimg = qimg.convertToFormat(QImage.Format_RGBA8888)
    width = qimg.width()
    height = qimg.height()
    ptr = qimg.bits()
    ptr.setsize(qimg.byteCount())
    arr = np.array(ptr).reshape(height, width, 4)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bgr2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    rgba2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGBA)
    return QImage(rgba2.data, width, height, qimg.bytesPerLine(), QImage.Format_RGBA8888).copy()

def cvMatToQImage(frame_bgr):
    """Convert an OpenCV BGR frame to a QImage (RGBA)."""
    h, w, ch = frame_bgr.shape
    if ch == 3:
        frame_rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
    else:
        frame_rgba = frame_bgr
    return QImage(frame_rgba.data, w, h, 4*w, QImage.Format_RGBA8888).copy()

# ---------- PyQt5ShapeDrawer (Drawing Widget) ----------
class PyQt5ShapeDrawer(QWidget):
    modeChanged = pyqtSignal(str)  # Emitted when drawing mode is changed

    def __init__(self, file=None, is_video=False):
        super().__init__()
        self.mode_list = ["line", "rectangle", "polygon", "circle", "ellipse", "point"]
        self.drawing_mode = self.mode_list[0]
        self.shapes = []  # List of (mode, shape_data)
        self.main_window = None

        # State for current shape
        self.start_point_orig = None
        self.end_point_orig = None
        self.polygon_points_orig = []
        # For "point" mode in video: one point per frame
        self.points_by_frame = {}  # {frame_number: QPoint}
        self.points_orig = []      # for image mode
        self.temp_point_orig = None
        self.drawing_active = False
        self.last_click_orig = None
        self.final_output = None

        self.is_video = is_video
        self.background_image = None
        self.orig_width, self.orig_height = 640, 480
        if not is_video and file:
            file_path = os.path.expanduser(file)
            bg = QImage(file_path)
            if not bg.isNull():
                self.background_image = bg
                self.orig_width = bg.width()
                self.orig_height = bg.height()

        self.mask_image = QImage(self.orig_width, self.orig_height, QImage.Format_Grayscale8)
        self.mask_image.fill(255)
        self.crosshair_enabled = False
        self.mouse_pos = QPoint(0, 0)
        self.drawing_color = QColor(0, 255, 0)  # default green

        self.blackwhite = False
        self.show_mask = False
        self.inverted = False

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        QTimer.singleShot(0, self.setFocus)

    def setShowMask(self, checked):
        self.show_mask = checked
        self.update()

    def setInvertMask(self, checked):
        self.inverted = checked
        self.update()

    def setBackgroundImage(self, qimg):
        self.background_image = qimg
        if qimg:
            new_w, new_h = qimg.width(), qimg.height()
            if new_w != self.orig_width or new_h != self.orig_height:
                self.orig_width, self.orig_height = new_w, new_h
                self.mask_image = QImage(new_w, new_h, QImage.Format_Grayscale8)
                self.mask_image.fill(255)
        self.update()

    def mousePressEvent(self, event):
        # Only accept clicks within the displayed image bounds.
        scale, ox, oy = self.currentScaleAndOffset()
        draw_w = self.orig_width * scale
        draw_h = self.orig_height * scale
        if (event.pos().x() < ox or event.pos().x() > ox+draw_w or
            event.pos().y() < oy or event.pos().y() > oy+draw_h):
            return
        if event.button() == Qt.LeftButton:
            orig_pos = self.convertToOriginal(event.pos())
            # In ROI mode, force rectangle.
            if self.main_window and self.main_window.opmode_combo.currentText().lower() == "roi":
                self.drawing_mode = "rectangle"
            if self.drawing_mode in ["line", "rectangle", "circle", "ellipse"]:
                self.start_point_orig = orig_pos
                self.end_point_orig = orig_pos
                self.drawing_active = True
            elif self.drawing_mode == "polygon":
                self.polygon_points_orig.append(orig_pos)
            elif self.drawing_mode == "point":
                if self.is_video and self.main_window:
                    cur_frame = self.main_window.current_frame_idx
                    if cur_frame not in self.points_by_frame:
                        self.points_by_frame[cur_frame] = orig_pos
                else:
                    if not self.points_orig:
                        self.points_orig.append(orig_pos)
            self.last_click_orig = orig_pos
            self.update()
            self.setFocus()
            if self.main_window:
                self.main_window.proxyUpdate()

    def mouseMoveEvent(self, event):
        self.mouse_pos = event.pos()
        if self.drawing_mode in ["line", "rectangle", "circle", "ellipse"]:
            if self.drawing_active and self.start_point_orig is not None:
                self.end_point_orig = self.convertToOriginal(event.pos())
        elif self.drawing_mode == "polygon":
            self.temp_point_orig = self.convertToOriginal(event.pos())
        self.update()
        if self.main_window:
            self.main_window.proxyUpdate()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.drawing_mode in ["line", "rectangle", "circle", "ellipse"]:
                self.end_point_orig = self.convertToOriginal(event.pos())
                self.drawing_active = False
            self.update()
            if self.main_window:
                self.main_window.proxyUpdate()

    def mouseDoubleClickEvent(self, event):
        if self.drawing_mode == "polygon" and len(self.polygon_points_orig) >= 2:
            if self.polygon_points_orig[0] != self.polygon_points_orig[-1]:
                self.polygon_points_orig.append(self.polygon_points_orig[0])
            self.addCurrentShape()
            self.update()

    def addCurrentShapeIfNeeded(self):
        if self.drawing_mode in ["line", "rectangle", "circle", "ellipse"]:
            if self.start_point_orig and self.end_point_orig:
                self.addCurrentShape()
        elif self.drawing_mode == "polygon" and self.polygon_points_orig:
            if self.polygon_points_orig[0] != self.polygon_points_orig[-1]:
                self.polygon_points_orig.append(self.polygon_points_orig[0])
            self.addCurrentShape()
        elif self.drawing_mode == "point":
            # In video mode, point is already stored.
            if not self.is_video and self.points_orig:
                self.addCurrentShape()

    def clearCurrentShape(self):
        self.start_point_orig = None
        self.end_point_orig = None
        self.polygon_points_orig = []
        self.points_orig = []
        self.temp_point_orig = None

    def addCurrentShape(self):
        mode = self.drawing_mode
        shape_data = None
        if mode in ["line", "rectangle", "circle", "ellipse"]:
            if self.start_point_orig and self.end_point_orig:
                if mode == "line":
                    shape_data = ((self.start_point_orig.x(), self.start_point_orig.y()),
                                  (self.end_point_orig.x(), self.end_point_orig.y()))
                elif mode == "rectangle":
                    x1 = min(self.start_point_orig.x(), self.end_point_orig.x())
                    y1 = min(self.start_point_orig.y(), self.end_point_orig.y())
                    x2 = max(self.start_point_orig.x(), self.end_point_orig.x())
                    y2 = max(self.start_point_orig.y(), self.end_point_orig.y())
                    shape_data = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
                elif mode == "circle":
                    cx = (self.start_point_orig.x() + self.end_point_orig.x()) // 2
                    cy = (self.start_point_orig.y() + self.end_point_orig.y()) // 2
                    dx = self.end_point_orig.x() - self.start_point_orig.x()
                    dy = self.end_point_orig.y() - self.start_point_orig.y()
                    radius = int(math.hypot(dx, dy) / 2)
                    shape_data = ((cx, cy), radius)
                elif mode == "ellipse":
                    cx = (self.start_point_orig.x() + self.end_point_orig.x()) // 2
                    cy = (self.start_point_orig.y() + self.end_point_orig.y()) // 2
                    rx = abs(self.end_point_orig.x() - self.start_point_orig.x()) // 2
                    ry = abs(self.end_point_orig.y() - self.start_point_orig.y()) // 2
                    shape_data = ((cx, cy), (rx, ry))
        elif mode == "polygon":
            if len(self.polygon_points_orig) >= 2:
                pts = [(pt.x(), pt.y()) for pt in self.polygon_points_orig]
                if pts[0] != pts[-1]:
                    pts.append(pts[0])
                shape_data = tuple(pts)
        elif mode == "point":
            # In image mode, add the point.
            if not self.is_video and self.points_orig:
                shape_data = tuple((pt.x(), pt.y()) for pt in self.points_orig)
        if shape_data is not None:
            self.shapes.append((mode, shape_data))
            # In mask (default) mode, update the mask.
            if (self.main_window and 
                self.main_window.opmode_combo.currentText().lower() == "mask"):
                self.addShapeToMask(mode, shape_data)
            self.clearCurrentShape()

    def addShapeToMask(self, mode, shape_data):
        painter = QPainter(self.mask_image)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(0), 3)
        painter.setPen(pen)
        painter.setBrush(QColor(0))
        if mode == "line":
            p1, p2 = shape_data
            painter.drawLine(QPoint(*p1), QPoint(*p2))
        elif mode == "rectangle":
            top_left = shape_data[0]
            bottom_right = shape_data[2]
            rect = QRect(QPoint(*top_left), QPoint(*bottom_right))
            painter.drawRect(rect)
        elif mode == "polygon":
            points = [QPoint(x, y) for (x, y) in shape_data]
            painter.drawPolygon(QPolygon(points))
        elif mode == "circle":
            center, radius = shape_data
            painter.drawEllipse(QPoint(*center), radius, radius)
        elif mode == "ellipse":
            center, axes = shape_data
            rx, ry = axes
            painter.drawEllipse(QPoint(*center), rx, ry)
        painter.end()

    def convertToOriginal(self, pos):
        scale, ox, oy = self.currentScaleAndOffset()
        x = (pos.x() - ox) / scale
        y = (pos.y() - oy) / scale
        return QPoint(int(x), int(y))

    def convertToDisplay(self, orig_point):
        scale, ox, oy = self.currentScaleAndOffset()
        x = orig_point.x() * scale + ox
        y = orig_point.y() * scale + oy
        return QPoint(int(x), int(y))

    def currentScaleAndOffset(self):
        scale = min(self.width() / self.orig_width, self.height() / self.orig_height)
        draw_width = self.orig_width * scale
        draw_height = self.orig_height * scale
        offset_x = (self.width() - draw_width) / 2
        offset_y = (self.height() - draw_height) / 2
        return scale, offset_x, offset_y

    def getAdjustedBackground(self):
        if not self.background_image or self.background_image.isNull():
            return None
        bg = self.background_image
        if self.blackwhite:
            bg = toGrayscale(bg)
        scale, _, _ = self.currentScaleAndOffset()
        w = int(bg.width() * scale)
        h = int(bg.height() * scale)
        return bg.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        scale, offset_x, offset_y = self.currentScaleAndOffset()
        bg = self.getAdjustedBackground()
        if bg:
            painter.drawImage(int(offset_x), int(offset_y), bg)
        else:
            painter.fillRect(self.rect(), QColor(255,255,255))
        painter.setPen(QPen(self.drawing_color, 3))
        mode = self.drawing_mode
        if mode in ["line", "rectangle", "circle", "ellipse"]:
            if self.start_point_orig and self.end_point_orig:
                p1 = self.convertToDisplay(self.start_point_orig)
                p2 = self.convertToDisplay(self.end_point_orig)
                if mode == "line":
                    painter.drawLine(p1, p2)
                elif mode == "rectangle":
                    painter.drawRect(QRect(p1, p2).normalized())
                elif mode == "circle":
                    center = QPoint((p1.x()+p2.x())//2, (p1.y()+p2.y())//2)
                    radius = int(math.hypot(p2.x()-p1.x(), p2.y()-p1.y())/2)
                    painter.drawEllipse(center, radius, radius)
                elif mode == "ellipse":
                    center = QPoint((p1.x()+p2.x())//2, (p1.y()+p2.y())//2)
                    rx = abs(p2.x()-p1.x())//2
                    ry = abs(p2.y()-p1.y())//2
                    painter.drawEllipse(center, rx, ry)
        elif mode == "polygon":
            if self.polygon_points_orig:
                pts_disp = [self.convertToDisplay(pt) for pt in self.polygon_points_orig]
                painter.drawPolyline(QPolygon(pts_disp))
                if self.temp_point_orig:
                    painter.drawLine(pts_disp[-1], self.convertToDisplay(self.temp_point_orig))
        elif mode == "point":
            if self.is_video and self.main_window:
                cur_frame = self.main_window.current_frame_idx
                pt = self.points_by_frame.get(cur_frame, None)
                if pt:
                    center = self.convertToDisplay(pt)
                    painter.drawEllipse(center, 5, 5)
            else:
                if self.points_orig:
                    for pt in self.points_orig:
                        center = self.convertToDisplay(pt)
                        painter.drawEllipse(center, 5, 5)
        if self.crosshair_enabled:
            painter.setPen(QPen(QColor(0,0,0), 1))
            painter.drawLine(self.mouse_pos.x(), 0, self.mouse_pos.x(), self.height())
            painter.drawLine(0, self.mouse_pos.y(), self.width(), self.mouse_pos.y())
        if self.show_mask:
            painter.setOpacity(0.8)
            mask_disp = self.mask_image.scaled(
                int(self.orig_width * scale),
                int(self.orig_height * scale),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            painter.drawImage(int(offset_x), int(offset_y), mask_disp)
            painter.setOpacity(1.0)
        painter.end()

# ---------- PyQt5ShapeDrawerWindow (Main Window) ----------
class PyQt5ShapeDrawerWindow(QMainWindow):
    def __init__(self, file=None):
        super().__init__()
        self.setGeometry(0, 0, 1024, 768)
        self.setMinimumSize(1024, 768)
        self.setWindowTitle("ATracker - PyQt5 Shape Drawer")
        self.cap = None
        self.is_video = False
        self.timer = None
        self.total_frames = 0
        # We'll use 0-index internally; display as 1-indexed.
        self.current_frame_idx = 0
        self.tresh_params = {"blur":0, "erode":0, "blur2":0, "treshold":0, "min_area":0, "max_area":0}
        self.img_tresh = None

        if file:
            path = os.path.expanduser(file)
            cap_test = cv2.VideoCapture(path)
            if cap_test.isOpened():
                self.is_video = True
                self.cap = cap_test
                self.total_frames = int(cap_test.get(cv2.CAP_PROP_FRAME_COUNT))
            else:
                cap_test.release()

        # Fixed sidebar widget with width 350
        self.left_widget = QWidget()
        self.left_widget.setFixedWidth(350)
        # Operation Mode
        self.opmode_group = self.makeGroupBox("Operation Mode")
        self.opmode_combo = QComboBox()
        self.opmode_combo.addItems(["roi", "mask", "conv", "point", "framelimits", "tresholding"])
        self.opmode_combo.currentIndexChanged.connect(self.onOpModeChanged)
        self.opmode_group.layout().addWidget(self.opmode_combo)
        # Info group (frame info at top)
        self.info_group = self.makeGroupBox("Info")
        self.info_label = QLabel("Frame : 0 / 0\nCurrent Position: N/A\nLast Click: N/A")
        self.info_group.layout().addWidget(self.info_label)
        # Video Controls (with FPS and Frame spin boxes)
        self.video_group = None
        if self.is_video and self.cap:
            self.video_group = self.makeGroupBox("Video Controls")
            self.video_slider = QSlider(Qt.Horizontal)
            self.video_slider.setRange(0, max(0, self.total_frames - 1))
            self.video_group.layout().addWidget(self.video_slider)
            h_buttons = QHBoxLayout()
            self.btn_start = QPushButton("<")
            self.btn_play = QPushButton("Play")
            self.btn_stop = QPushButton("Stop")
            self.btn_end = QPushButton(">")
            h_buttons.addWidget(self.btn_start)
            h_buttons.addWidget(self.btn_play)
            h_buttons.addWidget(self.btn_stop)
            h_buttons.addWidget(self.btn_end)
            self.video_group.layout().addLayout(h_buttons)
            # FPS and Frame spin boxes
            video_ctrl_container = QWidget()
            video_ctrl_container.setFixedWidth(175)
            h_fps_frame = QHBoxLayout()
            h_fps_frame.setContentsMargins(0,0,0,0)
            h_fps_frame.addWidget(QLabel("FPS:"))
            self.fps_spin = QSpinBox()
            self.fps_spin.setRange(1, 999)
            self.fps_spin.setValue(25)
            self.fps_spin.setFixedWidth(50)
            h_fps_frame.addWidget(self.fps_spin)
            h_fps_frame.addSpacing(10)
            h_fps_frame.addWidget(QLabel("Frame:"))
            self.frame_spin = QSpinBox()
            self.frame_spin.setRange(1, max(1, self.total_frames))
            self.frame_spin.setFixedWidth(50)
            self.frame_spin.editingFinished.connect(self.onFrameSpinEditingFinished)
            h_fps_frame.addWidget(self.frame_spin)
            video_ctrl_container.setLayout(h_fps_frame)
            self.video_group.layout().addWidget(video_ctrl_container)
            self.video_slider.valueChanged.connect(self.onVideoSliderChanged)
            self.btn_start.clicked.connect(self.onStartClicked)
            self.btn_play.clicked.connect(self.onPlayClicked)
            self.btn_stop.clicked.connect(self.onStopClicked)
            self.btn_end.clicked.connect(self.onEndClicked)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.nextFrame)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            QTimer.singleShot(0, self.nextFrame)
        # Drawing Mode group – hidden in certain opmodes
        self.mode_group = self.makeGroupBox("Drawing Mode")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["line", "rectangle", "polygon", "circle", "ellipse", "point"])
        self.mode_group.layout().addWidget(self.mode_combo)
        # Functions group – checkboxes on top, hue slider below.
        self.func_group = self.makeGroupBox("Functions")
        h_ticks = QHBoxLayout()
        self.cb_blackwhite = QCheckBox("Image B/W")
        self.cb_showmask = QCheckBox("Show Mask")
        self.cb_invertmask = QCheckBox("Invert Mask")
        h_ticks.addWidget(self.cb_blackwhite)
        h_ticks.addWidget(self.cb_showmask)
        h_ticks.addWidget(self.cb_invertmask)
        self.func_group.layout().addLayout(h_ticks)
        h_hue = QHBoxLayout()
        h_hue.addWidget(QLabel("Drawing Hue:"))
        self.hue_slider = QSlider(Qt.Horizontal)
        self.hue_slider.setRange(0, 359)
        self.hue_slider.setValue(120)
        self.hue_slider.setFixedWidth(100)
        h_hue.addWidget(self.hue_slider)
        self.func_group.layout().addLayout(h_hue)
        # Frame Limits group
        self.flim_group = self.makeGroupBox("Frame Limits")
        self.flim_group.setVisible(False)
        h_flim = QHBoxLayout()
        h_flim.setContentsMargins(0,0,0,0)
        h_flim.addWidget(QLabel("Start:"))
        self.flim_start_spin = QSpinBox()
        self.flim_start_spin.setRange(1, max(1, self.total_frames))
        self.flim_start_spin.setValue(1)
        self.flim_start_spin.setFixedWidth(50)
        h_flim.addWidget(self.flim_start_spin)
        self.btn_update_start = QPushButton("Set")
        self.btn_update_start.setFixedWidth(40)
        self.btn_update_start.clicked.connect(self.onUpdateStart)
        h_flim.addWidget(self.btn_update_start)
        h_flim.addSpacing(10)
        h_flim.addWidget(QLabel("Stop:"))
        self.flim_stop_spin = QSpinBox()
        self.flim_stop_spin.setRange(1, max(1, self.total_frames))
        self.flim_stop_spin.setValue(self.total_frames)
        self.flim_stop_spin.setFixedWidth(50)
        h_flim.addWidget(self.flim_stop_spin)
        self.btn_update_stop = QPushButton("Set")
        self.btn_update_stop.setFixedWidth(40)
        self.btn_update_stop.clicked.connect(self.onUpdateStop)
        h_flim.addWidget(self.btn_update_stop)
        self.flim_group.layout().addLayout(h_flim)
        # Tresholding Controls group
        self.tresh_group = self.makeGroupBox("Tresholding Controls")
        self.tresh_group.setVisible(False)
        layout_t = QVBoxLayout()
        # For each parameter, create a horizontal layout with label and slider.
        for label_text, max_val in [("Blur:", 40), ("Erode:", 40), ("Blur2:", 40), ("Treshold:", 255), ("Min Area:", 2000), ("Max Area:", 2000)]:
            h = QHBoxLayout()
            h.addWidget(QLabel(label_text))
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, max_val)
            slider.setValue(0)
            slider.setFixedWidth(150)
            h.addWidget(slider)
            layout_t.addLayout(h)
            # Save slider as attribute for later access:
            if label_text.startswith("Blur:"):
                self.sl_blur = slider
            elif label_text.startswith("Erode:"):
                self.sl_erode = slider
            elif label_text.startswith("Blur2:"):
                self.sl_blur2 = slider
            elif label_text.startswith("Treshold:"):
                self.sl_tresh = slider
            elif label_text.startswith("Min Area:"):
                self.sl_minarea = slider
            elif label_text.startswith("Max Area:"):
                self.sl_maxarea = slider
            slider.valueChanged.connect(self.updateTresholding)
        self.tresh_group.layout().addLayout(layout_t)
        # Tresholded Image group
        self.tresh_img_group = self.makeGroupBox("Tresholded Image")
        self.tresh_img_label = QLabel()
        self.tresh_img_group.layout().addWidget(self.tresh_img_label)
        self.tresh_img_group.setVisible(False)
        # Assemble left sidebar
        self.left_layout = QVBoxLayout()
        self.left_layout.addWidget(self.info_group)
        self.left_layout.addWidget(self.opmode_group)
        if self.video_group:
            self.left_layout.addWidget(self.video_group)
        self.left_layout.addWidget(self.mode_group)
        self.left_layout.addWidget(self.func_group)
        self.left_layout.addWidget(self.flim_group)
        self.left_layout.addWidget(self.tresh_group)
        self.left_layout.addWidget(self.tresh_img_group)
        self.left_layout.addStretch()
        self.left_widget.setLayout(self.left_layout)
        # Right: Drawing Area
        self.drawing_group = self.makeGroupBox("Drawing Area")
        self.drawing_widget = PyQt5ShapeDrawer(file=file, is_video=self.is_video)
        self.drawing_widget.main_window = self
        self.drawing_group.layout().addWidget(self.drawing_widget)
        # Main Layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.left_widget)
        main_layout.addWidget(self.drawing_group, 1)
        central = QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)
        # Connect signals
        self.mode_combo.currentIndexChanged.connect(self.onModeComboChanged)
        self.drawing_widget.modeChanged.connect(self.onModeChangedByWidget)
        self.cb_blackwhite.stateChanged.connect(self.onBlackWhiteChanged)
        self.cb_showmask.stateChanged.connect(self.onShowMaskChanged)
        self.cb_invertmask.stateChanged.connect(self.onInvertMaskChanged)
        self.hue_slider.valueChanged.connect(self.onHueChanged)
        QTimer.singleShot(0, self.drawing_widget.setFocus)

    def makeGroupBox(self, title):
        grp = QGroupBox(title)
        grp.setStyleSheet(GROUPBOX_STYLE)
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 20, 10, 10)
        grp.setLayout(layout)
        return grp

    def proxyUpdate(self):
        w = self.drawing_widget
        mouse_pt = w.convertToOriginal(w.mouse_pos)
        current_str = f"Current Position ({mouse_pt.x()}, {mouse_pt.y()})"
        last_str = ""
        if w.last_click_orig:
            last_str = f"Last Click ({w.last_click_orig.x()}, {w.last_click_orig.y()})"
        metric = ""
        if w.start_point_orig and w.end_point_orig:
            dx = w.end_point_orig.x() - w.start_point_orig.x()
            dy = w.end_point_orig.y() - w.start_point_orig.y()
            if w.drawing_mode == "rectangle":
                metric = f"Rect width : {abs(dx)}, height : {abs(dy)}"
            elif w.drawing_mode == "line":
                length = int(math.hypot(dx, dy))
                metric = f"Line length : {length}"
            elif w.drawing_mode == "circle":
                radius = int(math.hypot(dx, dy) / 2)
                metric = f"Circle diameter : {2*radius}"
        frame_info = ""
        if self.is_video:
            frame_info = f"Frame : {self.current_frame_idx+1} / {self.total_frames}"
        self.info_label.setText(f"{frame_info}\n{current_str}\n{last_str}\n{metric}")
        w.update()

    # ---------- Video Methods ----------
    def onVideoSliderChanged(self, value):
        if self.is_video and self.cap:
            self.timer.stop()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, value)
            self.nextFrame()

    def onPlayClicked(self):
        if self.is_video and self.cap and self.timer:
            fps = self.fps_spin.value()
            self.timer.start(int(1000 / fps))
            self.drawing_widget.setFocus()

    def onStopClicked(self):
        if self.timer:
            self.timer.stop()
        self.drawing_widget.setFocus()

    def onStartClicked(self):
        if self.is_video and self.cap:
            self.timer.stop()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.nextFrame()
        self.drawing_widget.setFocus()

    def onEndClicked(self):
        if self.is_video and self.cap:
            self.timer.stop()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.total_frames)
            self.nextFrame()
        self.drawing_widget.setFocus()

    def onFrameSpinEditingFinished(self):
        if self.is_video and self.cap:
            value = self.frame_spin.value()
            self.timer.stop()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, value)
            self.nextFrame()
        self.drawing_widget.setFocus()

    def nextFrame(self):
        if not self.is_video or not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return
        self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        self.video_slider.blockSignals(True)
        self.video_slider.setValue(self.current_frame_idx + 1)
        self.video_slider.blockSignals(False)
        qimg = cvMatToQImage(frame)
        self.drawing_widget.setBackgroundImage(qimg)
        self.drawing_widget.update()
        self.proxyUpdate()

    # ---------- Frame Limits Methods ----------
    def onUpdateStart(self):
        if self.is_video:
            self.flim_start_spin.setValue(self.current_frame_idx + 1)
        self.drawing_widget.setFocus()

    def onUpdateStop(self):
        if self.is_video:
            self.flim_stop_spin.setValue(self.current_frame_idx + 1)
        self.drawing_widget.setFocus()

    # ---------- Slots ----------
    def onModeComboChanged(self, index):
        new_mode = self.mode_combo.itemText(index)
        self.drawing_widget.drawing_mode = new_mode
        self.drawing_widget.clearCurrentShape()
        self.drawing_widget.update()

    def onModeChangedByWidget(self, mode):
        idx = self.mode_combo.findText(mode)
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)

    def onBlackWhiteChanged(self, state):
        self.drawing_widget.blackwhite = (state == Qt.Checked)
        self.drawing_widget.update()
        self.drawing_widget.setFocus()

    def onShowMaskChanged(self, state):
        self.drawing_widget.setShowMask(state == Qt.Checked)
        self.drawing_widget.setFocus()

    def onInvertMaskChanged(self, state):
        self.drawing_widget.setInvertMask(state == Qt.Checked)
        self.drawing_widget.setFocus()

    def onHueChanged(self, value):
        self.drawing_widget.drawing_color = QColor.fromHsv(value, 255, 255)
        self.drawing_widget.update()

    def onOpModeChanged(self, index):
        opmode = self.opmode_combo.itemText(index).lower()
        print("Operation mode changed to:", opmode)
        if opmode in ["roi", "conv", "point", "framelimits", "tresholding"]:
            self.mode_group.setVisible(False)
            self.func_group.setVisible(False)
        else:
            self.mode_group.setVisible(True)
            self.func_group.setVisible(True)
        if opmode in ["framelimits", "tresholding"]:
            self.drawing_widget.setEnabled(False)
        else:
            self.drawing_widget.setEnabled(True)
        if opmode == "roi":
            self.drawing_widget.drawing_mode = "rectangle"
        elif opmode == "conv":
            self.drawing_widget.drawing_mode = "line"
        elif opmode == "point":
    elif opmode == "mask":
        self.drawing_widget.drawing_mode = "polygon"
        self.mode_combo.clear(); self.mode_combo.addItems(["rectangle", "polygon", "circle", "ellipse"])
        self.mode_combo.setCurrentText("polygon")
            self.drawing_widget.drawing_mode = "point"
        if opmode == "framelimits":
            self.flim_group.setVisible(True)
        else:
            self.flim_group.setVisible(False)
        if opmode == "tresholding":
            self.tresh_group.setVisible(True)
            self.tresh_img_group.setVisible(True)
            self.drawing_widget.setEnabled(False)
            self.updateTresholding()
        else:
            self.tresh_group.setVisible(False)
            self.tresh_img_group.setVisible(False)
        self.drawing_widget.clearCurrentShape()
        self.drawing_widget.update()

    def updateTresholding(self):
        t_blur = self.sl_blur.value()
        t_erode = self.sl_erode.value()
        t_blur2 = self.sl_blur2.value()
        t_tresh = self.sl_tresh.value()
        t_minarea = self.sl_minarea.value()
        t_maxarea = self.sl_maxarea.value()
        self.tresh_params = {"blur": t_blur, "erode": t_erode, "blur2": t_blur2,
                             "treshold": t_tresh, "min_area": t_minarea, "max_area": t_maxarea}
        if self.is_video and self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx + 1)
            ret, frame = self.cap.read()
            if ret:
                IP = ImgProcessor(frame, None, None, "tresh_bw",
                                  t_blur, t_erode, t_blur2, t_tresh, t_minarea, t_maxarea, simple=True)
                self.img_tresh, _, _ = IP.process()
                qimg = cvMatToQImage(self.img_tresh)
                self.tresh_img_label.setPixmap(QPixmap.fromImage(qimg).scaled(300,300, Qt.KeepAspectRatio))
        else:
            if self.drawing_widget.background_image:
                frame = self.drawing_widget.background_image
                IP = ImgProcessor(frame, None, None, "tresh_bw",
                                  t_blur, t_erode, t_blur2, t_tresh, t_minarea, t_maxarea, simple=True)
                self.img_tresh, _, _ = IP.process()
                qimg = cvMatToQImage(self.img_tresh)
                self.tresh_img_label.setPixmap(QPixmap.fromImage(qimg).scaled(300,300, Qt.KeepAspectRatio))
        self.proxyUpdate()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_F:
            if self.isFullScreen():
                self.showNormal()
                if hasattr(self, 'default_geometry'):
                    self.setGeometry(self.default_geometry)
            else:
                self.default_geometry = self.geometry()
                self.showFullScreen()
        elif key == Qt.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
                if hasattr(self, 'default_geometry'):
                    self.setGeometry(self.default_geometry)
            else:
                super().keyPressEvent(event)
        elif key == Qt.Key_Q:
            self.onStartClicked()
        elif key == Qt.Key_W:
            if self.is_video:
                new = max(0, self.current_frame_idx - self.fps_spin.value())
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, new + 1)
                self.nextFrame()
        elif key == Qt.Key_E:
            if self.is_video:
                new = max(0, self.current_frame_idx - 1)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, new + 1)
                self.nextFrame()
        elif key == Qt.Key_R:
            if self.is_video:
                new = min(self.total_frames - 1, self.current_frame_idx + 1)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, new + 1)
                self.nextFrame()
        elif key == Qt.Key_T:
            if self.is_video:
                new = min(self.total_frames - 1, self.current_frame_idx + self.fps_spin.value())
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, new + 1)
                self.nextFrame()
        elif key == Qt.Key_Y:
            if self.is_video:
                self.onEndClicked()
        else:
            super().keyPressEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        if not hasattr(self, 'default_geometry'):
            self.default_geometry = self.geometry()

# ---------- set_interactive ----------
def set_interactive(file=None, mode="roi"):
    """
    Launches the interactive drawing interface.
    
    Operation Modes and Their Outputs:
      - "roi": Allows only rectangle (ROI) drawing. Output: ("roi", ((top_left_x, top_left_y), (bottom_right_x, bottom_right_y))).
      - "mask": Allows drawing of multiple shapes. Output: ("mask", shapes) where shapes is a list of shape coordinate tuples.
      - "conv": Allows only line drawing. Output: ("conv", line_length) in pixels.
      - "point": Allows drawing of one point per frame (video) or one point for image. Output in video: ("point", {frame: (x,y)}), in image: ("point", [(x,y), ...]).
      - "framelimits": (Video only) Allows setting start and stop frames. Output: ("framelimits", (start, stop)).
      - "tresholding": Disables drawing and shows tresholding controls. Output: ("tresholding", {parameter_dictionary}).
    
    Keys:
      - Escape: Exit fullscreen (if active) or cancel.
      - S: Finalize and return output.
      - T: Cycle drawing modes (only in "mask" mode).
      - A: Manually add shape.
      - C: Toggle crosshair.
      - F: Toggle fullscreen/default window size.
      - Q, W, E, R, T, Y: Video navigation controls.
    
    Returns None if canceled.
    """
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = PyQt5ShapeDrawerWindow(file)
    idx = window.opmode_combo.findText(mode.lower())
    if idx >= 0:
        window.opmode_combo.setCurrentIndex(idx)
    window.show()
    QTimer.singleShot(0, window.drawing_widget.setFocus)
    app.exec_()
    result = window.drawing_widget.final_output
    if result is not None and result[0] == "point" and window.is_video:
        pts_dict = {}
        for frame, pt in window.drawing_widget.points_by_frame.items():
            pts_dict[frame] = (pt.x(), pt.y())
        result = ("point", pts_dict)
    return result

# ---------- Demo ----------
if __name__ == '__main__':
    # Modes: "roi", "mask", "conv", "point", "framelimits", "tresholding"
    result = set_interactive(file="~/Desktop/sample_video.mp4", mode="roi")
    if result is not None:
        print("Output:", result)
    else:
        print("Canceled.")



