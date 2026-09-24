#! /usr/bin/env python

from ._utils import *
from ._canvas import PyQt5ShapeDrawer, GROUPBOX_STYLE
from ._prefs import load_prefs, save_prefs
from ._timeline import TrackingTimeline
from pythutils.mediautils import get_vid_params
from atracker.helpers.media import get_media_type
from atracker.helpers.data import load_and_convert_tracking_dataframe
from atracker.helpers.detection import ProcessImage
from PyQt5.QtWidgets import QMessageBox, QInputDialog
from PyQt5.QtCore import QByteArray

# Maps human-readable purpose labels (lowercased) to internal opmode strings
_PURPOSE_MAP = {
    "mask": "mask",
    "roi": "roi",
    "zones": "zones",
    "frame limits": "framelimits",
    "coordinate data": "timepoints",
    "measurement": "measure",
    "thresholding": "thresholding",
    "thresholding color": "thresholding color",
    # passthrough for existing opmode values
    "framelimits": "framelimits",
    "measure": "measure",
    "points": "points",
    "timepoints": "timepoints",
    "default": "default",
}

_MULTI_FILE_PURPOSES = [
    "Mask", "ROI", "Zones", "Frame limits",
    "Coordinate data", "Measurement", "Thresholding", "Thresholding color"
]

class PyQt5ShapeDrawerWindow(QMainWindow):
    _last_esc_press_time = 0  # Class variable shared across all instances

    def __init__(self, file=None, background_file=None, threshold_dict=None, mask_qimg=None, mode="default",
                 total_frames=1, width=1280, height=960, roi=None,
                 file_infos=None, file_idx=0, save_callback=None):
        super().__init__()

        # --- Multi-file state ---
        self._file_infos = file_infos
        self._file_idx = file_idx
        self._save_callback = save_callback
        self._file_states = {}   # {file_idx: {purpose_text: state_dict}}
        self._saved_states = {}
        self._active_purpose = None
        self._unsaved = set()    # {(file_idx, purpose)} - modified but not saved
        self._stored = set()     # {(file_idx, purpose_text)} - stored this session

        # Override file params from file_infos[file_idx] if multi-file mode
        if file_infos and len(file_infos) > file_idx:
            _fi = file_infos[file_idx]
            if _fi.get("video_path"):
                file = _fi["video_path"]
            if _fi.get("background_path"):
                background_file = _fi["background_path"]
            if roi is None and _fi.get("roi"):
                roi = _fi["roi"]
            if not threshold_dict and _fi.get("threshold_dict"):
                threshold_dict = _fi["threshold_dict"]

        # === 0. GENERAL SETUP ===
        self.setGeometry(0, 0, 1024, 768)
        self.setMinimumSize(1024, 768)
        self.setWindowTitle("ATracker - PyQt5 Shape Drawer")
        self.cap = None
        self.timer = None
        self.current_frame_idx = 0  # 0-indexed for PyQt, 1-indexed for users
        self.arrow_length = 20
        self.was_fullscreen = False
        self.roi = roi
        self.loop = False
        # event-frame buffer for recording frames via key (K)
        self._event_frames = []

        # --- Threshold parameters ---
        self.thresh_params = threshold_dict if threshold_dict else {
            "blur": 9, "erode": 1, "blur2": 1, "threshold": 50,
            "min_area": 100, "max_area": 5000,
            "hue_lo": 30, "hue_hi": 90, "sat_lo": 50, "sat_hi": 255, "val_lo": 50, "val_hi": 255
        }
        self.img_thresh = None
        self.background_file = background_file
        self.edit_drag_active = False
        self.edit_drag_target = None
        self.zones_overlay = None

        # === 1. LOAD MEDIA & SET DIMENSIONS ===
        self.background_image = None

        if file:
            path = os.path.expanduser(file)
            media_type = get_media_type(path)
            if media_type == "vid":
                self.is_video = True
                self.cap = cv2.VideoCapture(path)
                self.fps, self.orig_width, self.orig_height, self.total_frames = get_vid_params(self.cap)
                self.is_video = self.total_frames > 1
            elif media_type == "img":
                self.is_video = False
                self.cap = None
                bg = QImage(path)
                if not bg.isNull():
                    self.background_image = bg
                    self.orig_width = bg.width()
                    self.orig_height = bg.height()
                self.total_frames = 1
                self.is_video = False
            else:
                raise ValueError("File neither video nor image!")
        else:
            # Blank canvas
            self.orig_width = width
            self.orig_height = height
            self.total_frames = total_frames if total_frames else 1
            self.is_video = self.total_frames > 1
            self.background_image = QImage(self.orig_width, self.orig_height, QImage.Format_RGB32)
            self.background_image.fill(QColor("white"))
            self.cap = None
            self.fps = 10

        # === 2. LEFT SIDEBAR SETUP ===
        self.left_widget = QWidget()
        self.left_widget.setFixedWidth(350)
        self.left_layout = QVBoxLayout()
        self.left_widget.setLayout(self.left_layout)

        # --- Info Panel ---
        self.info_group = self.makeGroupBox("Info")
        self.info_label = QLabel("Frame : 0 / 0\nCurrent Position: N/A\nLast Click: N/A")
        self.info_group.layout().addWidget(self.info_label)
        self.left_layout.addWidget(self.info_group)

        # --- File Navigation (multi-file mode only) ---
        if self._file_infos:
            self.nav_group = self.makeGroupBox("File")
            nav_h = QHBoxLayout()
            self.btn_prev = QPushButton("← Prev")
            self.btn_prev.setFixedWidth(65)
            self.btn_prev.clicked.connect(lambda: self._navigate_to(self._file_idx - 1))
            self.nav_name_label = QLabel()
            self.nav_name_label.setAlignment(Qt.AlignCenter)
            self.nav_name_label.setWordWrap(True)
            self.btn_next = QPushButton("Next →")
            self.btn_next.setFixedWidth(65)
            self.btn_next.clicked.connect(lambda: self._navigate_to(self._file_idx + 1))
            nav_h.addWidget(self.btn_prev)
            nav_h.addStretch(1)
            nav_h.addWidget(self.btn_next)
            self.nav_group.layout().addWidget(self.nav_name_label)
            self.nav_group.layout().addLayout(nav_h)
            self.left_layout.addWidget(self.nav_group)
            self._update_nav_label()

        # --- Purpose / Operation Mode ---
        self.opmode_group = self.makeGroupBox("Purpose" if self._file_infos else "Operation Mode")
        self.opmode_combo = QComboBox()
        if self._file_infos:
            self.opmode_combo.addItems(_MULTI_FILE_PURPOSES)
        else:
            self.opmode_combo.addItems([
                "default", "framelimits", "measure", "roi", "mask", "zones", "points",
                "timepoints", "thresholding", "thresholding color"
            ])
        self.opmode_combo.currentIndexChanged.connect(self.onOpModeChanged)
        self.opmode_group.layout().addWidget(self.opmode_combo)
        self.left_layout.addWidget(self.opmode_group)

        # --- Time Controls ---
        self.video_group = None
        if self.is_video and self.total_frames > 1:
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
            h_fps_frame = QHBoxLayout()
            h_fps_frame.addWidget(QLabel("FPS:"))
            self.fps_spin = QSpinBox()
            self.fps_spin.setRange(1, 999)
            self.fps_spin.setValue(25)
            self.fps_spin.editingFinished.connect(self.refocusToCanvas)
            h_fps_frame.addWidget(self.fps_spin)
            h_fps_frame.addSpacing(20)
            h_fps_frame.addWidget(QLabel("Frame:"))
            self.frame_spin = QSpinBox()
            self.frame_spin.setRange(1, max(1, self.total_frames))
            self.frame_spin.editingFinished.connect(self.onFrameSpinEditingFinished)
            h_fps_frame.addWidget(self.frame_spin)
            self.video_group.layout().addLayout(h_fps_frame)
            self.video_slider.valueChanged.connect(self.onVideoSliderChanged)
            self.btn_start.clicked.connect(self.onStartClicked)
            self.btn_play.clicked.connect(self.onPlayClicked)
            self.btn_stop.clicked.connect(self.onStopClicked)
            self.btn_end.clicked.connect(self.onEndClicked)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.nextFrame)
            if self.cap is not None:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                QTimer.singleShot(0, self.nextFrame)
            else:
                self.current_frame_idx = 0
            self.left_layout.addWidget(self.video_group)
        
        # --- Frame limits ---
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
        self.left_layout.addWidget(self.flim_group)

        # --- Drawing Mode ---
        self.mode_group = self.makeGroupBox("Drawing Mode")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["line", "rectangle", "polygon", "circle", "ellipse", "point"])
        self.mode_group.layout().addWidget(self.mode_combo)
        self.left_layout.addWidget(self.mode_group)

        # === 3. FUNCTIONS/VISUALISATIONS ===
        self.func_group = self.makeGroupBox("Drawing functions")
        
        h_hue = QHBoxLayout()
        self.hue_label = QLabel("Drawing Hue:")
        h_hue.addWidget(self.hue_label)
        self.hue_slider = QSlider(Qt.Horizontal)
        self.hue_slider.setRange(0, 359)
        self.hue_slider.setValue(120)
        self.hue_slider.setFixedWidth(150)
        h_hue.addWidget(self.hue_slider)
        self.func_group.layout().addLayout(h_hue)

        h_trans = QHBoxLayout()
        self.bgtrans_label = QLabel("Image Transparency:")
        h_trans.addWidget(self.bgtrans_label)
        self.bgtrans_slider = QSlider(Qt.Horizontal)
        self.bgtrans_slider.setRange(0, 100)
        self.bgtrans_slider.setValue(100)
        self.bgtrans_slider.setFixedWidth(150)
        h_trans.addWidget(self.bgtrans_slider)
        self.func_group.layout().addLayout(h_trans)

        h_gamma = QHBoxLayout()
        self.gamma_label = QLabel("Gamma: 1.00")
        h_gamma.addWidget(self.gamma_label)
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(25, 400)
        self.gamma_slider.setValue(100)
        self.gamma_slider.setMinimumWidth(80)
        self.gamma_slider.setToolTip(
            "Display brightness only: 1.00 is normal; higher values brighten dark areas."
        )
        h_gamma.addWidget(self.gamma_slider, 1)
        self.gamma_reset = QPushButton("Reset")
        self.gamma_reset.setFixedWidth(50)
        self.gamma_reset.clicked.connect(lambda: self.gamma_slider.setValue(100))
        h_gamma.addWidget(self.gamma_reset)
        self.func_group.layout().addLayout(h_gamma)

        h_point_op = QHBoxLayout()
        self.point_op_label = QLabel("Points Opacity:")
        h_point_op.addWidget(self.point_op_label)
        self.point_op_slider = QSlider(Qt.Horizontal)
        self.point_op_slider.setRange(0, 100)
        self.point_op_slider.setValue(100)  # fully opaque by default
        self.point_op_slider.setFixedWidth(150)
        h_point_op.addWidget(self.point_op_slider)
        self.func_group.layout().addLayout(h_point_op)
        self.point_opacity = 1.0
        self.point_op_slider.valueChanged.connect(self.on_point_opacity_changed)

        h_mask_op = QHBoxLayout()
        self.mask_op_label = QLabel("Mask / Zones Opacity:")
        h_mask_op.addWidget(self.mask_op_label)
        self.mask_op_slider = QSlider(Qt.Horizontal)
        self.mask_op_slider.setRange(0, 100)
        self.mask_op_slider.setValue(80)
        self.mask_op_slider.setFixedWidth(150)
        h_mask_op.addWidget(self.mask_op_slider)
        self.func_group.layout().addLayout(h_mask_op)

        h_point_size = QHBoxLayout()
        self.point_size_label = QLabel("Point Size:")
        h_point_size.addWidget(self.point_size_label)
        self.point_size_slider = QSlider(Qt.Horizontal)
        self.point_size_slider.setRange(2, 20)
        self.point_size_slider.setValue(6)
        self.point_size_slider.setFixedWidth(150)
        h_point_size.addWidget(self.point_size_slider)
        self.func_group.layout().addLayout(h_point_size)

        h_ticks = QHBoxLayout()
        self.cb_blackwhite = QCheckBox("Image BW")
        self.cb_showmask = QCheckBox("Mask")
        self.cb_invertmask = QCheckBox("Invert Mask")
        self.cb_showzones = QCheckBox("Zones")
        h_ticks.addWidget(self.cb_blackwhite)
        h_ticks.addWidget(self.cb_showmask)
        h_ticks.addWidget(self.cb_invertmask)
        h_ticks.addWidget(self.cb_showzones)
        self.func_group.layout().addLayout(h_ticks)

        h_helper = QHBoxLayout()
        self.cb_helperlines = QCheckBox("Helperlines")
        self.cb_helperlines.setChecked(False)
        h_helper.addWidget(self.cb_helperlines)
        self.cb_crosshair = QCheckBox("Crosshair")
        self.cb_crosshair.setChecked(False)
        h_helper.addWidget(self.cb_crosshair)
        self.func_group.layout().addLayout(h_helper)
        self.cb_showloop = QCheckBox("Loop")
        self.cb_showloop.setChecked(False)
        h_helper.addWidget(self.cb_showloop)

        h_show_toggles = QHBoxLayout()
        self.cb_showthreshimg = QCheckBox("Threshimage")
        self.cb_showthreshimg.setChecked(True)
        h_show_toggles.addWidget(self.cb_showthreshimg)
        self.cb_showoverlay = QCheckBox("Overlay")
        self.cb_showoverlay.setChecked(True)
        h_show_toggles.addWidget(self.cb_showoverlay)
        self.func_group.layout().addLayout(h_show_toggles)

        self.left_layout.addWidget(self.func_group)
        
        # === 5. THRESHOLDING ===
        self.thresh_group = self.makeGroupBox("Thresholding Controls")
        self.thresh_group.setVisible(False)
        self.slider_labels = {}
        self.sliders_bw_widgets = []
        self.sliders_color_widgets = []
        # B/W Sliders
        _frame_px = self.orig_width * self.orig_height
        for label_text, max_val in [
            ("Blur:", 39), ("Erode:", 39), ("Blur2:", 39), ("Threshold:", 255),
            ("Min Area:", _frame_px), ("Max Area:", _frame_px)
        ]:
            slider = QSlider(Qt.Horizontal)
            slider.setRange(1, max_val)
            param_key = label_text.lower().replace(":", "").replace(" ", "_")
            default_val = self.thresh_params.get(param_key, 1)
            label = QLabel(label_text)
            label.setFixedWidth(80)
            value_spin = QSpinBox()
            value_spin.setRange(1, max_val)
            value_spin.setValue(default_val)
            value_spin.setFixedWidth(70)
            slider.setValue(default_val)
            slider.setFixedWidth(150)
            slider.valueChanged.connect(value_spin.setValue)
            value_spin.valueChanged.connect(slider.setValue)
            slider.valueChanged.connect(lambda _: self.updateThresholdingImage())
            h = QHBoxLayout()
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(label)
            h.addWidget(value_spin)
            h.addWidget(slider)
            container = QWidget()
            container.setLayout(h)
            self.thresh_group.layout().addWidget(container)
            self.sliders_bw_widgets.append((label_text, slider, container))
            # Save references
            if label_text.startswith("Blur:"):
                self.sl_blur = slider
            elif label_text.startswith("Erode:"):
                self.sl_erode = slider
            elif label_text.startswith("Blur2:"):
                self.sl_blur2 = slider
            elif label_text.startswith("Threshold:"):
                self.sl_thresh = slider
            elif label_text.startswith("Min Area:"):
                self.sl_minarea = slider
            elif label_text.startswith("Max Area:"):
                self.sl_maxarea = slider

        # Add the hue panel
        self.hue_panel_label = QLabel()
        self.hue_panel_label.setFixedSize(150, 15)
        self.hue_panel_label.setVisible(False)
        hue_layout = QHBoxLayout()
        hue_layout.setContentsMargins(0, 0, 0, 0)
        hue_layout.addSpacing(145)
        hue_layout.addWidget(self.hue_panel_label)
        self.thresh_group.layout().addLayout(hue_layout)
        # HSV Sliders
        for label_text, max_val in [
            ("Hue Low:", 179), ("Hue High:", 179),
            ("Sat Low:", 255), ("Sat High:", 255),
            ("Val Low:", 255), ("Val High:", 255)
        ]:
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, max_val)
            param_key = label_text.lower().replace(" ", "_").replace(":", "").replace("low", "lo").replace("high", "hi")
            default_val = self.thresh_params[param_key] if param_key in self.thresh_params else (0 if "lo" in param_key else max_val)
            slider.setValue(default_val)
            slider.setFixedWidth(150)
            h = QHBoxLayout()
            h.setContentsMargins(0, 0, 0, 0)
            label = QLabel(label_text)
            label.setFixedWidth(80)
            value_spin = QSpinBox()
            value_spin.setRange(0, max_val)
            value_spin.setValue(default_val)
            value_spin.setFixedWidth(50)
            h.addWidget(label)
            h.addWidget(value_spin)
            h.addWidget(slider)

            # Sync slider and spinbox both ways
            slider.valueChanged.connect(value_spin.setValue)
            value_spin.valueChanged.connect(slider.setValue)
            slider.valueChanged.connect(lambda _, s=self: (s.updateThresholdingImage(), s.updateHuePanel()))
            container = QWidget()
            container.setLayout(h)
            container.setVisible(True)
            self.thresh_group.layout().addWidget(container)
            self.sliders_color_widgets.append((label_text, slider, container))
            if label_text.startswith("Hue Low:"):
                self.sl_hue_lo = slider
            elif label_text.startswith("Hue High:"):
                self.sl_hue_hi = slider
            elif label_text.startswith("Sat Low:"):
                self.sl_sat_lo = slider
            elif label_text.startswith("Sat High:"):
                self.sl_sat_hi = slider
            elif label_text.startswith("Val Low:"):
                self.sl_val_lo = slider
            elif label_text.startswith("Val High:"):
                self.sl_val_hi = slider

        self.left_layout.addWidget(self.thresh_group)

        # Thresholded Image group
        self.thresh_img_group = self.makeGroupBox("Thresholded Image")
        self.thresh_img_label = QLabel()
        self.thresh_img_group.layout().addWidget(self.thresh_img_label)
        self.thresh_img_group.setVisible(False)
        self.left_layout.addWidget(self.thresh_img_group)

        # === 5. Timepoints Editing Group ===
        self.timepoints_group = self.makeGroupBox("Timepoints Editing")
        timepoints_layout = self.timepoints_group.layout()

        # ---- TIMEPOINTS ENHANCEMENTS ----
        self.tp_show_lines = False
        self.tp_total_ids = 1
        self.tp_current_id = 0
        self.tp_visible_range = 100

        # ---- POINT TYPE SELECTION + HIGHLIGHT CURRENT ----
        self.ptype_options = [("Centroid", "c"), ("Head", "h"), ("Tail", "t")]
        self.current_ptype_idx = 0
        self.current_ptype = self.ptype_options[self.current_ptype_idx][1]
        h_ptype = QHBoxLayout()
        h_ptype.addWidget(QLabel("Point type:"))
        self.ptype_dropdown = QComboBox()
        for name, code in self.ptype_options:
            self.ptype_dropdown.addItem(name, code)
        self.ptype_dropdown.setCurrentIndex(self.current_ptype_idx)
        self.ptype_dropdown.currentIndexChanged.connect(self.on_ptype_change)
        h_ptype.addWidget(self.ptype_dropdown)

        # ---- HIGHLIGHT CURRENT FRAME (move here) ----
        self.highlight_current_checkbox = QCheckBox("Highlight current")
        self.highlight_current_checkbox.setChecked(True)
        h_ptype.addSpacing(20)  # Optional: some space between selector and checkbox
        h_ptype.addWidget(self.highlight_current_checkbox)
        h_ptype.addStretch()
        timepoints_layout.addLayout(h_ptype)


        # ---- ID CREATION AND SELECTION ----
        h_id_inputs = QHBoxLayout()
        h_id_inputs.addWidget(QLabel("Nr IDs:"))
        self.input_num_ids = QSpinBox()
        self.input_num_ids.setMinimum(1)
        self.input_num_ids.setMaximum(101)
        self.input_num_ids.setValue(1)
        self.input_num_ids.valueChanged.connect(self.updateNumIDs)
        self.input_num_ids.editingFinished.connect(self.refocusToCanvas)
        h_id_inputs.addWidget(self.input_num_ids)
        h_id_inputs.addSpacing(10)
        h_id_inputs.addWidget(QLabel("Current ID:"))
        self.current_id_box = QSpinBox()
        self.current_id_box.setMinimum(1)
        self.current_id_box.setValue(1)
        self.current_id_box.valueChanged.connect(self.updateCurrentID)
        self.current_id_box.editingFinished.connect(self.refocusToCanvas)
        h_id_inputs.addWidget(self.current_id_box)
        timepoints_layout.addLayout(h_id_inputs)

        # ---- ID COLOR LEGEND ----
        self.id_color_buttons = []
        self.id_grid_widget = QWidget()
        self.id_grid_layout = QGridLayout(self.id_grid_widget)
        self.id_grid_layout.setSpacing(2)
        self.id_grid_layout.setAlignment(Qt.AlignLeft) 
        self.id_grid_widget.setLayout(self.id_grid_layout)
        timepoints_layout.addWidget(self.id_grid_widget)
        id_grid_row = QHBoxLayout()
        id_grid_row.addWidget(self.id_grid_widget)
        id_grid_row.addStretch()
        timepoints_layout.addLayout(id_grid_row)

        # ---- VISIBLE RANGE SELECTOR ----
        h_range = QHBoxLayout()
        label = QLabel("Visible range:")
        h_range.addWidget(label)
        self.range_slider = QSlider(Qt.Horizontal)
        self.range_slider.setMinimum(0)
        self.range_slider.setMaximum(self.total_frames)
        self.range_slider.setValue(self.tp_visible_range)
        self.range_slider.setFixedWidth(130)
        h_range.addWidget(self.range_slider)
        self.range_value_box = QSpinBox()
        self.range_value_box.setRange(0, self.total_frames)
        self.range_value_box.setValue(self.tp_visible_range)
        self.range_value_box.setFixedWidth(60)
        h_range.addWidget(self.range_value_box)
        self.range_value_box.valueChanged.connect(self.range_slider.setValue)
        self.range_value_box.editingFinished.connect(self.refocusToCanvas)
        timepoints_layout.addLayout(h_range)

        # ---- RANGE DIRECTION ----
        h_range_dir = QHBoxLayout()
        h_range_dir.addWidget(QLabel("Direction:"))
        self.rb_range_all = QRadioButton("All")
        self.rb_range_all.setChecked(True)
        self.rb_range_past = QRadioButton("Past")
        self.rb_range_future = QRadioButton("Future")
        self.range_dir_group = QButtonGroup()
        self.range_dir_group.addButton(self.rb_range_all)
        self.range_dir_group.addButton(self.rb_range_past)
        self.range_dir_group.addButton(self.rb_range_future)
        h_range_dir.addWidget(self.rb_range_all)
        h_range_dir.addWidget(self.rb_range_past)
        h_range_dir.addWidget(self.rb_range_future)
        timepoints_layout.addLayout(h_range_dir)

        # ---- SCOPE RADIO BUTTONS ----
        h_scope = QHBoxLayout()
        h_scope.addWidget(QLabel("Selector:"))
        self.rb_scope_any = QRadioButton("Any ID")
        self.rb_scope_any.setChecked(True)
        self.rb_scope_id = QRadioButton("Current ID")
        h_scope.addWidget(self.rb_scope_any)
        h_scope.addWidget(self.rb_scope_id)
        self.scope_group = QButtonGroup()
        self.scope_group.addButton(self.rb_scope_any)
        self.scope_group.addButton(self.rb_scope_id)
        def scope_mode_changed():
            if self.rb_scope_id.isChecked():
                self.edit_scope = "id"
                self.drawing_widget.scope_mode = "id"
                self.current_id_box.setValue(1)  # triggers valueChanged
                self.drawing_widget.tp_current_id = 0  # 0-based for ID1
            else:
                self.edit_scope = "any"
                self.drawing_widget.scope_mode = "any"
            self.drawing_widget.update()
        self.rb_scope_any.toggled.connect(scope_mode_changed)
        self.rb_scope_id.toggled.connect(scope_mode_changed)
        self.edit_scope = "any"
        timepoints_layout.addLayout(h_scope)
        
        # ---- MODE SELECTOR ----
        h_mode = QVBoxLayout()
        h_mode.addWidget(QLabel("Mode:"))
        self.rb_draw = QRadioButton("(re)Draw point at current frame")
        self.rb_draw.setChecked(True)
        self.rb_move = QRadioButton("Move point nearest to mouse")
        self.rb_angle = QRadioButton("Change angle of nearest point")
        h_mode.addWidget(self.rb_draw)
        h_mode.addWidget(self.rb_move)
        h_mode.addWidget(self.rb_angle)
        self.mode_button_group = QButtonGroup()
        self.mode_button_group.addButton(self.rb_draw)
        self.mode_button_group.addButton(self.rb_move)
        self.mode_button_group.addButton(self.rb_angle)
        timepoints_layout.addLayout(h_mode)
        def on_mode_changed():
            if self.rb_angle.isChecked():
                self.drawing_widget.move_mode_active = False
                self.drawing_widget.angle_mode_active = True
            elif self.rb_move.isChecked():
                self.drawing_widget.move_mode_active = True
                self.drawing_widget.angle_mode_active = False
            else:
                self.drawing_widget.move_mode_active = False
                self.drawing_widget.angle_mode_active = False
            self.drawing_widget.update()
        self.rb_draw.toggled.connect(on_mode_changed)
        self.rb_move.toggled.connect(on_mode_changed)
        self.rb_angle.toggled.connect(on_mode_changed)

        # ---- ACTION BUTTONS ----
        h_actions = QHBoxLayout()
        h_actions.addWidget(QLabel("Delete pts:"))
        self.btn_delete_last = QPushButton("Last")
        self.btn_delete_last.clicked.connect(self.deleteCurrentPoint)
        h_actions.addWidget(self.btn_delete_last)
        self.btn_delete_cropped = QPushButton("In shape")
        self.btn_delete_cropped.clicked.connect(self.deleteCroppedPoints)
        h_actions.addWidget(self.btn_delete_cropped)
        self.btn_delete_visible = QPushButton("Visible")
        self.btn_delete_visible.clicked.connect(self.deleteVisiblePoints)
        h_actions.addWidget(self.btn_delete_visible)
        timepoints_layout.addLayout(h_actions)

        # ---- SHOW POINTS, LINES, FRAMENRS ----
        h_show_options = QHBoxLayout()
        h_show_options.addWidget(QLabel("Show: "))
        self.show_points_checkbox = QCheckBox("Points")
        self.show_points_checkbox.setChecked(True)
        h_show_options.addWidget(self.show_points_checkbox)
        self.show_lines_checkbox = QCheckBox("Lines")
        self.show_lines_checkbox.setChecked(False)
        self.show_lines_checkbox.stateChanged.connect(self.toggleLineVisibility)
        h_show_options.addWidget(self.show_lines_checkbox)
        self.show_framenrs_checkbox = QCheckBox("Frames")
        self.show_framenrs_checkbox.setChecked(False)
        h_show_options.addWidget(self.show_framenrs_checkbox)
        self.show_nearframe_checkbox = QCheckBox("Near")
        self.show_nearframe_checkbox.setChecked(False)
        h_show_options.addWidget(self.show_nearframe_checkbox)
        self.fade_points_checkbox = QCheckBox("Fade")
        self.fade_points_checkbox.setChecked(False)
        self.fade_points_checkbox.setToolTip("Fade points with distance from current frame")
        h_show_options.addWidget(self.fade_points_checkbox)
        timepoints_layout.addLayout(h_show_options)      
        self.left_layout.addWidget(self.timepoints_group)

        # ---- ANGLES VISIBILITY ----
        h_angle_options = QHBoxLayout()
        h_angle_options.addWidget(QLabel("Angles: "))
        self.angle_show_arrows_checkbox = QCheckBox("Arrows")
        self.angle_show_arrows_checkbox.setChecked(False)
        h_angle_options.addWidget(self.angle_show_arrows_checkbox)
        self.angle_show_lines_checkbox = QCheckBox("Lines")
        self.angle_show_lines_checkbox.setChecked(False)
        h_angle_options.addWidget(self.angle_show_lines_checkbox)
        timepoints_layout.addLayout(h_angle_options)

        # ---- ARROW LENGTH SELECTOR ----
        h_arrow_length = QHBoxLayout()
        label_arrow = QLabel("Arrow length:")
        h_arrow_length.addWidget(label_arrow)
        self.arrow_length_slider = QSlider(Qt.Horizontal)
        self.arrow_length_slider.setMinimum(0)
        self.arrow_length_slider.setMaximum(50)
        self.arrow_length_slider.setValue(20)
        self.arrow_length_slider.setFixedWidth(130)
        h_arrow_length.addWidget(self.arrow_length_slider)
        self.arrow_length_value_box = QSpinBox()
        self.arrow_length_value_box.setRange(1, 50)
        self.arrow_length_value_box.setValue(20)
        self.arrow_length_value_box.setFixedWidth(60)
        h_arrow_length.addWidget(self.arrow_length_value_box)
        self.arrow_length_value_box.valueChanged.connect(self.arrow_length_slider.setValue)
        self.arrow_length_slider.valueChanged.connect(self.arrow_length_value_box.setValue)
        self.arrow_length_value_box.editingFinished.connect(self.refocusToCanvas)
        timepoints_layout.addLayout(h_arrow_length)

        # ---- UNDO/REDO ROW ----
        h_undo = QHBoxLayout()
        self.btn_undo = QPushButton("Undo")
        self.btn_undo.setFixedWidth(60)
        h_undo.addWidget(self.btn_undo)
        h_undo.addStretch()
        self.btn_undo.clicked.connect(self.undoLastDelete)
        # Optionally add redo here if you implement it
        # self.btn_redo = QPushButton("Redo")
        # self.btn_redo.clicked.connect(self.redoLastDelete)
        # h_undo.addWidget(self.btn_redo)
        timepoints_layout.addLayout(h_undo)

        self.left_layout.addStretch()

        # --- Actions (multi-file mode only) ---
        if self._file_infos:
            self.action_group = self.makeGroupBox("")
            actions_h = QHBoxLayout()
            self.btn_store = QPushButton("Save current")
            self.btn_store.setToolTip("Store current drawing for this file and purpose")
            self.btn_store.clicked.connect(self._store_current)
            self.btn_save_all = QPushButton("Close")
            self.btn_save_all.setToolTip("Close the editor; prompt if edits remain unsaved")
            self.btn_save_all.clicked.connect(self._save_all)
            actions_h.addWidget(self.btn_store)
            actions_h.addWidget(self.btn_save_all)
            self.action_group.layout().addLayout(actions_h)
            self.left_layout.addWidget(self.action_group)

        # === 6. DRAWING AREA (RIGHT) ===
        self.drawing_group = self.makeGroupBox("Drawing Area")
        self.drawing_widget = PyQt5ShapeDrawer(
            file=file,
            is_video=self.is_video,
            mask_qimg=mask_qimg,
            width=self.orig_width,
            height=self.orig_height,
            mode=mode
        )
        if self.background_image is not None:
            self.drawing_widget.setBackgroundImage(self.background_image)
        self.drawing_widget.main_window = self
        self.drawing_group.layout().addWidget(self.drawing_widget)
        self.drawing_widget.update()
        self.cb_showoverlay.stateChanged.connect(self.drawing_widget.update) # Trigger a redraw when toggled
            
        # === 7. SIGNAL CONNECTIONS (draw_widget must be created first) ===
        self.updateIDButtons()
        self.cb_helperlines.stateChanged.connect(lambda state: self.drawing_widget.setHelperlinesEnabled(state == Qt.Checked))
        self.cb_crosshair.stateChanged.connect(lambda state: setattr(self.drawing_widget, "crosshair_enabled", state == Qt.Checked) or self.drawing_widget.update())
        self.cb_showthreshimg.stateChanged.connect(lambda state: self.thresh_img_group.setVisible(state == Qt.Checked))
        self.range_slider.valueChanged.connect(self.updateVisibleRange)
        self.show_points_checkbox.stateChanged.connect(self.drawing_widget.update)
        self.show_lines_checkbox.stateChanged.connect(self.toggleLineVisibility)
        self.show_lines_checkbox.stateChanged.connect(self.drawing_widget.update)
        self.show_framenrs_checkbox.stateChanged.connect(self.drawing_widget.update)
        self.show_nearframe_checkbox.stateChanged.connect(self.drawing_widget.update)
        self.cb_blackwhite.stateChanged.connect(self.onBlackWhiteChanged)
        self.cb_showmask.stateChanged.connect(self.onShowMaskChanged)
        self.cb_invertmask.stateChanged.connect(self.onInvertMaskChanged)
        self.cb_showzones.stateChanged.connect(self.onShowZonesChanged)
        self.hue_slider.valueChanged.connect(self.onHueChanged)
        self.mode_combo.currentIndexChanged.connect(self.onModeComboChanged)
        self.drawing_widget.modeChanged.connect(self.onModeChangedByWidget)
        self.angle_show_arrows_checkbox.stateChanged.connect(self.drawing_widget.update)
        self.angle_show_lines_checkbox.stateChanged.connect(self.drawing_widget.update)
        self.bgtrans_slider.valueChanged.connect(self.drawing_widget.update)
        self.gamma_slider.valueChanged.connect(self.onGammaChanged)
        self.mask_op_slider.valueChanged.connect(self.drawing_widget.update)
        self.point_size_slider.valueChanged.connect(self.drawing_widget.update)
        self.rb_range_all.toggled.connect(self.drawing_widget.update)
        self.rb_range_past.toggled.connect(self.drawing_widget.update)
        self.rb_range_future.toggled.connect(self.drawing_widget.update)
        self.fade_points_checkbox.stateChanged.connect(self.drawing_widget.update)
        self.cb_showloop.stateChanged.connect(
            lambda state: setattr(self.drawing_widget, "loop", state == Qt.Checked) or self.drawing_widget.update()
        )
        # Sync selector mode
        self.rb_scope_any.toggled.connect(lambda: setattr(self.drawing_widget, "scope_mode", "any"))
        self.rb_scope_id.toggled.connect(lambda: setattr(self.drawing_widget, "scope_mode", "id"))
        self.rb_scope_any.toggled.connect(self.drawing_widget.update)
        self.rb_scope_id.toggled.connect(self.drawing_widget.update)
        # Sync current ID
        self.current_id_box.valueChanged.connect(lambda val: setattr(self.drawing_widget, "tp_current_id", val - 1))
        self.current_id_box.valueChanged.connect(self.drawing_widget.update)
        self.highlight_current_checkbox.stateChanged.connect(self.drawing_widget.update)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()
        QTimer.singleShot(0, self.drawing_widget.setFocus)
        self.onOpModeChanged(self.opmode_combo.currentIndex())
        self.arrow_length_slider.valueChanged.connect(self.updateArrowLength)
        self.arrow_length_value_box.valueChanged.connect(self.updateArrowLength)
           
        # === 8. MAIN LAYOUT ===
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.left_widget)
        main_layout.addWidget(self.drawing_group, 1)
        self.timeline_group = QGroupBox("Tracking coverage — click or drag to seek")
        self.timeline_group.setCheckable(True)
        self.timeline_group.setChecked(True)
        timeline_layout = QVBoxLayout(self.timeline_group)
        self.timeline_use_range = QCheckBox("Use visible range")
        self.timeline_use_range.setToolTip(
            "Expand the visible frame range across the timeline, using All / Past / Future."
        )
        timeline_layout.addWidget(self.timeline_use_range)
        self.timeline_scroll = QScrollArea()
        self.timeline_scroll.setWidgetResizable(True)
        self.timeline_scroll.setMaximumHeight(180)
        self.timeline_scroll.setMinimumHeight(90)
        self.timeline = TrackingTimeline(self)
        self.timeline_scroll.setWidget(self.timeline)
        timeline_layout.addWidget(self.timeline_scroll)
        self.timeline_group.toggled.connect(self.timeline_scroll.setVisible)
        self.timeline_group.toggled.connect(self.timeline_use_range.setVisible)
        self.timeline_use_range.toggled.connect(self.timeline.update)
        self.range_slider.valueChanged.connect(self.timeline.update)
        for button in (self.rb_range_all, self.rb_range_past, self.rb_range_future):
            button.toggled.connect(self.timeline.update)
        self.flim_start_spin.valueChanged.connect(self.timeline.invalidate)
        self.flim_stop_spin.valueChanged.connect(self.timeline.invalidate)
        outer_layout = QVBoxLayout()
        outer_layout.addLayout(main_layout, 1)
        outer_layout.addWidget(self.timeline_group)
        central = QWidget()
        central.setLayout(outer_layout)
        self.setCentralWidget(central)
        if self._file_infos:
            self._set_timeline_limits(self._file_infos[self._file_idx])
        self.syncTimeline()

        # === 9. LOAD SAVED PREFERENCES ===
        self._apply_prefs(load_prefs())


    def _set_timeline_limits(self, info):
        total = max(1, self.total_frames)
        self.flim_start_spin.setRange(1, total)
        self.flim_stop_spin.setRange(1, total)
        for spin, key, default in [(self.flim_start_spin, "frame_start", 1),
                                   (self.flim_stop_spin, "frame_stop", total)]:
            value = info.get(key)
            spin.setValue(int(value) if value is not None and pd.notna(value) else default)

    def syncTimeline(self):
        if hasattr(self, "timeline"):
            self.timeline_group.setVisible(self.currentOperationMode() == "timepoints")
            self.timeline.sync()

    def updateIDButtons(self):
        # Remove old buttons from the layout
        for btn in self.id_color_buttons:
            btn.deleteLater()
        self.id_color_buttons = []

        num_ids = self.input_num_ids.value()
        self.tp_id_colors = generate_distinct_colors(num_ids)

        max_cols = 10
        for idx in range(num_ids):
            color = self.tp_id_colors[idx]
            is_selected = (self.current_id_box.value() - 1 == idx) and self.rb_scope_id.isChecked()
            border_style = "1px solid #000" if is_selected else "1px solid #FFF"
            btn = QPushButton(str(idx + 1))
            btn.setCheckable(False)  # Set to True if you want toggling
            btn.setFixedSize(25, 25)
            btn.setStyleSheet(
                f"background: {color.name()}; border: {border_style}; color: #fff; font-size:11px;"
            )
            # Connect click only if "Current ID" radio is checked
            btn.clicked.connect(lambda checked, i=idx: self.onIDButtonClicked(i))
            row = idx // max_cols
            col = idx % max_cols
            self.id_grid_layout.addWidget(btn, row, col)
            self.id_color_buttons.append(btn)
    
    
    def updateArrowLength(self, val):
        self.arrow_length = val
        self.drawing_widget.arrow_length = val
        self.drawing_widget.update()
    
    def toggleEditPointMode(self, enabled):
        self.input_num_ids.setEnabled(not enabled)
        self.current_id_box.setEnabled(not enabled)

    def onGammaChanged(self, value):
        self.gamma_label.setText(f"Gamma: {value / 100.0:.2f}")
        self.drawing_widget.update()

    def on_point_opacity_changed(self, value):
        self.point_opacity = value / 100.0
        self.update()  # or self.proxyUpdate() if you have it

    def undoLastDelete(self):
        self.timeline.invalidate()
        buf = getattr(self, "_undo_buffer", {})
        if not buf:
            print("Nothing to undo.")
            return

        if "actions" in buf:
            actions = buf["actions"]
            pw = self.drawing_widget.points_by_frame

            for scope, id_, ptype, restored in actions:
                if not isinstance(restored, dict):
                    continue
                if id_ not in pw:
                    pw[id_] = {'c': {}, 'h': {}, 't': {}, 'a': {}}
                if ptype not in pw[id_]:
                    pw[id_][ptype] = {}

                for frame, value in restored.items():
                    if value is not None:
                        pw[id_][ptype][frame] = value
                        print(f"Restored point for ID={id_}, frame={frame}.")
                    else:
                        if frame in pw[id_][ptype]:
                            del pw[id_][ptype][frame]
                        print(f"Removed point for ID={id_}, frame={frame}.")
            print("Undo completed.")
            self._undo_buffer = {}
            self.drawing_widget.update()
            return

        # fallback for legacy undo buffer format
        id_ = buf.get("id")
        ptype = buf.get("ptype")
        restored = buf.get("points")
        if not isinstance(restored, dict):
            print("Undo buffer does not contain valid points.")
            return

        pw = self.drawing_widget.points_by_frame

        if id_ not in pw:
            pw[id_] = {'c': {}, 'h': {}, 't': {}, 'a': {}}
        if ptype not in pw[id_]:
            pw[id_][ptype] = {}

        for frame, value in restored.items():
            if value is not None:
                pw[id_][ptype][frame] = value
                print(f"Restored point for ID={id_}, frame={frame}.")
            else:
                if frame in pw[id_][ptype]:
                    del pw[id_][ptype][frame]
                print(f"Removed point for ID={id_}, frame={frame}.")
        self._undo_buffer = {}
        self.drawing_widget.update()

        

    def pointInsideShape(self, pt: QPoint, mode, shape_data):
        if mode == "rectangle":
            rect = QRect(QPoint(*shape_data[0]), QPoint(*shape_data[2])).normalized()
            return rect.contains(pt)
        elif mode == "circle":
            (cx, cy), radius = shape_data
            dx = pt.x() - cx
            dy = pt.y() - cy
            return dx * dx + dy * dy <= radius * radius
        elif mode == "ellipse":
            (cx, cy), (rx, ry) = shape_data
            dx = pt.x() - cx
            dy = pt.y() - cy
            return (dx * dx) / (rx * rx) + (dy * dy) / (ry * ry) <= 1
        elif mode == "polygon":
            poly = QPolygon([QPoint(x, y) for x, y in shape_data])
            return poly.containsPoint(pt, Qt.OddEvenFill)
        return False


    def deleteCroppedPoints(self):
        self.timeline.invalidate()
        mode = self.drawing_widget.drawing_mode
        shape_data = None

        # --- Get current drawn shape as before ---
        if mode == "rectangle":
            s = self.drawing_widget.start_point_orig
            e = self.drawing_widget.end_point_orig
            if s and e:
                x1, y1 = s.x(), s.y()
                x2, y2 = e.x(), e.y()
                shape_data = ((min(x1, x2), min(y1, y2)),
                            (max(x1, x2), min(y1, y2)),
                            (max(x1, x2), max(y1, y2)),
                            (min(x1, x2), max(y1, y2)))
            else:
                print("No shape drawn..")
                return
        elif mode == "polygon":
            pts = self.drawing_widget.polygon_points_orig
            if pts and len(pts) > 2:
                shape_data = [(pt.x(), pt.y()) for pt in pts]
            else:
                print("No shape drawn..")
                return
        elif mode == "circle":
            s = self.drawing_widget.start_point_orig
            e = self.drawing_widget.end_point_orig
            if s and e:
                cx = (s.x() + e.x()) // 2
                cy = (s.y() + e.y()) // 2
                r = int(math.hypot(e.x() - s.x(), e.y() - s.y()) / 2)
                shape_data = ((cx, cy), r)
            else:
                print("No shape drawn..")
                return
        elif mode == "ellipse":
            s = self.drawing_widget.start_point_orig
            e = self.drawing_widget.end_point_orig
            if s and e:
                cx = (s.x() + e.x()) // 2
                cy = (s.y() + e.y()) // 2
                rx = abs(e.x() - s.x()) // 2
                ry = abs(e.y() - s.y()) // 2
                shape_data = ((cx, cy), (rx, ry))
            else:
                print("No shape drawn..")
                return
        else:
            print("Switch to rectangle, polygon, circle, or ellipse mode to draw a shape.")
            return

        cur_frame = self.current_frame_idx + 1
        visible_range = self.tp_visible_range
        cur_ptype = self.current_ptype if hasattr(self, "current_ptype") else "c"
        scope = getattr(self, "edit_scope", None)
        deleted = []

        if scope == "any":
            # All IDs for current point type
            for id_, ptypedict in self.drawing_widget.points_by_frame.items():
                frame_dict = ptypedict.get(cur_ptype, {})
                to_delete = []
                for f, pt in frame_dict.items():
                    f_int = int(f)
                    if abs(f_int - cur_frame) > visible_range:
                        continue
                    if self.pointInsideShape(pt, mode, shape_data):
                        to_delete.append(f)
                if to_delete:
                    deleted.append(("any", id_, cur_ptype, {f: frame_dict[f] for f in to_delete}))
                    for f in to_delete:
                        del frame_dict[f]
        elif scope == "id":
            cur_id = self.tp_current_id
            frame_dict = self.drawing_widget.points_by_frame.get(cur_id, {}).get(cur_ptype, {})
            to_delete = []
            for f, pt in frame_dict.items():
                f_int = int(f)
                if abs(f_int - cur_frame) > visible_range:
                    continue
                if self.pointInsideShape(pt, mode, shape_data):
                    to_delete.append(f)
            if to_delete:
                deleted.append(("id", cur_id, cur_ptype, {f: frame_dict[f] for f in to_delete}))
                for f in to_delete:
                    del frame_dict[f]
        else:
            print("No edit scope selected.")
            return

        if deleted:
            # Only store the last action for undo (could do a stack if you want multiple levels)
            self._undo_buffer = {
                "actions": deleted
            }
            print(f"Deleted {sum(len(a[3]) for a in deleted)} points inside shape")
        else:
            print("No points to delete inside shape.")
        self.drawing_widget.update()


    def deleteCurrentPoint(self):
        self.timeline.invalidate()
        cur_frame = self.current_frame_idx + 1
        cur_ptype = self.current_ptype if hasattr(self, "current_ptype") else "c"
        scope = getattr(self, "edit_scope", None)
        deleted = []

        if scope == "any":
            for id_, ptypedict in self.drawing_widget.points_by_frame.items():
                frame_dict = ptypedict.get(cur_ptype, {})
                if cur_frame in frame_dict:
                    deleted.append(("any", id_, cur_ptype, {cur_frame: frame_dict[cur_frame]}))
                    del frame_dict[cur_frame]
        elif scope == "id":
            cur_id = self.tp_current_id
            frame_dict = self.drawing_widget.points_by_frame.get(cur_id, {}).get(cur_ptype, {})
            if cur_frame in frame_dict:
                deleted.append(("id", cur_id, cur_ptype, {cur_frame: frame_dict[cur_frame]}))
                del frame_dict[cur_frame]
        else:
            print("No edit scope selected.")
            return

        if deleted:
            self._undo_buffer = {
                "actions": deleted
            }
            print(f"Deleted {len(deleted)} point(s) for frame {cur_frame}")
        else:
            print("No points to delete.")
        self.drawing_widget.update()


    def deleteVisiblePoints(self):
        self.timeline.invalidate()
        cur_frame = self.current_frame_idx + 1
        visible_range = self.tp_visible_range
        start = max(0, cur_frame - visible_range)
        end = cur_frame + visible_range
        cur_ptype = self.current_ptype if hasattr(self, "current_ptype") else "c"
        scope = getattr(self, "edit_scope", None)
        deleted = []

        if scope == "any":
            for id_, ptypedict in self.drawing_widget.points_by_frame.items():
                frame_dict = ptypedict.get(cur_ptype, {})
                to_delete = {int(f): pt for f, pt in frame_dict.items() if start <= int(f) <= end}
                if to_delete:
                    deleted.append(("any", id_, cur_ptype, to_delete))
                    for f in to_delete:
                        del frame_dict[f]
        elif scope == "id":
            cur_id = self.tp_current_id
            frame_dict = self.drawing_widget.points_by_frame.get(cur_id, {}).get(cur_ptype, {})
            to_delete = {int(f): pt for f, pt in frame_dict.items() if start <= int(f) <= end}
            if to_delete:
                deleted.append(("id", cur_id, cur_ptype, to_delete))
                for f in to_delete:
                    del frame_dict[f]
        else:
            print("No edit scope selected.")
            return

        if deleted:
            self._undo_buffer = {
                "actions": deleted
            }
            print(f"Deleted {sum(len(a[3]) for a in deleted)} points in visible range")
        else:
            print("No points to delete in visible range.")
        self.drawing_widget.update()
 
    def refocusToCanvas(self):
        QTimer.singleShot(0, self.drawing_widget.setFocus)
      
    def makeGroupBox(self, title):
        grp = QGroupBox(title)
        grp.setStyleSheet(GROUPBOX_STYLE)
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 20, 10, 10)
        grp.setLayout(layout)
        return grp

    def proxyUpdate(self):
        self.syncTimeline()
        # Track unsaved changes in multi-file mode

        w = self.drawing_widget
        mouse_pt = w.convertToOriginal(w.mouse_pos)
        current_str = f"Current position: ({mouse_pt.x()}, {mouse_pt.y()})"
        if w.last_click_orig:
            last_str = f"Last click: ({w.last_click_orig.x()}, {w.last_click_orig.y()})"
        else:
            last_str = "Last click:"
        metric = ""
        opmode_text = self.currentOperationMode()
        opmode = _PURPOSE_MAP.get(opmode_text, opmode_text)
        if opmode in ["roi", "mask", "measure"]:
            if w.start_point_orig and w.end_point_orig:
                dx = w.end_point_orig.x() - w.start_point_orig.x()
                dy = w.end_point_orig.y() - w.start_point_orig.y()
                if w.drawing_mode == "rectangle":
                    metric = f"Rect width : {abs(dx)}, height : {abs(dy)}, area : {abs(dx) * abs(dy)}"
                elif w.drawing_mode == "line":
                    length = int(math.hypot(dx, dy))
                    metric = f"Line length : {length}"
                elif w.drawing_mode == "circle":
                    radius = int(math.hypot(dx, dy) / 2)
                    metric = f"Circle diameter : {2 * radius}"

        frame_info = ""
        if self.is_video:
            frame_info = f"Frame : {self.current_frame_idx+1} / {self.total_frames}"

        if opmode.startswith("thresholding"):
            info_text = f"{frame_info}\n{current_str}"
            all_sz = getattr(self, "all_contour_sizes", [])
            focal_sz = getattr(self, "contour_sizes", [])
            if all_sz:
                sz_range = f"{min(all_sz)}–{max(all_sz)}"
                info_text += f"\nAll blobs: {len(all_sz)}  (areas: {sz_range})"
            else:
                info_text += "\nAll blobs: 0"
            info_text += f"\nAccepted:  {len(focal_sz)}"
            if focal_sz:
                info_text += f"  (areas: {min(focal_sz)}–{max(focal_sz)})"
        else:
            info_text = f"{frame_info}\n{current_str}\n{last_str}\n{metric}"

        self.info_label.setText(info_text.strip())

        w.update()

    def updateNumIDs(self):
        num_ids = self.input_num_ids.value()
        self.tp_total_ids = num_ids
        self.current_id_box.setMaximum(num_ids)
        self.updateIDButtons()   # <-- Make sure this is called here
        self.drawing_widget.tp_total_ids = num_ids
        self.drawing_widget.update()

    def onIDButtonClicked(self, idx):
        if self.rb_scope_id.isChecked():
            self.current_id_box.setValue(idx + 1)
            self.updateIDButtons()
        
    def updateCurrentID(self, val):
        self.tp_current_id = val - 1
        self.drawing_widget.update()

    def updateVisibleRange(self, val):
        self.tp_visible_range = val
        self.range_value_box.blockSignals(True)
        self.range_value_box.setValue(val)
        self.range_value_box.blockSignals(False)
        self.drawing_widget.update()
    
    def toggleLineVisibility(self, checked):
        self.tp_show_lines = bool(checked)
        self.drawing_widget.update()
    
    def on_ptype_change(self, idx):
        self.current_ptype_idx = idx
        self.current_ptype = self.ptype_options[idx][1]
        # Optionally update display or clear selection per type
        self.drawing_widget.update()

    def updateLineWindow(self, val):
        self.tp_line_window = val
    
    # ---------- Video Methods ----------
    def onVideoSliderChanged(self, value):
        value = min(max(0, value), max(0, self.total_frames - 1))
        self.current_frame_idx = value
        if hasattr(self, "frame_spin"):
            self.frame_spin.setValue(value + 1)
        if hasattr(self, "video_slider"):
            self.video_slider.blockSignals(True)
            self.video_slider.setValue(value)
            self.video_slider.blockSignals(False)
        if self.cap:
            self.timer.stop()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, value)
            self.nextFrame()
        else:
            self.proxyUpdate()

    def onPlayClicked(self):
        if self.cap and self.timer:
            fps = self.fps_spin.value()
            self.timer.start(int(1000 / fps))
        self.drawing_widget.setFocus()

    def onStopClicked(self):
        if self.timer:
            self.timer.stop()
        self.drawing_widget.setFocus()

    def onStartClicked(self):
        self.timer.stop()
        self.current_frame_idx = 0
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.nextFrame()
        else:
            self.frame_spin.setValue(1)
            self.video_slider.setValue(0)
            self.proxyUpdate()
        self.drawing_widget.setFocus()

    def onEndClicked(self):
        self.timer.stop()
        self.current_frame_idx = self.total_frames - 1
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            self.nextFrame()
        else:
            self.frame_spin.setValue(self.current_frame_idx + 1)
            self.video_slider.setValue(self.current_frame_idx)
            self.proxyUpdate()
        self.drawing_widget.setFocus()

    def nextFrame(self):
        if self.is_video and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                self.timer.stop()
                return
            # === Apply ROI cropping here ===
            if self.roi is not None:
                (x0, y0), (x1, y1) = self.roi
                frame = frame[y0:y1, x0:x1]
            # ===============================

            self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            self.frame_spin.setValue(self.current_frame_idx + 1)
            self.video_slider.blockSignals(True)
            self.video_slider.setValue(self.current_frame_idx)
            self.video_slider.blockSignals(False)
            # Always update background image
            qimg = cvMatToQImage(frame)
            self.drawing_widget.setBackgroundImage(qimg)
            _opmode_now = _PURPOSE_MAP.get(self.currentOperationMode(), self.currentOperationMode())
            if _opmode_now in ["thresholding", "thresholding color"]:
                self.updateThresholdingImage(frame)
            self.proxyUpdate()
            self.proxyUpdate()
        elif self.is_video:
            # Blank/timeseries mode: just advance frame
            if self.current_frame_idx < self.total_frames - 1:
                self.current_frame_idx += 1
                self.proxyUpdate()


    def onFrameSpinEditingFinished(self):
        value = self.frame_spin.value() - 1
        self.current_frame_idx = value
        if self.cap:
            self.timer.stop()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, value)
            self.nextFrame()
        else:
            self.video_slider.setValue(self.current_frame_idx)
            self.proxyUpdate()
        self.drawing_widget.setFocus()

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
        # Only clear temp shape if switching to "point" mode
        if new_mode == "point":
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
    
    def onShowZonesChanged(self, state):
        self.drawing_widget.show_zones = (state == Qt.Checked)
        self.drawing_widget.update()

    def onHueChanged(self, value):
        self.drawing_widget.drawing_color = QColor.fromHsv(value, 255, 255)
        self.drawing_widget.update()

    def currentOperationMode(self):
        """Return the internal mode for either editor's displayed label."""
        label = self.opmode_combo.currentText().lower()
        return _PURPOSE_MAP.get(label, label)

    def onOpModeChanged(self, index):
        if self._file_infos and self._active_purpose is not None:
            self._remember_current()
        self._switching_purpose = True
        drawing_mode = self.mode_combo.currentText()
        opmode_text = self.opmode_combo.itemText(index).lower()
        opmode = _PURPOSE_MAP.get(opmode_text, opmode_text)
        self._undo_buffer = {}

        # Clear shapes and overlay if needed
        self.drawing_widget.clearCurrentShape()
        self.drawing_widget.shapes.clear()
        if opmode not in ["thresholding", "thresholding color"]:
            self.drawing_widget.drawing_overlay = None
        self.drawing_widget.drawing_allowed = True

        # Show everything first
        self.mode_group.setVisible(True)
        self.func_group.setVisible(True)
        self.flim_group.setVisible(False)
        self.timepoints_group.setVisible(False)
        self.thresh_group.setVisible(False)
        self.thresh_img_group.setVisible(False)
        self.cb_showthreshimg.setVisible(False)
        self.cb_showoverlay.setVisible(False)
        self.cb_showmask.setVisible(True)
        self.cb_invertmask.setVisible(True)
        self.cb_showzones.setVisible(False)
        self.hue_slider.setVisible(True)
        self.hue_label.setVisible(True)
        self.hue_panel_label.setVisible(True)

        for _, _, widget in self.sliders_bw_widgets:
            widget.setVisible(True)
        for _, _, widget in self.sliders_color_widgets:
            widget.setVisible(True)

        # === Mode-specific settings ===
        if opmode == "default":
            self.drawing_widget.setEnabled(True)
            self.mode_group.setVisible(True)
            self.func_group.setVisible(True)
            self.hue_slider.setVisible(True)
            self.mode_combo.clear()
            self.mode_combo.addItems(["point", "line", "rectangle", "polygon", "circle", "ellipse"])
            self.hue_label.setVisible(True)
            self.cb_showmask.setVisible(True)
            self.cb_invertmask.setVisible(True)

        elif opmode == "roi":
            self.func_group.setVisible(True)
            self.hue_slider.setVisible(True)
            self.drawing_widget.setEnabled(True)
            self.drawing_widget.drawing_mode = "rectangle"
            self.mode_combo.clear()
            self.mode_combo.addItem("rectangle")

        elif opmode == "measure":
            self.func_group.setVisible(True)
            self.hue_slider.setVisible(True)
            self.drawing_widget.setEnabled(True)
            self.drawing_widget.drawing_mode = "measure_polyline"
            # clear previous shape
            self.drawing_widget.measure_polyline_orig = []

        elif opmode == "points":
            self.func_group.setVisible(True)
            self.hue_slider.setVisible(True)
            self.drawing_widget.setEnabled(True)
            self.drawing_widget.drawing_mode = "points"
            self.mode_combo.setCurrentText("points")

        elif opmode == "timepoints":
            self.func_group.setVisible(True)
            self.hue_slider.setVisible(True)
            self.drawing_widget.setEnabled(True)
            self.drawing_widget.drawing_mode = "point"
            self.mode_group.setVisible(True)
            self.timepoints_group.setVisible(True)
            self.mode_combo.clear()
            self.mode_combo.addItems(["point", "rectangle", "polygon", "circle", "ellipse"])
            self.mode_combo.setCurrentText("point")

        elif opmode == "mask":
            self.func_group.setVisible(True)
            self.hue_slider.setVisible(True)
            self.drawing_widget.setEnabled(True)
            self.drawing_widget.drawing_mode = "rectangle"
            self.mode_group.setVisible(True)
            self.mode_combo.clear()
            self.mode_combo.addItems(["rectangle", "polygon", "circle", "ellipse"])
            self.mode_combo.setCurrentText("rectangle")
            self.cb_showmask.setVisible(True)
            self.cb_invertmask.setVisible(True)
            self.hue_label.setVisible(True)

        elif opmode == "zones":
            self.func_group.setVisible(True)
            self.hue_slider.setVisible(True)
            self.drawing_widget.setEnabled(True)
            self.drawing_widget.drawing_mode = "polygon"
            self.mode_group.setVisible(True)
            self.mode_combo.clear()
            self.mode_combo.addItems(["rectangle", "polygon", "circle", "ellipse"])
            self.mode_combo.setCurrentText("rectangle")
            self.hue_label.setVisible(True)
            self.cb_showzones.setVisible(True)
            self.cb_showzones.setChecked(True)
            
        elif opmode == "framelimits":
            self.func_group.setVisible(True)
            self.hue_slider.setVisible(True)
            self.drawing_widget.drawing_allowed = False
            self.flim_group.setVisible(True)

        elif opmode == "thresholding":
            self.func_group.setVisible(True)
            self.hue_slider.setVisible(True)
            self.drawing_widget.drawing_allowed = False
            self.thresh_group.setVisible(True)
            self.thresh_img_group.setVisible(True)
            self.cb_showthreshimg.setVisible(True)
            self.cb_showoverlay.setVisible(True)
            self.hue_panel_label.setVisible(False)
            # Show all B/W sliders
            for _, _, widget in self.sliders_bw_widgets:
                widget.setVisible(True)
            # Hide all HSV sliders
            for _, _, widget in self.sliders_color_widgets:
                widget.setVisible(False)
            self.updateThresholdingImage()

        elif opmode == "thresholding color":
            self.func_group.setVisible(True)
            self.hue_slider.setVisible(True)
            self.drawing_widget.drawing_allowed = False
            self.thresh_group.setVisible(True)
            self.thresh_img_group.setVisible(True)
            self.cb_showthreshimg.setVisible(True)
            self.cb_showoverlay.setVisible(True)
            self.hue_panel_label.setVisible(True)
            # Show only the general BW sliders (blur, min_area, max_area)
            for label_text, slider, widget in self.sliders_bw_widgets:
                if label_text in ("Blur:", "Min Area:", "Max Area:"):
                    widget.setVisible(True)
                else:
                    widget.setVisible(False)
            # Show all HSV sliders
            for _, _, widget in self.sliders_color_widgets:
                widget.setVisible(True)
            self.updateHuePanel()
            self.updateThresholdingImage()


        if opmode == "measure":
            if hasattr(self, "mode_combo"):
                self.mode_combo.setEnabled(False)
        else:
            if hasattr(self, "mode_combo"):
                self.mode_combo.setEnabled(True)

        # Rebuilding the available tools must not reset the user's selection.
        if opmode in ("default", "roi", "mask", "zones") and self.mode_combo.findText(drawing_mode) >= 0:
            self.mode_combo.setCurrentText(drawing_mode)

        if self._file_infos:
            purpose = self.opmode_combo.itemText(index)
            self._active_purpose = purpose
            fi = self._file_infos[self._file_idx]
            # Setup drawings use full-frame coordinates; tracked points use ROI coordinates.
            self.roi = fi.get("roi") if opmode in ("timepoints", "thresholding", "thresholding color") else None
            if opmode != "mask":
                mask = QImage(fi.get("mask_path") or "")
                if mask.isNull():
                    mask = QImage(self.orig_width, self.orig_height, QImage.Format_Grayscale8)
                    mask.fill(255)
                if self.roi:
                    (x0, y0), (x1, y1) = self.roi
                    mask = mask.copy(x0, y0, x1 - x0, y1 - y0)
                self.drawing_widget.mask_image = mask
            if self.cap and hasattr(self, "video_slider"):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
                self.nextFrame()
            state = self._file_states.get(self._file_idx, {}).get(purpose)
            if state is not None:
                self._restore_state(state)
            else:
                self._auto_load_purpose_data(fi, purpose)
        self._switching_purpose = False
        if opmode in ("thresholding", "thresholding color"):
            self.updateThresholdingImage()
        if self._file_infos and state is None:
            self._saved_states[(self._file_idx, purpose)] = self._collect_state(purpose)
        # Trigger redraw
        self.drawing_widget.update()
        self.proxyUpdate()

    def updateThresholdingImage(self, frame=None):
        if getattr(self, "_restoring_threshold", False) or getattr(self, "_switching_purpose", False):
            return
        self.thresh_params = {
            "blur": self.sl_blur.value(),
            "erode": self.sl_erode.value(),
            "blur2": self.sl_blur2.value(),
            "threshold": self.sl_thresh.value(),
            "min_area": self.sl_minarea.value(),
            "max_area": self.sl_maxarea.value(),
            "hue_lo": self.sl_hue_lo.value(),
            "hue_hi": self.sl_hue_hi.value(),
            "sat_lo": self.sl_sat_lo.value(),
            "sat_hi": self.sl_sat_hi.value(),
            "val_lo": self.sl_val_lo.value(),
            "val_hi": self.sl_val_hi.value(),
        }

        mask_np = None
        if self.drawing_widget.mask_image and not self.drawing_widget.mask_image.isNull():
            mask_np = qimage_to_numpy(self.drawing_widget.mask_image)
            if mask_np.ndim == 3:
                mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)

        opmode = _PURPOSE_MAP.get(self.currentOperationMode(), self.currentOperationMode())

        # Get a fresh frame if none was passed in
        if frame is None:
            if self.is_video and self.cap:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("Could not grab frame.")
                    return
                if self.roi is not None:
                    (x0, y0), (x1, y1) = self.roi
                    frame = frame[y0:y1, x0:x1]
            else:
                print("No frame and not a video.")
                return

        # Ensure image has 3 channels
        if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] != 3):
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        bg = None
        if self.background_file:
            bg_qimg = QImage(self.background_file)
            if not bg_qimg.isNull():
                bg = qimage_to_numpy(bg_qimg)
                if self.roi is not None:
                    (x0, y0), (x1, y1) = self.roi
                    bg = bg[y0:y1, x0:x1]
                if len(bg.shape) == 2 or (len(bg.shape) == 3 and bg.shape[2] != 3):
                    bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)

        # Parse parameters depending on mode
        params = self.thresh_params

        if opmode == "thresholding":
            PI = ProcessImage(
                frame, bg, mask_np, "bw",
                params["blur"], params["erode"], params["blur2"],
                params["threshold"], params["min_area"], params["max_area"],
                simple=True
            )

        elif opmode == "thresholding color":
            colmin = (params["hue_lo"], params["sat_lo"], params["val_lo"])
            colmax = (params["hue_hi"], params["sat_hi"], params["val_hi"])
            PI = ProcessImage(
                frame, bg, mask_np, "color",
                blur=params["blur"], min_area=params["min_area"], max_area=params["max_area"],
                colmin=colmin, colmax=colmax, simple=True
            )
        else:
            return  # Not a thresholding mode

        # Process the image
        self.img_thresh, allcons, conlist = PI.process()

        # Display thresholded image
        qimg = cvMatToQImage(self.img_thresh)
        self.thresh_img_label.setPixmap(QPixmap.fromImage(qimg).scaled(300, 300, Qt.KeepAspectRatio))

        # ===== DRAW OVERLAY =====
        bg = self.drawing_widget.background_image
        if bg is None or bg.isNull():
            return

        w, h = bg.width(), bg.height()
        overlay = np.zeros((h, w, 4), dtype=np.uint8)

        # Colors
        red = (0, 0, 255, 255)
        blue = (255, 0, 0, 255)
        white = (255, 255, 255, 255)
        black = (0, 0, 0, 255)

        show_overlay = self.cb_showoverlay.isChecked()

        # Convert contourlist to list if needed
        if isinstance(conlist, dict) and "contour" in conlist:
            focal_cons = [c for c in conlist["contour"] if c is not None]
        else:
            focal_cons = []

        # Compute areas for all detected contours (for info display)
        all_areas = [int(cv2.contourArea(c)) for c in (allcons or [])]
        focal_areas = [int(cv2.contourArea(c)) for c in focal_cons]

        if show_overlay:
            # Blue outlines: all detected blobs (including those outside area range)
            if allcons:
                cv2.drawContours(overlay, allcons, -1, blue, 1)
            # Red outlines + area labels: blobs within the current area filter
            if focal_cons:
                cv2.drawContours(overlay, focal_cons, -1, red, 2)
                for i, c in enumerate(focal_cons):
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.circle(overlay, (cX, cY), 4, red, -1)
                        text = str(focal_areas[i])
                        cv2.putText(overlay, text, (cX + 6, cY - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, black, 2, cv2.LINE_AA)
                        cv2.putText(overlay, text, (cX + 6, cY - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1, cv2.LINE_AA)
            # Grey area labels on excluded blobs so user can see what they're filtering out
            grey = (180, 180, 180, 200)
            min_a = self.thresh_params.get("min_area", 0)
            max_a = self.thresh_params.get("max_area", 999999)
            for c in (allcons or []):
                area = int(cv2.contourArea(c))
                if area < min_a or area > max_a:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        text = str(area)
                        cv2.putText(overlay, text, (cX + 4, cY - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, grey, 1, cv2.LINE_AA)

        overlay_qimg = QImage(overlay.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
        self.drawing_widget.drawing_overlay = overlay_qimg

        # Save size data for info panel
        self.contour_sizes = focal_areas
        self.all_contour_sizes = all_areas

        self.drawing_widget.update()
        self.proxyUpdate()

    def updateHuePanel(self):
        """Draws the hue gradient and highlights selected hue range."""
        hue_lo = self.sl_hue_lo.value()
        hue_hi = self.sl_hue_hi.value()

        # Generate hue panel: horizontal hue gradient from 0 to 179
        panel = np.zeros((30, 180, 3), dtype=np.uint8)
        for i in range(180):
            panel[:, i] = [i, 255, 255]
        panel = cv2.cvtColor(panel, cv2.COLOR_HSV2RGB)

        # Darken areas outside hue range
        panel[:, :hue_lo] //= 3
        panel[:, hue_hi:] //= 3

        # Draw vertical lines at hue_lo and hue_hi
        cv2.line(panel, (hue_lo, 0), (hue_lo, panel.shape[0] - 1), (0, 0, 0), 1)
        cv2.line(panel, (hue_hi, 0), (hue_hi, panel.shape[0] - 1), (0, 0, 0), 1)

        # Convert to QImage and set it
        h, w, ch = panel.shape
        qimg = QImage(panel.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.hue_panel_label.setPixmap(pixmap)

    def keyPressEvent(self, event):
        # Allow pressing Enter to finalize the polygon (but NOT measure polyline!)
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.drawing_widget.drawing_mode == "polygon" and len(self.polygon_points_orig) >= 2:
                if self.polygon_points_orig[0] != self.polygon_points_orig[-1]:
                    self.polygon_points_orig.append(self.polygon_points_orig[0])
                self.temp_point_orig = None
                self.update()
            return
        
        key = event.key()
        
        if key == Qt.Key_Escape:
            now = time.time()
            if now - self._last_esc_press_time < 1.2:
                self.drawing_widget.final_output = "exit"
                self.was_fullscreen = self.isFullScreen()
                self._last_esc_press_time = 0
                self.close()
            else:
                self.drawing_widget.final_output = None
                self._last_esc_press_time = now
        elif key == Qt.Key_Q:
            # --- Q: Go to start ---
            if self.is_video:
                self.current_frame_idx = 0
                if self.cap:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.nextFrame()
                else:
                    self.frame_spin.setValue(1)
                    self.video_slider.setValue(0)
                    self.proxyUpdate()
        elif key == Qt.Key_W:
            # --- W: Jump back by FPS ---
            if self.is_video:
                jump = self.fps_spin.value()
                new_idx = max(0, self.current_frame_idx - jump)
                self.current_frame_idx = new_idx
                if self.cap:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_idx)
                    self.nextFrame()
                else:
                    self.frame_spin.setValue(self.current_frame_idx + 1)
                    self.video_slider.setValue(self.current_frame_idx)
                    self.proxyUpdate()
        elif key == Qt.Key_E:
            # --- E: Go back 1 frame ---
            if self.is_video:
                new_idx = max(0, self.current_frame_idx - 1)
                self.current_frame_idx = new_idx
                if self.cap:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_idx)
                    self.nextFrame()
                else:
                    self.frame_spin.setValue(self.current_frame_idx + 1)
                    self.video_slider.setValue(self.current_frame_idx)
                    self.proxyUpdate()
        elif key == Qt.Key_R:
            # --- R: Go forward 1 frame ---
            if self.is_video:
                new_idx = min(self.total_frames - 1, self.current_frame_idx + 1)
                self.current_frame_idx = new_idx
                if self.cap:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_idx)
                    self.nextFrame()
                else:
                    self.frame_spin.setValue(self.current_frame_idx + 1)
                    self.video_slider.setValue(self.current_frame_idx)
                    self.proxyUpdate()
        elif key == Qt.Key_T:
            # --- T: Jump forward by FPS ---
            if self.is_video:
                jump = self.fps_spin.value()
                new_idx = min(self.total_frames - 1, self.current_frame_idx + jump)
                self.current_frame_idx = new_idx
                if self.cap:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_idx)
                    self.nextFrame()
                else:
                    self.frame_spin.setValue(self.current_frame_idx + 1)
                    self.video_slider.setValue(self.current_frame_idx)
                    self.proxyUpdate()
        elif key == Qt.Key_Y:
            # --- Y: Go to end ---
            if self.is_video:
                self.current_frame_idx = self.total_frames - 1
                if self.cap:
                    self.onEndClicked()
                else:
                    self.frame_spin.setValue(self.current_frame_idx + 1)
                    self.video_slider.setValue(self.current_frame_idx)
                    self.proxyUpdate() 
        elif key == Qt.Key_A:
            self.drawing_widget.addCurrentShapeIfNeeded()
            self.drawing_widget.update()
        elif key == Qt.Key_S:
            # In multi-file mode, S stores the current drawing without closing
            if self._file_infos:
                self._store_current()
                return

            self.drawing_widget.addCurrentShapeIfNeeded()
            result = None
            opmode_text = self.currentOperationMode()
            opmode = _PURPOSE_MAP.get(opmode_text, opmode_text)

             # If user recorded event frames with the event key, return them immediately
            if getattr(self, "_event_frames", None):
                frames = sorted(set(int(f) for f in self._event_frames))
                if frames:
                    # finish the printed line
                    print("", flush=True)
                    result = ("events", frames)
                    self._event_frames = []
                    self.drawing_widget.final_output = result
                    self.close()
                    return
                else:
                    # empty buffer -> clear any printed line and fall through
                    print("", flush=True)
                    self._event_frames = []

            if opmode == "roi":
                shape = self.drawing_widget.shapes[-1][1] if self.drawing_widget.shapes else None
                shape = tuple(shape[i] for i in (0,2))
                result = ("roi", shape)
            elif opmode == "mask":
                img = self.drawing_widget.mask_image
                arr = qimage_to_numpy(img)
                if self.drawing_widget.inverted:
                    arr = 255 - arr
                result = ("mask", arr)
            elif opmode == "zones":
                self.drawing_widget.commitZones()
                if self.drawing_widget.zones_overlay:
                    # Composite on white background
                    zones_img = self.drawing_widget.zones_overlay
                    white_bg = QImage(zones_img.size(), QImage.Format_ARGB32)
                    white_bg.fill(Qt.white)
                    painter = QPainter(white_bg)
                    painter.drawImage(0, 0, zones_img)
                    painter.end()
                    img = qimage_to_numpy(white_bg)
                    result = ("zones", img)
            elif opmode == "measure":
                pts = self.drawing_widget.measure_polyline_orig
                if len(pts) >= 2:
                    shape = [(pt.x(), pt.y()) for pt in pts]
                    self.drawing_widget.shapes.append(("measure_polyline", shape))
                    
                    # Calculate length
                    total = 0.0
                    for i in range(1, len(shape)):
                        dx = shape[i][0] - shape[i-1][0]
                        dy = shape[i][1] - shape[i-1][1]
                        total += math.hypot(dx, dy)
                    
                    # Calculate area if there are 4 or more points
                    area = None
                    if len(shape) >= 4:
                        area = 0.0
                        for i in range(len(shape)):
                            x1, y1 = shape[i]
                            x2, y2 = shape[(i + 1) % len(shape)]
                            area += x1 * y2 - y1 * x2
                        area = abs(area) / 2.0
                    
                    # Prepare the result
                    result = ("polygon", shape, int(total), int(area) if area else None)
                    
                    # Clear the polyline and update the canvas
                    self.drawing_widget.measure_polyline_orig = []
                    self.drawing_widget.update()
                else:
                    result = None  # Not enough points
            elif opmode == "points":
                if self.drawing_widget.points_orig:
                    result = ("points", [(pt.x(), pt.y()) for pt in self.drawing_widget.points_orig])
            elif opmode == "timepoints":
                data = []
                # For each ID (individual)
                for id_, typedict in self.drawing_widget.points_by_frame.items():
                    # Collect all unique frames for this ID
                    frames = set()
                    for typ in ["c", "h", "t", "a"]:
                        frames.update(typedict.get(typ, {}).keys())
                    for f in sorted(frames):
                        row = {"frame": f, "id": id_}
                        # Centroid
                        c = typedict.get("c", {}).get(f)
                        if c is not None and hasattr(c, "x"):
                            row["cx"] = c.x()
                            row["cy"] = c.y()
                        else:
                            row["cx"], row["cy"] = None, None
                        # Head
                        h = typedict.get("h", {}).get(f)
                        if h is not None and hasattr(h, "x"):
                            row["hx"] = h.x()
                            row["hy"] = h.y()
                        else:
                            row["hx"], row["hy"] = None, None
                        # Tail
                        t = typedict.get("t", {}).get(f)
                        if t is not None and hasattr(t, "x"):
                            row["tx"] = t.x()
                            row["ty"] = t.y()
                        else:
                            row["tx"], row["ty"] = None, None
                        # Angle
                        a = typedict.get("a", {}).get(f)
                        row["angle"] = a if a is not None else None
                        data.append(row)
                df_out = pd.DataFrame(data).sort_values(["id", "frame"]).reset_index(drop=True)
                result = ("timepoints", df_out)
            elif opmode == "framelimits":
                result = ("framelimits", (self.flim_start_spin.value(), self.flim_stop_spin.value()))
            elif opmode in ["thresholding", "thresholding color"]:
                result = ("thresholding", self.thresh_params)
            self.drawing_widget.final_output = result
            self.close()
        elif key == Qt.Key_D:
            draw_mode = self.drawing_widget.drawing_mode
            opmode = self.currentOperationMode()
            if draw_mode == "polygon" or opmode == "measure":  # Treat "measure" as "polygon"
                if opmode == "measure" and self.drawing_widget.measure_polyline_orig:
                    self.drawing_widget.measure_polyline_orig.pop()
                    self.drawing_widget.update()  # Refresh the canvas
                    print("Deleted the last point in measure mode.")
                elif self.drawing_widget.polygon_points_orig:
                    self.drawing_widget.polygon_points_orig.pop()
                    self.drawing_widget.update()  # Refresh the canvas
                    print("Deleted the last point in polygon mode.")
            elif draw_mode == "points":
                if self.drawing_widget.points_orig:
                    self.drawing_widget.points_orig.pop()
                    self.drawing_widget.update()  # Refresh the canvas
                    print("Deleted the last point in points mode.")
            elif draw_mode == "point" and self.is_video:
                cur_id = self.tp_current_id
                cur_frame = self.current_frame_idx
                if cur_id in self.drawing_widget.points_by_frame:
                    if cur_frame in self.drawing_widget.points_by_frame[cur_id]:
                        del self.drawing_widget.points_by_frame[cur_id][cur_frame]
                        self.drawing_widget.update()  # Refresh the canvas
                        print(f"Deleted point for ID {cur_id} at frame {cur_frame}")
            elif opmode == "zones":
                self.drawing_widget.zones_overlay.fill(QColor(255, 255, 255))
            elif opmode == "mask":
                self.drawing_widget.mask_image.fill(QColor(255, 255, 255))
            self.drawing_widget.update()
            self.proxyUpdate()
        # add event-frame record key (K)
        elif key == Qt.Key_K:
            frame = self.current_frame_idx + 1 if self.is_video else 1
            if not hasattr(self, "_event_frames"):
                self._event_frames = []
            if frame not in self._event_frames:
                self._event_frames.append(frame)
                # # print header once, then append the frame on the same line
                # if len(self._event_frames) == 1:
                #     print("Event frames...", end=" ")
                print(f"Added {frame}", end=" ", flush=True)
            # duplicate presses silently ignored
            self.drawing_widget.setFocus()
        # remove last recorded event frame (L)
        elif key == Qt.Key_L:
            if getattr(self, "_event_frames", None):
                removed = self._event_frames.pop()
                print("Removed", removed, end=" ", flush=True)
            self.drawing_widget.setFocus()
        elif key == Qt.Key_F:
            if self.isFullScreen():
                self.was_fullscreen = False
                self.showNormal()
                if hasattr(self, 'default_geometry'):
                    self.setGeometry(self.default_geometry)
            else:
                self.was_fullscreen = True
                self.default_geometry = self.geometry()
                self.showFullScreen()
        elif key == Qt.Key_H:
            # Toggle helperlines checkbox & widget state
            checked = not self.cb_helperlines.isChecked()
            self.cb_helperlines.setChecked(checked)
            self.drawing_widget.setHelperlinesEnabled(checked)    
        elif key == Qt.Key_U:
            # Cycle draw/move/angle radio buttons
            if self.rb_draw.isChecked():
                self.rb_move.setChecked(True)
                print("Mode: Move points")
            elif self.rb_move.isChecked():
                self.rb_angle.setChecked(True)
                print("Mode: Change angle")
            else:
                self.rb_draw.setChecked(True)
                print("Mode: Draw points")
        elif key == Qt.Key_I:
            if self.currentOperationMode() == "timepoints" and self.tp_total_ids > 1:
                new_val = (self.current_id_box.value() - 2) % self.tp_total_ids + 1  # subtract 2 because value() is 1-based
                self.current_id_box.setValue(new_val)
                print(f"Switched to ID: {new_val}")

        elif key == Qt.Key_O:
            if self.currentOperationMode() == "timepoints" and self.tp_total_ids > 1:
                new_val = (self.current_id_box.value()) % self.tp_total_ids + 1  # value() is 1-based
                self.current_id_box.setValue(new_val)
                print(f"Switched to ID: {new_val}")
        elif key == Qt.Key_P:
            # Cycle point types (centroid/head/tail)
            self.current_ptype_idx = (self.current_ptype_idx + 1) % len(self.ptype_options)
            self.current_ptype = self.ptype_options[self.current_ptype_idx][1]
            self.ptype_dropdown.setCurrentIndex(self.current_ptype_idx)
            print(f"Switched point type to: {self.ptype_options[self.current_ptype_idx][0]}")
            self.drawing_widget.update()         
        elif key == Qt.Key_Z:
            if self.currentOperationMode() == "zones":
                self.drawing_widget.commitZones()
        elif key == Qt.Key_Space:
            if self.is_video:
                if self.timer.isActive():
                    self.onStopClicked()
                else:
                    self.onPlayClicked()
        else:
            event.ignore()

    def showEvent(self, event):
        super().showEvent(event)
        if not hasattr(self, 'default_geometry'):
            self.default_geometry = self.geometry()

    def _apply_prefs(self, prefs):
        geometry = prefs.get("window_geometry")
        if isinstance(geometry, str):
            try:
                self.restoreGeometry(QByteArray.fromHex(geometry.encode("ascii")))
            except (ValueError, UnicodeError):
                pass
        idx = self.mode_combo.findText(prefs.get("drawing_mode", "rectangle"))
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)
        self.hue_slider.setValue(prefs.get("hue", 120))
        self.bgtrans_slider.setValue(prefs.get("image_transparency", 100))
        self.gamma_slider.setValue(prefs.get("display_gamma", 100))
        self.point_op_slider.setValue(prefs.get("point_opacity", 100))
        self.mask_op_slider.setValue(prefs.get("mask_opacity", 80))
        self.point_size_slider.setValue(prefs.get("point_size", 6))
        self.cb_blackwhite.setChecked(prefs.get("blackwhite", False))
        self.cb_showmask.setChecked(prefs.get("show_mask", False))
        self.cb_invertmask.setChecked(prefs.get("invert_mask", False))
        self.cb_showzones.setChecked(prefs.get("show_zones", False))
        self.cb_helperlines.setChecked(prefs.get("helperlines", False))
        self.cb_crosshair.setChecked(prefs.get("crosshair", False))
        self.cb_showloop.setChecked(prefs.get("show_loop", False))
        self.show_points_checkbox.setChecked(prefs.get("show_points", True))
        self.show_lines_checkbox.setChecked(prefs.get("show_lines", False))
        self.show_framenrs_checkbox.setChecked(prefs.get("show_framenrs", False))
        self.show_nearframe_checkbox.setChecked(prefs.get("show_nearframe", False))
        self.angle_show_arrows_checkbox.setChecked(prefs.get("show_arrows", False))
        self.angle_show_lines_checkbox.setChecked(prefs.get("show_angle_lines", False))
        self.arrow_length_slider.setValue(prefs.get("arrow_length", 20))
        self.range_slider.setValue(prefs.get("visible_range", 100))
        self.highlight_current_checkbox.setChecked(prefs.get("highlight_current", True))
        range_dir = prefs.get("range_direction", "all")
        self.rb_range_past.setChecked(range_dir == "past")
        self.rb_range_future.setChecked(range_dir == "future")
        self.rb_range_all.setChecked(range_dir not in ("past", "future"))
        self.fade_points_checkbox.setChecked(prefs.get("fade_points", False))
        self.ptype_dropdown.setCurrentIndex(prefs.get("point_type_idx", 0))
        scope = prefs.get("scope_mode", "any")
        self.rb_scope_id.setChecked(scope == "id")
        self.rb_scope_any.setChecked(scope != "id")
        edit_mode = prefs.get("edit_mode", "draw")
        self.rb_move.setChecked(edit_mode == "move")
        self.rb_angle.setChecked(edit_mode == "angle")
        self.rb_draw.setChecked(edit_mode == "draw")
        if hasattr(self, "fps_spin"):
            self.fps_spin.setValue(prefs.get("fps", 25))

    def _collect_prefs(self):
        prefs = {
            "window_geometry": bytes(self.saveGeometry().toHex()).decode("ascii"),
            "drawing_mode": self.mode_combo.currentText(),
            "hue": self.hue_slider.value(),
            "image_transparency": self.bgtrans_slider.value(),
            "display_gamma": self.gamma_slider.value(),
            "point_opacity": self.point_op_slider.value(),
            "mask_opacity": self.mask_op_slider.value(),
            "point_size": self.point_size_slider.value(),
            "blackwhite": self.cb_blackwhite.isChecked(),
            "show_mask": self.cb_showmask.isChecked(),
            "invert_mask": self.cb_invertmask.isChecked(),
            "show_zones": self.cb_showzones.isChecked(),
            "helperlines": self.cb_helperlines.isChecked(),
            "crosshair": self.cb_crosshair.isChecked(),
            "show_loop": self.cb_showloop.isChecked(),
            "show_points": self.show_points_checkbox.isChecked(),
            "show_lines": self.show_lines_checkbox.isChecked(),
            "show_framenrs": self.show_framenrs_checkbox.isChecked(),
            "show_nearframe": self.show_nearframe_checkbox.isChecked(),
            "show_arrows": self.angle_show_arrows_checkbox.isChecked(),
            "show_angle_lines": self.angle_show_lines_checkbox.isChecked(),
            "arrow_length": self.arrow_length_slider.value(),
            "visible_range": self.range_slider.value(),
            "highlight_current": self.highlight_current_checkbox.isChecked(),
            "range_direction": ("past" if self.rb_range_past.isChecked()
                                else "future" if self.rb_range_future.isChecked() else "all"),
            "fade_points": self.fade_points_checkbox.isChecked(),
            "point_type_idx": self.ptype_dropdown.currentIndex(),
            "scope_mode": "id" if self.rb_scope_id.isChecked() else "any",
            "edit_mode": ("angle" if self.rb_angle.isChecked()
                          else "move" if self.rb_move.isChecked() else "draw"),
        }
        if hasattr(self, "fps_spin"):
            prefs["fps"] = self.fps_spin.value()
        return prefs

    def closeEvent(self, event):
        if self._file_infos:
            self._remember_current()
        if self._file_infos and self._unsaved:
            reply = QMessageBox.question(
                self, "Unsaved changes",
                "Some video/purpose edits have not been saved. Discard them and close?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
        self.was_fullscreen = self.isFullScreen()
        save_prefs(self._collect_prefs())
        super().closeEvent(event)

    # ==================== Multi-file editor methods ====================

    def _update_nav_label(self):
        if not self._file_infos:
            return
        fi = self._file_infos[self._file_idx]
        name = fi.get("output_basename", fi.get("video_name", f"File {self._file_idx + 1}"))
        total = len(self._file_infos)
        self.nav_name_label.setText(f"{name}\n({self._file_idx + 1} / {total})")
        self.nav_name_label.setToolTip(name)
        self.setWindowTitle(f"ATracker - {name}")
        self.btn_prev.setEnabled(self._file_idx > 0)
        self.btn_next.setEnabled(self._file_idx < total - 1)

    def _remember_current(self):
        if self._active_purpose is None:
            return
        purpose = self._active_purpose
        state = self._collect_state(purpose)
        self._file_states.setdefault(self._file_idx, {})[purpose] = state
        key = (self._file_idx, purpose)
        if state != self._saved_states.get(key):
            self._unsaved.add(key)
        else:
            self._unsaved.discard(key)

    def _collect_state(self, purpose=None):
        """Collect canvas + widget state for the current purpose into a dict."""
        opmode_text = (purpose or self.currentOperationMode()).lower()
        opmode = _PURPOSE_MAP.get(opmode_text, opmode_text)
        state = {"opmode": opmode}
        w = self.drawing_widget
        if opmode in ("roi", "mask", "zones"):
            state["shapes"] = list(w.shapes)
            pending_mode = w.drawing_mode if (w.start_point_orig is not None or
                w.end_point_orig is not None or w.polygon_points_orig) else None
            state["pending"] = (pending_mode,
                QPoint(w.start_point_orig) if w.start_point_orig is not None else None,
                QPoint(w.end_point_orig) if w.end_point_orig is not None else None,
                [QPoint(p) for p in w.polygon_points_orig])
        if opmode == "mask":
            if self.drawing_widget.mask_image and not self.drawing_widget.mask_image.isNull():
                state["mask_qimg"] = self.drawing_widget.mask_image.copy()
        elif opmode == "roi":
            state["shapes"] = list(self.drawing_widget.shapes)
        elif opmode == "zones":
            if self.drawing_widget.zones_overlay:
                state["zones_overlay"] = self.drawing_widget.zones_overlay.copy()
        elif opmode == "framelimits":
            state["frame_start"] = self.flim_start_spin.value()
            state["frame_stop"] = self.flim_stop_spin.value()
        elif opmode == "timepoints":
            state["points_by_frame"] = {i: {t: {f: QPoint(p) if isinstance(p, QPoint) else p
                for f, p in frames.items()} for t, frames in types.items()}
                for i, types in w.points_by_frame.items()}
            state["num_ids"] = self.input_num_ids.value()
        elif opmode == "measure":
            state["measure_polyline"] = list(getattr(self.drawing_widget, "measure_polyline_orig", []))
        elif opmode in ("thresholding", "thresholding color"):
            state["thresh_params"] = dict(self.thresh_params)
            state["threshold_type"] = self._file_infos[self._file_idx].get("threshold_type")
        return state

    def _restore_state(self, state):
        """Restore canvas + widget state from a collected dict."""
        if not state:
            return
        if "mask_qimg" in state:
            self.drawing_widget.mask_image = state["mask_qimg"].copy()
        if "shapes" in state:
            self.drawing_widget.shapes = list(state["shapes"])
        if "pending" in state:
            mode, start, end, polygon = state["pending"]
            if mode is not None:
                self.mode_combo.setCurrentText(mode)
            self.drawing_widget.start_point_orig = QPoint(start) if start is not None else None
            self.drawing_widget.end_point_orig = QPoint(end) if end is not None else None
            self.drawing_widget.polygon_points_orig = [QPoint(p) for p in polygon]
        if "zones_overlay" in state:
            self.drawing_widget.zones_overlay = state["zones_overlay"].copy()
        if "frame_start" in state:
            self.flim_start_spin.setValue(state["frame_start"])
        if "frame_stop" in state:
            self.flim_stop_spin.setValue(state["frame_stop"])
        if "points_by_frame" in state:
            self.drawing_widget.points_by_frame = {i: {t: {f: QPoint(p) if isinstance(p, QPoint) else p
                for f, p in frames.items()} for t, frames in types.items()}
                for i, types in state["points_by_frame"].items()}
            if hasattr(self, "timeline"):
                self.timeline.invalidate()
            self.input_num_ids.setValue(state.get("num_ids", 1))
        if "measure_polyline" in state:
            self.drawing_widget.measure_polyline_orig = list(state["measure_polyline"])
        if "thresh_params" in state:
            if "threshold_type" in state:
                self._file_infos[self._file_idx]["threshold_type"] = state["threshold_type"]
            self.thresh_params = dict(state["thresh_params"])
            self._restoring_threshold = True
            for key, value in self.thresh_params.items():
                attr = {"threshold": "thresh", "min_area": "minarea", "max_area": "maxarea"}.get(key, key)
                slider = getattr(self, "sl_" + attr, None)
                if slider is not None:
                    slider.setValue(int(value))
            self._restoring_threshold = False
            self.updateThresholdingImage()
        self.drawing_widget.update()

    def _auto_load_purpose_data(self, fi, purpose_text):
        """Auto-load existing data from file_info for the given purpose."""
        opmode = _PURPOSE_MAP.get(purpose_text.lower(), purpose_text.lower())
        if opmode == "mask":
            self.drawing_widget.mask_image = QImage(self.orig_width, self.orig_height, QImage.Format_Grayscale8)
            self.drawing_widget.mask_image.fill(255)
        elif opmode == "zones":
            self.drawing_widget.zones_overlay = None
        elif opmode == "timepoints":
            self.drawing_widget.points_by_frame = {}
            self.input_num_ids.setValue(1)
            fi["id_labels"] = {}
        if opmode == "mask":
            mask_path = fi.get("mask_path")
            if mask_path and os.path.isfile(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    from ._utils import numpy_to_qimage
                    self.drawing_widget.mask_image = numpy_to_qimage(mask, force_grayscale=True)
        elif opmode == "zones":
            zones_path = fi.get("zones_path")
            if zones_path and os.path.isfile(zones_path):
                zones = cv2.imread(zones_path, cv2.IMREAD_UNCHANGED)
                if zones is not None:
                    from ._utils import numpy_to_qimage
                    self.drawing_widget.zones_overlay = numpy_to_qimage(zones).convertToFormat(QImage.Format_ARGB32)
        elif opmode == "roi":
            roi = fi.get("roi")
            if roi:
                (x0, y0), (x1, y1) = roi
                self.drawing_widget.shapes = [("rectangle", ((x0, y0), (x1, y0), (x1, y1), (x0, y1)))]
        elif opmode == "framelimits":
            start = fi.get("frame_start")
            stop = fi.get("frame_stop")
            if start:
                self.flim_start_spin.setValue(int(start))
            if stop:
                self.flim_stop_spin.setValue(int(stop))
        elif opmode == "timepoints":
            csv_path = fi.get("tracked_csv")
            if csv_path and os.path.isfile(csv_path):
                from ._utils import build_points_by_frame
                df = load_and_convert_tracking_dataframe(csv_path)
                if df is not None and len(df) > 0:
                    if "IDstr" in df.columns:
                        unique_ids = sorted([s for s in df["IDstr"].unique() if pd.notna(s)], key=str)
                        id_map = {s: i for i, s in enumerate(unique_ids)}
                        fi["id_labels"] = {i: s for s, i in id_map.items()}
                        df["ID"] = df["IDstr"].map(id_map).fillna(-1).astype(int)
                    pbf = build_points_by_frame(df)
                    self.drawing_widget.points_by_frame = pbf
                    if pbf:
                        self.input_num_ids.setValue(len(pbf))
                        self.current_id_box.setValue(1)
                        self.updateCurrentID(1)
                        self.drawing_widget.tp_current_id = 0
        elif opmode in ("thresholding", "thresholding color"):
            options = fi.get("threshold_options", {})
            selected = fi.get("threshold_type")
            if len(options) > 1:
                selected, ok = QInputDialog.getItem(self, "Threshold configuration",
                    "Configuration to edit:", list(options), 0, False)
                if not ok:
                    selected = None
            elif options:
                selected = next(iter(options))
            fi["threshold_type"] = selected
            params = options.get(selected, fi.get("threshold_dict", {}))
            if params:
                self._restore_state({"thresh_params": params})
        self.drawing_widget.update()

    def _navigate_to(self, new_idx):
        """Navigate to a different file, saving current state."""
        if not self._file_infos:
            return
        if new_idx < 0 or new_idx >= len(self._file_infos):
            return
        if new_idx == self._file_idx:
            return

        # Navigation retains each purpose's pending work without writing it.
        self._remember_current()
        self._active_purpose = None

        # Release old media
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.timer:
            self.timer.stop()

        self._file_idx = new_idx
        fi = self._file_infos[new_idx]

        # Load new media
        new_file = fi.get("video_path")
        if new_file and os.path.isfile(new_file):
            path = os.path.expanduser(new_file)
            media_type = get_media_type(path)
            if media_type == "vid":
                self.is_video = True
                self.cap = cv2.VideoCapture(path)
                self.fps, self.orig_width, self.orig_height, self.total_frames = get_vid_params(self.cap)
                self.is_video = self.total_frames > 1
            elif media_type == "img":
                bg = QImage(path)
                if not bg.isNull():
                    self.drawing_widget.setBackgroundImage(bg)
                    self.orig_width = bg.width()
                    self.orig_height = bg.height()
                self.cap = None
                self.total_frames = 1
                self.is_video = False

        # Load background
        bgpath = fi.get("background_path")
        self.background_file = bgpath
        if bgpath and os.path.isfile(bgpath):
            bg = QImage(bgpath)
            if not bg.isNull():
                self.drawing_widget.setBackgroundImage(bg)

        # Reset canvas
        self.drawing_widget.shapes.clear()
        self.drawing_widget.clearCurrentShape()
        blank_mask = QImage(self.orig_width, self.orig_height, QImage.Format_Grayscale8)
        blank_mask.fill(255)
        self.drawing_widget.mask_image = blank_mask
        self.drawing_widget.zones_overlay = None
        self.drawing_widget.points_by_frame = {}
        self._set_timeline_limits(fi)
        limits = self._file_states.get(new_idx, {}).get("Frame limits", {})
        if limits:
            self._set_timeline_limits(limits)
        self.timeline.invalidate()

        # Update video controls
        if hasattr(self, "video_slider"):
            self.video_slider.setRange(0, max(0, self.total_frames - 1))
        if hasattr(self, "frame_spin"):
            self.frame_spin.setRange(1, max(1, self.total_frames))
        self.current_frame_idx = 0
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            QTimer.singleShot(0, self.nextFrame)

        self.onOpModeChanged(self.opmode_combo.currentIndex())
        self._update_nav_label()
        self.drawing_widget.update()
        self.proxyUpdate()

    def _get_current_purpose_data(self):
        """Return the current drawing data for the active purpose."""
        opmode_text = self.currentOperationMode()
        opmode = _PURPOSE_MAP.get(opmode_text, opmode_text)

        if opmode == "mask":
            img = self.drawing_widget.mask_image
            if img is None or img.isNull():
                return None
            arr = qimage_to_numpy(img)
            if getattr(self.drawing_widget, "inverted", False):
                arr = 255 - arr
            return arr

        elif opmode == "roi":
            if self.drawing_widget.shapes:
                shape = self.drawing_widget.shapes[-1][1]
                return tuple(shape[i] for i in (0, 2))
            return None

        elif opmode == "zones":
            self.drawing_widget.commitZones()
            if self.drawing_widget.zones_overlay:
                white_bg = QImage(self.drawing_widget.zones_overlay.size(), QImage.Format_ARGB32)
                white_bg.fill(Qt.white)
                painter = QPainter(white_bg)
                painter.drawImage(0, 0, self.drawing_widget.zones_overlay)
                painter.end()
                return qimage_to_numpy(white_bg)
            return None

        elif opmode == "framelimits":
            return (self.flim_start_spin.value(), self.flim_stop_spin.value())

        elif opmode == "timepoints":
            data = []
            for id_, typedict in self.drawing_widget.points_by_frame.items():
                frames = set()
                for typ in ["c", "h", "t", "a"]:
                    frames.update(typedict.get(typ, {}).keys())
                for f in sorted(frames):
                    row = {"frame": f, "id": id_}
                    for coord_key, pt_key in [("cx", "c"), ("hx", "h"), ("tx", "t")]:
                        pt = typedict.get(pt_key, {}).get(f)
                        if pt is not None and hasattr(pt, "x"):
                            row[coord_key] = pt.x()
                            row[coord_key.replace("x", "y")] = pt.y()
                        else:
                            row[coord_key] = None
                            row[coord_key.replace("x", "y")] = None
                    row["angle"] = typedict.get("a", {}).get(f)
                    data.append(row)
            return pd.DataFrame(data, columns=["frame", "id", "cx", "cy", "hx", "hy", "tx", "ty", "angle"]).sort_values(["id", "frame"]).reset_index(drop=True)

        elif opmode == "measure":
            pts = getattr(self.drawing_widget, "measure_polyline_orig", [])
            if len(pts) >= 2:
                shape = [(pt.x(), pt.y()) for pt in pts]
                total = sum(
                    math.hypot(shape[i + 1][0] - shape[i][0], shape[i + 1][1] - shape[i][1])
                    for i in range(len(shape) - 1)
                )
                area = None
                if len(shape) >= 4:
                    area = abs(sum(
                        shape[i][0] * shape[(i + 1) % len(shape)][1]
                        - shape[(i + 1) % len(shape)][0] * shape[i][1]
                        for i in range(len(shape))
                    )) / 2.0
                return (shape, total, area)
            return None

        elif opmode in ("thresholding", "thresholding color"):
            return dict(self.thresh_params)

        return None

    def _store_current(self):
        """Store the current drawing for the current file and purpose."""
        if not self._file_infos:
            return
        self.drawing_widget.addCurrentShapeIfNeeded()
        fi = self._file_infos[self._file_idx]
        purpose_text = self.opmode_combo.currentText()
        opmode_text = purpose_text.lower()
        opmode = _PURPOSE_MAP.get(opmode_text, opmode_text)

        data = self._get_current_purpose_data()
        if data is None:
            QMessageBox.information(self, "Nothing to store",
                f"No data drawn for purpose: {purpose_text}")
            return

        if opmode == "measure":
            mm, ok = QInputDialog.getDouble(self, "Calibration", "Known length (mm):", 1.0, 0.000001, 1e9, 6)
            if not ok:
                return
            if data[1] <= 0:
                QMessageBox.warning(self, "Invalid measurement", "Draw a line with nonzero length.")
                return
            data = mm / data[1]
        try:
            if self._save_callback:
                self._save_callback(self._file_idx, fi.get("ind"), opmode, data)
        except Exception as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
            self._remember_current()
            return

        self._stored.add((self._file_idx, opmode))
        self._saved_states[(self._file_idx, purpose_text)] = self._collect_state(purpose_text)
        self._remember_current()
        name = fi.get("output_basename", fi.get("video_name", f"File {self._file_idx + 1}"))
        self.statusBar().showMessage(f"Saved {purpose_text} for {name}")
        print(f"Saved {purpose_text} for {name}")

    def _save_all(self):
        """Close, letting closeEvent check all video/purpose edits."""
        self.drawing_widget.final_output = "saved"
        save_prefs(self._collect_prefs())
        self.close()
