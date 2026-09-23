#! /usr/bin/env python

from ._utils import *

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

class PyQt5ShapeDrawer(QWidget):
    modeChanged = pyqtSignal(str)  # Emitted when drawing mode is changed

    def __init__(self, file=None, is_video=False, mask_qimg=None, width=640, height=480, mode="default"):
        super().__init__()
        self.orig_width = width
        self.orig_height = height

        self.mode_list = ["line", "rectangle", "polygon", "circle", "ellipse", "point"]
        self.drawing_mode = self.mode_list[0]
        self.drawing_overlay = None  # Optional extra layer (e.g. contour drawing)
        self.shapes = []  # List of (mode, shape_data)
        self.main_window = None
        self.test_points = []
        self.scope_mode = "any"
        self.tp_current_id = 1
        self.show_loop = False
        self.loop = False
        self.LOOP_SIZE = 80
        self.is_video = is_video

        self.helperlines_enabled = False  # Toggle for helperlines

        # State for current shape
        self.start_point_orig = None
        self.end_point_orig = None
        self.polygon_points_orig = []
        self.measure_polyline_orig = []
        # For "point" mode in video: one point per frame     
        self.points_by_frame = defaultdict(lambda: defaultdict(dict))  # id -> ptype -> frame -> QPoint
        self.points_orig = []      # for image mode
        self.temp_point_orig = None
        self.drawing_active = False
        self.angle_mode_active = False
        self.edit_drag_active = False
        self.edit_drag_target = None
        self.last_click_orig = None
        self.final_output = None

        # --- Timepoints mode state ---
        self.tp_total_ids = 1
        self.tp_current_id = 1
        self.tp_show_lines = False
        self.tp_line_window = 50  # Default frame window
        self.tp_id_colors = {}  # id -> QColor
        self.move_mode_active = False
        self.hovered_arrow = None  # (id_num, frame, tip_point) if hovering an arrow
        self.arrow_tip_hover_thresh = 14  # Pixels for detecting arrow tip hover
        
        if mode == "zones":
            # Saved JPEG zones load as RGB888; normalize them for painting.
            if mask_qimg is not None and mask_qimg.width() == self.orig_width and mask_qimg.height() == self.orig_height:
                self.zones_overlay = mask_qimg.convertToFormat(QImage.Format_ARGB32)
            else:
                self.zones_overlay = QImage(self.orig_width, self.orig_height, QImage.Format_ARGB32)
                self.zones_overlay.fill(QColor(0, 0, 0, 0))  # transparent
            # Always start with a blank mask image (not used in zones mode, but needed for consistency)
            self.mask_image = QImage(self.orig_width, self.orig_height, QImage.Format_Grayscale8)
            self.mask_image.fill(255)
        else:
            if mask_qimg is not None:
                self.mask_image = mask_qimg
            else:
                self.mask_image = QImage(self.orig_width, self.orig_height, QImage.Format_Grayscale8)
                self.mask_image.fill(255)
            self.zones_overlay = QImage(self.orig_width, self.orig_height, QImage.Format_ARGB32)
            self.zones_overlay.fill(QColor(0, 0, 0, 0))
        
        self.zone_shapes = []
        self.show_zones = False  # toggled by the checkbox
        self.zone_color_index = 0
        self.zone_colors = [QColor.fromHsv(h, 255, 255).rgb() for h in range(0, 360, 36)]
        self.background_image = None

        self.crosshair_enabled = False
        self.mouse_pos = QPoint(0, 0)
        self.drawing_color = QColor(0, 255, 0)  # default green

        self.blackwhite = False
        self.show_mask = False
        self.inverted = False

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    def keyPressEvent(self, event):
        if self.main_window:
            self.main_window.keyPressEvent(event)
        else:
            self.close()
        QTimer.singleShot(0, self.setFocus)

    def setShowMask(self, checked):
        self.show_mask = checked
        self.update()
    
    def setHelperlinesEnabled(self, checked):
        self.helperlines_enabled = checked
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
                if self.mask_image is None or self.mask_image.isNull():
                    self.mask_image = QImage(new_w, new_h, QImage.Format_Grayscale8)
                    self.mask_image.fill(255)
        self.update()

    def updateZonesOverlay(self):
        # Only refresh temporary preview – do NOT clear existing zone drawings
        if not self.zones_overlay:
            self.zones_overlay = QImage(self.orig_width, self.orig_height, QImage.Format_ARGB32)
            self.zones_overlay.fill(QColor(255, 255, 255))  # white background

        # Create a transparent layer for temporary shapes only
        temp_overlay = QImage(self.orig_width, self.orig_height, QImage.Format_ARGB32)
        temp_overlay.fill(QColor(0, 0, 0, 0))  # fully transparent

        painter = QPainter(temp_overlay)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 200)))  # semi-transparent black

        for mode, shape_data in self.shapes:
            if mode == "rectangle":
                x1, y1 = shape_data[0]
                x2, y2 = shape_data[2]
                painter.drawRect(QRect(x1, y1, x2 - x1, y2 - y1))
            elif mode == "polygon":
                points = [QPoint(x, y) for (x, y) in shape_data]
                painter.drawPolygon(QPolygon(points))
            elif mode == "circle":
                (cx, cy), radius = shape_data
                painter.drawEllipse(QPoint(cx, cy), radius, radius)
            elif mode == "ellipse":
                (cx, cy), (rx, ry) = shape_data
                painter.drawEllipse(QPoint(cx, cy), rx, ry)

        painter.end()

        # Composite: zones_overlay + temp_overlay (to show both confirmed and temporary zones)
        combined = QPainter(self.zones_overlay)
        combined.setCompositionMode(QPainter.CompositionMode_SourceOver)
        combined.drawImage(0, 0, temp_overlay)
        combined.end()

        self.update()

    def draw_mouse_loop(self, painter):
        # Only draw if background image is present
        if not self.loop or not self.background_image or self.background_image.isNull():
            return

        # Get mouse position in original image coordinates
        mouse_x, mouse_y = self.convertToOriginal(self.mouse_pos).x(), self.convertToOriginal(self.mouse_pos).y()

        # Clamp so we don't go out of bounds
        half_size = self.LOOP_SIZE // 2
        x1 = max(0, mouse_x - half_size)
        y1 = max(0, mouse_y - half_size)
        x2 = min(self.orig_width, mouse_x + half_size)
        y2 = min(self.orig_height, mouse_y + half_size)

        # Extract region from original QImage (background)
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return
        region = self.background_image.copy(x1, y1, w, h)
        if region.isNull():
            return
        region = region.scaled(self.LOOP_SIZE, self.LOOP_SIZE, Qt.KeepAspectRatio, Qt.FastTransformation)

        # Draw black border rectangle around
        border = 2
        loop_pixmap = QPixmap(self.LOOP_SIZE + 2*border, self.LOOP_SIZE + 2*border)
        loop_pixmap.fill(Qt.transparent)
        p = QPainter(loop_pixmap)
        p.fillRect(0, 0, self.LOOP_SIZE + 2*border, self.LOOP_SIZE + 2*border, Qt.black)
        p.drawImage(border, border, region)
        p.end()

        # Position loop slightly offset from mouse, but not out of bounds
        x_disp = self.mouse_pos.x() + 20
        y_disp = self.mouse_pos.y() + 20
        max_x = self.width() - loop_pixmap.width()
        max_y = self.height() - loop_pixmap.height()
        x_disp = min(x_disp, max_x)
        y_disp = min(y_disp, max_y)

        painter.drawPixmap(x_disp, y_disp, loop_pixmap)
    
    def mousePressEvent(self, event):
        # --- MOVE MODE: only allow moving, block drawing ---
        if getattr(self, "move_mode_active", False):
            # Determine move scope
            move_mode = getattr(self.main_window, "edit_scope", None)
            cur_ptype = getattr(self.main_window, "current_ptype", "c")
            mouse_pt = self.convertToOriginal(event.pos())
            nearest = None
            nearest_dist = float("inf")
            nearest_id = None
            nearest_frame = None

            # Build list of candidate IDs
            id_list = []
            if move_mode == "any":
                id_list = [id_ for id_ in self.points_by_frame.keys() if id_ <= self.main_window.tp_total_ids]
            elif move_mode == "id":
                cur_id = self.main_window.tp_current_id
                if cur_id > self.main_window.tp_total_ids:
                    self.edit_drag_active = False
                    self.edit_drag_target = None
                    return
                id_list = [cur_id]

            # Find nearest point to drag
            for id_ in id_list:
                frame_dict = self.points_by_frame.get(id_, {}).get(cur_ptype, {})
                for f, pt in frame_dict.items():
                    try:
                        f_int = int(f)
                    except Exception:
                        continue
                    if abs(f_int - self.main_window.current_frame_idx) > self.main_window.tp_visible_range:
                        continue
                    dist = (pt - mouse_pt).manhattanLength()
                    if dist < nearest_dist:
                        nearest = pt
                        nearest_dist = dist
                        nearest_id = id_
                        nearest_frame = f_int

            if nearest is not None:
                self.edit_drag_active = True
                self.edit_drag_target = (nearest_id, nearest_frame)
                return  # DRAGGING A POINT, exit after setting target

            # Not dragging, but move mode is on: block drawing
            self.edit_drag_active = False
            self.edit_drag_target = None
            return  # <-- KEY: block further drawing code

        # --- Angle tip click: Jump to angle's frame ---
        if (
            self.main_window.opmode_combo.currentText().lower() == "timepoints" and
            self.main_window.show_nearframe_checkbox.isChecked() and
            hasattr(self, "hovered_arrow") and
            self.hovered_arrow is not None and
            event.button() == Qt.LeftButton
        ):
            id_num, frame, angle_deg, tip_pt = self.hovered_arrow
            # Set current frame in main window
            self.main_window.current_frame_idx = frame
            if hasattr(self.main_window, "frame_spin"):
                self.main_window.frame_spin.setValue(frame + 1)
            if hasattr(self.main_window, "video_slider"):
                self.main_window.video_slider.setValue(frame)
            # Optionally: Switch to angle edit mode and set correct ID
            self.main_window.rb_angle.setChecked(True)
            if hasattr(self.main_window, "current_id_box"):
                self.main_window.tp_current_id = id_num
                self.main_window.current_id_box.setValue(id_num + 1)
            self.update()
            self.main_window.proxyUpdate()
            print(f"Jumped to angle: ID={id_num}, frame={frame}")
            return
        
        # ANGLE MODE: change angle by click (show angles/arrows/lines enabled)
        if getattr(self, "angle_mode_active", False) and self.main_window and self.main_window.opmode_combo.currentText().lower() == "timepoints":
            cur_frame = self.main_window.current_frame_idx + 1 if self.main_window.is_video else 1
            id_ = self.main_window.tp_current_id if self.main_window.tp_total_ids > 1 else 1
            ptype = "c"
            pt = self.points_by_frame.get(id_, {}).get(ptype, {}).get(cur_frame)
            if pt is not None:
                click = self.convertToOriginal(event.pos())
                dx = click.x() - pt.x()
                dy = pt.y() - click.y()   # Note: pt.y() - click.y() so "up" is positive (Y decreases)
                angle = np.degrees(np.arctan2(dx, dy))

                # --- UNDO SUPPORT for angle ---
                prev_angle = self.points_by_frame.get(id_, {}).get('a', {}).get(cur_frame, None)
                if id_ not in self.points_by_frame:
                    self.points_by_frame[id_] = {'c': {}, 'h': {}, 't': {}, 'a': {}}
                if 'a' not in self.points_by_frame[id_]:
                    self.points_by_frame[id_]['a'] = {}
                self.main_window._undo_buffer = {
                    "id": id_,
                    "ptype": "a",
                    "points": {cur_frame: prev_angle},
                }

                # Set new angle
                self.points_by_frame[id_]['a'][cur_frame] = angle
                self.last_click_orig = click
                self.update()
                if self.main_window:
                    self.main_window.proxyUpdate()
                print(f"Set angle for ID={int(id_)+1}, frame={cur_frame}: {angle:.1f} deg")
            else:
                print("No centroid to set angle for current frame/ID.")
            self.setFocus()
            return

        # --- DRAW MODE: normal drawing allowed ---
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
            # --- MEASURE MODE: Polyline drawing ---
            if self.main_window and self.main_window.opmode_combo.currentText().lower() == "measure":
                self.measure_polyline_orig.append(orig_pos)
                self.last_click_orig = orig_pos
                self.update()
                if self.main_window:
                    self.main_window.proxyUpdate()
                return  # Don't process further as normal shape
            if self.drawing_mode in ["line", "rectangle", "circle", "ellipse"]:
                self.start_point_orig = orig_pos
                self.end_point_orig = orig_pos
                self.drawing_active = True
            elif self.drawing_mode == "polygon":
                self.polygon_points_orig.append(orig_pos)
            elif self.drawing_mode == "points":
                self.points_orig.append(orig_pos)
            elif self.drawing_mode == "point":
                orig_pos = self.convertToOriginal(event.pos())
                if self.main_window and self.main_window.opmode_combo.currentText().lower() == "timepoints":
                    cur_frame = self.main_window.current_frame_idx + 1 if self.main_window.is_video else 1
                    id_ = self.main_window.tp_current_id
                    cur_ptype = getattr(self.main_window, "current_ptype", "c")

                    # Guarantee one point per frame per ID per type: move or add
                    if id_ not in self.points_by_frame:
                        self.points_by_frame[id_] = {'c': {}, 'h': {}, 't': {}, 'a': {}}

                    # === UNDO SUPPORT for drawing/moving a point by click ===
                    prev_val = self.points_by_frame[id_][cur_ptype].get(cur_frame, None)
                    if self.main_window is not None:
                        self.main_window._undo_buffer = {
                            "id": id_,
                            "ptype": cur_ptype,
                            "points": {cur_frame: prev_val},
                        }

                    self.points_by_frame[id_][cur_ptype][cur_frame] = orig_pos
                else:
                    # Non-timepoints mode, behave as before
                    self.points_orig.append(orig_pos)

            self.last_click_orig = orig_pos
            self.update()
            self.setFocus()
            if self.main_window:
                self.main_window.proxyUpdate()


    def mouseMoveEvent(self, event):
        # ANGLE MODE: Drag to set angle, supports undo
        if (
            getattr(self, "angle_mode_active", False)
            and self.edit_drag_active
            and self.edit_drag_target
        ):
            id_, frame = self.edit_drag_target
            # For now: angle always refers to the centroid point
            pt = self.points_by_frame[id_]['c'][frame]
            mouse_pt = self.convertToOriginal(event.pos())
            dx = mouse_pt.x() - pt.x()
            dy = mouse_pt.y() - pt.y()
            angle = (math.degrees(math.atan2(dx, -dy)) + 360) % 360
            # Save previous value the first time only
            if not hasattr(self, "_angle_undo_prev_value") or self._angle_undo_prev_value is None:
                self._angle_undo_prev_value = self.points_by_frame[id_]['a'].get(frame, None)
            self.points_by_frame[id_]['a'][frame] = angle
            # You can print feedback if wanted:
            # print(f"Set angle for ID={id_}, frame={frame}: {angle:.1f} deg")
            self.update()
            if self.main_window:
                self.main_window.proxyUpdate()
            return

        # MOVE MODE (points)
        if (
            getattr(self, "move_mode_active", False)
            and self.edit_drag_active
            and self.edit_drag_target
        ):
            id_, frame = self.edit_drag_target
            cur_ptype = getattr(self.main_window, "current_ptype", "c")
            if id_ <= self.main_window.tp_total_ids:
                pt = self.convertToOriginal(event.pos())
                # Save undo only at the start of the move
                if not hasattr(self, "_move_undo_buffer") or not getattr(self, "_move_undo_buffer_active", False):
                    old_pt = self.points_by_frame[id_][cur_ptype].get(frame)
                    if old_pt is not None:
                        self._move_undo_buffer = {
                            "id": id_,
                            "ptype": cur_ptype,
                            "frame": frame,
                            "point": old_pt,
                        }
                        self._move_undo_buffer_active = True
                # Move point
                self.points_by_frame[id_][cur_ptype][frame] = pt
                self.update()
                self.main_window.proxyUpdate()
            return

        # All other interactions (drawing, hover, etc.)
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
        if getattr(self, "move_mode_active", False) and self.edit_drag_active:
            id_, frame = self.edit_drag_target
            cur_ptype = getattr(self.main_window, "current_ptype", "c")
            print(f"Moved point for ID={id_} @ frame {frame}", end=" ")
            # Copy move buffer into undo buffer
            if hasattr(self, "_move_undo_buffer") and getattr(self, "_move_undo_buffer_active", False):
                self.main_window._undo_buffer = {
                    "id": self._move_undo_buffer["id"],
                    "ptype": self._move_undo_buffer["ptype"],
                    "points": {self._move_undo_buffer["frame"]: self._move_undo_buffer["point"]},
                }
                self._move_undo_buffer = {}
                self._move_undo_buffer_active = False
            self.edit_drag_active = False
            self.edit_drag_target = None
            return
        if getattr(self, "angle_mode_active", False) and self.edit_drag_active:
            id_, frame = self.edit_drag_target
            # Only do this if an angle was changed (should always be True if edit_drag_active)
            if hasattr(self, "_angle_undo_prev_value"):
                self.main_window._undo_buffer = {
                    "action": "angle_edit",
                    "id": id_,
                    "frame": frame,
                    "prev_angle": self._angle_undo_prev_value
                }
                self._angle_undo_prev_value = None  # Reset for next time
            self.edit_drag_active = False
            self.edit_drag_target = None
            return
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
            # Shape is now finished but stays visible for cropping
            self.temp_point_orig = None
            self.update()
            
    def clearCurrentShape(self):
        self.start_point_orig = None
        self.end_point_orig = None
        self.polygon_points_orig = []
        self.points_orig = []
        self.temp_point_orig = None
        self.measure_polyline_orig = []

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
            if (self.main_window and 
                self.main_window.opmode_combo.currentText().lower() == "zones" and 
                self.main_window.cb_showzones.isChecked()):
                self.updateZonesOverlay()
            self.clearCurrentShape()
    
    def addCurrentShapeIfNeeded(self):
        mode = self.drawing_mode
        if mode in ["line", "rectangle", "circle", "ellipse"]:
            if self.start_point_orig and self.end_point_orig:
                self.addCurrentShape()
        elif mode == "polygon" and self.polygon_points_orig:
            if self.polygon_points_orig[0] != self.polygon_points_orig[-1]:
                self.polygon_points_orig.append(self.polygon_points_orig[0])
            self.addCurrentShape()
        elif mode in ["point", "points"]:
            if self.points_orig:
                self.addCurrentShape()

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
            bg = qimg_to_grayscale(bg)
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
            alpha = self.main_window.bgtrans_slider.value() / 100.0
            painter.setOpacity(alpha)
            painter.drawImage(int(offset_x), int(offset_y), bg)
            painter.setOpacity(1.0)  # reset to full opacity for other drawing
        else:
            painter.fillRect(self.rect(), QColor(255, 255, 255))

        # --- DRAW MASK IMAGE ---
        if self.show_mask and not self.mask_image.isNull():
            mask_opacity = self.main_window.mask_op_slider.value() / 100.0
            painter.setOpacity(mask_opacity)
            mask_disp = self.mask_image.scaled(
                int(self.orig_width * scale),
                int(self.orig_height * scale),
                Qt.IgnoreAspectRatio,
                Qt.SmoothTransformation
            )
            if self.inverted:
                mask_disp.invertPixels()
            painter.drawImage(int(offset_x), int(offset_y), mask_disp)
            painter.setOpacity(1.0)

            # Draw a thin border around the mask outline
            try:
                ptr = self.mask_image.bits()
                ptr.setsize(self.mask_image.byteCount())
                arr = np.frombuffer(ptr, np.uint8).reshape(
                    (self.mask_image.height(), self.mask_image.width(), -1))
                gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.shape[2] >= 3 else arr[:, :, 0]
                if self.inverted:
                    gray = cv2.bitwise_not(gray)
                contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                border_pen = QPen(QColor(0, 0, 0, 200), 2)
                painter.setPen(border_pen)
                painter.setBrush(Qt.NoBrush)
                for contour in contours:
                    pts = [QPoint(
                        int(offset_x + p[0][0] * scale),
                        int(offset_y + p[0][1] * scale)
                    ) for p in contour]
                    if pts:
                        painter.drawPolygon(QPolygon(pts))
            except Exception:
                pass
        
        # ZONES MODE: draw colored zone overlay and temporary shapes
        if self.main_window.opmode_combo.currentText().lower() == "zones" and self.show_zones:
            if self.zones_overlay:
                scaled_overlay = self.zones_overlay.scaled(
                    int(self.orig_width * scale),
                    int(self.orig_height * scale),
                    Qt.IgnoreAspectRatio,
                    Qt.SmoothTransformation
                )
                painter.setOpacity(0.5)
                painter.drawImage(int(offset_x), int(offset_y), scaled_overlay)
                painter.setOpacity(1.0)

            # Draw currently drawn shapes as semi-transparent black
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(0, 0, 0, 200)))
            for mode, shape_data in self.shapes:
                if mode == "rectangle":
                    x1, y1 = shape_data[0]
                    x2, y2 = shape_data[2]
                    p1 = self.convertToDisplay(QPoint(x1, y1))
                    p2 = self.convertToDisplay(QPoint(x2, y2))
                    painter.drawRect(QRect(p1, p2))
                elif mode == "polygon":
                    points = [self.convertToDisplay(QPoint(x, y)) for (x, y) in shape_data]
                    painter.drawPolygon(QPolygon(points))
                elif mode == "circle":
                    (cx, cy), radius = shape_data
                    center = self.convertToDisplay(QPoint(cx, cy))
                    painter.drawEllipse(center, radius, radius)
                elif mode == "ellipse":
                    (cx, cy), (rx, ry) = shape_data
                    center = self.convertToDisplay(QPoint(cx, cy))
                    painter.drawEllipse(center, rx, ry)

        # Other drawing overlay (e.g., mask shapes)
        elif self.drawing_overlay and self.main_window.cb_showoverlay.isChecked():
            scaled_overlay = self.drawing_overlay.scaled(
                int(self.orig_width * scale),
                int(self.orig_height * scale),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            painter.setOpacity(1.0)
            painter.drawImage(int(offset_x), int(offset_y), scaled_overlay)

        # --- DRAW CURRENTLY ACTIVE SHAPE ---
        painter.setPen(QPen(self.drawing_color, 3))
        mode = self.drawing_mode

        # --- DRAW MEASURE POLYLINE ---
        if self.main_window and self.main_window.opmode_combo.currentText().lower() == "measure":
            if self.measure_polyline_orig:
                pts_disp = [self.convertToDisplay(pt) for pt in self.measure_polyline_orig]
                if len(pts_disp) >= 2:
                    painter.setPen(QPen(self.drawing_color, 3))
                    painter.drawPolyline(QPolygon(pts_disp))
                for p in pts_disp:
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QBrush(self.drawing_color))
                    painter.drawEllipse(p, 5, 5)

        if mode in ["line", "rectangle", "circle", "ellipse"]:
            if self.start_point_orig and self.end_point_orig:
                p1 = self.convertToDisplay(self.start_point_orig)
                p2 = self.convertToDisplay(self.end_point_orig)

                # Always draw with outline in drawing color, no fill during drag
                painter.setPen(QPen(self.drawing_color, 2))
                painter.setBrush(Qt.NoBrush)

                if mode == "line":
                    painter.drawLine(p1, p2)
                elif mode == "rectangle":
                    painter.drawRect(QRect(p1, p2).normalized())
                elif mode == "circle":
                    center = QPoint((p1.x() + p2.x()) // 2, (p1.y() + p2.y()) // 2)
                    radius = int(math.hypot(p2.x() - p1.x(), p2.y() - p1.y()) / 2)
                    painter.drawEllipse(center, radius, radius)
                elif mode == "ellipse":
                    center = QPoint((p1.x() + p2.x()) // 2, (p1.y() + p2.y()) // 2)
                    rx = abs(p2.x() - p1.x()) // 2
                    ry = abs(p2.y() - p1.y()) // 2
                    painter.drawEllipse(center, rx, ry)

        elif mode == "polygon":
            if self.polygon_points_orig:
                pts_disp = [self.convertToDisplay(pt) for pt in self.polygon_points_orig]
                painter.drawPolyline(QPolygon(pts_disp))
                if self.temp_point_orig:
                    painter.drawLine(pts_disp[-1], self.convertToDisplay(self.temp_point_orig))

        elif mode == "points":
            for pt in self.points_orig:
                center = self.convertToDisplay(pt)
                color = QColor(self.drawing_color)
                color.setAlphaF(self.main_window.point_opacity)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(color))
                painter.drawEllipse(center, self.main_window.point_size_slider.value(), self.main_window.point_size_slider.value())

        elif mode == "point":
            if self.main_window and self.main_window.opmode_combo.currentText().lower() != "timepoints":
                for pt in self.points_orig:
                    center = self.convertToDisplay(pt)
                    color = QColor(self.drawing_color)
                    color.setAlphaF(self.main_window.point_opacity)
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QBrush(color))
                    painter.drawEllipse(center, self.main_window.point_size_slider.value(), self.main_window.point_size_slider.value())

        # --- DRAW TIMEPOINTS ---
        if self.main_window.opmode_combo.currentText().lower() == "timepoints":
            cur_frame = self.main_window.current_frame_idx + 1 if self.main_window.is_video else 1
            visible_range = self.main_window.tp_visible_range
            ptype = self.main_window.current_ptype if self.main_window else "c"
            arrow_len = getattr(self.main_window, "arrow_length", 40)

            if self.scope_mode == "id":
                id_nums = [self.tp_current_id]
            else:
                id_nums = list(self.points_by_frame.keys())

            for id_num in id_nums:
                ptypedict = self.points_by_frame.get(id_num, {})
                frame_dict = {int(k): v for k, v in ptypedict.get(ptype, {}).items()}
                angle_dict = {int(k): v for k, v in ptypedict.get("a", {}).items()}
                if self.main_window.rb_range_past.isChecked():
                    start_frame = max(0, cur_frame - visible_range)
                    end_frame = cur_frame
                elif self.main_window.rb_range_future.isChecked():
                    start_frame = cur_frame
                    end_frame = cur_frame + visible_range
                else:
                    start_frame = max(0, cur_frame - visible_range)
                    end_frame = cur_frame + visible_range
                frames_sorted = sorted([f for f in frame_dict if start_frame <= f <= end_frame])
                points_in_range = [(f, frame_dict[f]) for f in frames_sorted]

                # --- 1. Draw all points except current frame (if highlight is ON) ---
                pt_radius = self.main_window.point_size_slider.value()
                fade_enabled = self.main_window.fade_points_checkbox.isChecked()
                if self.main_window.show_points_checkbox.isChecked():
                    for f, pt in points_in_range:
                        if f == cur_frame and self.main_window.highlight_current_checkbox.isChecked():
                            continue  # Skip current frame, will draw highlight below
                        center = self.convertToDisplay(pt)
                        color = self.main_window.tp_id_colors[id_num % len(self.main_window.tp_id_colors)]
                        color = QColor(color)
                        base_opacity = self.main_window.point_opacity
                        if fade_enabled and visible_range > 0:
                            dist_ratio = abs(f - cur_frame) / visible_range
                            base_opacity = base_opacity * max(0.0, 1.0 - dist_ratio)
                        color.setAlphaF(base_opacity)
                        painter.setPen(Qt.NoPen)
                        painter.setBrush(QBrush(color))
                        painter.drawEllipse(center, pt_radius, pt_radius)
                        if self.main_window.show_framenrs_checkbox.isChecked():
                            painter.setPen(Qt.black)
                            painter.setFont(self.font())
                            painter.drawText(center + QPoint(7, -7), str(f))

                # --- 2. Draw lines between points ---
                if self.main_window.tp_show_lines and len(points_in_range) > 1:
                    for i in range(len(points_in_range) - 1):
                        f1, pt1 = points_in_range[i]
                        f2, pt2 = points_in_range[i + 1]
                        dist = min(abs(cur_frame - f1), abs(cur_frame - f2))
                        alpha = int(255 * (1 - dist / visible_range))
                        color = QColor(self.main_window.tp_id_colors[id_num % len(self.main_window.tp_id_colors)])
                        color.setAlpha(alpha)
                        painter.setPen(QPen(color, 2))
                        p1 = self.convertToDisplay(pt1)
                        p2 = self.convertToDisplay(pt2)
                        painter.drawLine(p1, p2)

                # --- 3. Draw all angle arrows/lines except current frame ---
                if self.main_window.angle_show_arrows_checkbox.isChecked() or self.main_window.angle_show_lines_checkbox.isChecked():
                    for f in sorted(frame_dict):
                        if not (start_frame <= f <= end_frame):
                            continue
                        if f in angle_dict:
                            pt = frame_dict[f]
                            center = self.convertToDisplay(pt)
                            angle_deg = angle_dict[f]
                            angle_rad = np.radians(angle_deg - 90) 
                            dx = arrow_len * np.cos(angle_rad)
                            dy = arrow_len * np.sin(angle_rad)
                            end_point = QPoint(int(center.x() + dx), int(center.y() + dy))
                            normalized_angle = (180 - angle_deg) % 360
                            if normalized_angle > 180:
                                normalized_angle -= 360
                            norm = abs(normalized_angle) / 180.0
                            red = int(255 * norm)
                            green = int(255 * (1 - norm))
                            color = QColor(0, green, red)
                            painter.setPen(QPen(color, 2))
                            if self.main_window.angle_show_lines_checkbox.isChecked():
                                painter.drawLine(center, end_point)
                            if self.main_window.angle_show_arrows_checkbox.isChecked():
                                draw_arrow_head(painter, end_point, size=10, angle_rad=angle_rad)

                # --- 4. Optionally highlight current frame point and angle ---
                if self.main_window.highlight_current_checkbox.isChecked():

                    # --- Current point ---
                    if self.main_window.show_points_checkbox.isChecked():
                        pt = frame_dict.get(cur_frame)
                        if pt is not None:
                            center = self.convertToDisplay(pt)
                            painter.setPen(Qt.NoPen)
                            painter.setBrush(Qt.black)
                            painter.drawEllipse(center, pt_radius + 1, pt_radius + 1)
                            if self.main_window.show_framenrs_checkbox.isChecked():
                                painter.setPen(Qt.black)
                                painter.setFont(self.font())
                                painter.drawText(center + QPoint(7, -7), str(cur_frame))

                    # --- Current angle ---
                    if (self.main_window.angle_show_arrows_checkbox.isChecked() or self.main_window.angle_show_lines_checkbox.isChecked()) and cur_frame in frame_dict and cur_frame in angle_dict:
                        pt = frame_dict[cur_frame]
                        center = self.convertToDisplay(pt)
                        angle_deg = angle_dict[cur_frame]
                        angle_rad = np.radians(angle_deg - 90) 
                        dx = arrow_len * np.cos(angle_rad)
                        dy = arrow_len * np.sin(angle_rad)
                        end_point = QPoint(int(center.x() + dx), int(center.y() + dy))
                        painter.setPen(QPen(Qt.black, 3))  # slightly bolder
                        if self.main_window.angle_show_lines_checkbox.isChecked():
                            painter.drawLine(center, end_point)
                        if self.main_window.angle_show_arrows_checkbox.isChecked():
                            draw_arrow_head(painter, end_point, size=10, angle_rad=angle_rad)


            if self.main_window.show_nearframe_checkbox.isChecked():
                mouse_pt = self.convertToOriginal(self.mouse_pos)
                nearest_f = None
                nearest_id = None
                nearest_dist = float("inf")
                nearest_center = None
                nearest_is_angle = False
                nearest_angle_tip = None
                nearest_angle_deg = None

                cur_frame = self.main_window.current_frame_idx if getattr(self.main_window, "is_video", False) else 0
                visible_range = self.main_window.tp_visible_range
                arrow_len = 40
                id_nums_to_draw = [self.tp_current_id] if self.scope_mode == "id" else list(self.points_by_frame.keys())

                for id_num in id_nums_to_draw:
                    ptypedict = self.points_by_frame.get(id_num, {})
                    frame_dict = ptypedict.get(ptype, {})
                    angle_dict = ptypedict.get("a", {})

                    # Points: restrict to visible range
                    for f, pt in frame_dict.items():
                        f_int = int(f)
                        if abs(f_int - cur_frame) > visible_range:
                            continue
                        dist = (pt - mouse_pt).manhattanLength()
                        if dist < nearest_dist:
                            nearest_dist = dist
                            nearest_f = f_int
                            nearest_id = id_num
                            nearest_center = self.convertToDisplay(pt)
                            nearest_is_angle = False
                            nearest_angle_tip = None

                    # Angles: restrict to visible range
                    for f, angle_deg in angle_dict.items():
                        f_int = int(f)
                        if f_int not in frame_dict:
                            continue
                        if abs(f_int - cur_frame) > visible_range:
                            continue
                        pt = frame_dict[f_int]
                        center = self.convertToDisplay(pt)
                        angle_rad = np.radians(angle_deg)
                        dx = arrow_len * np.sin(angle_rad)
                        dy = -arrow_len * np.cos(angle_rad)
                        tip_pt = QPoint(
                            int(center.x() + dx),
                            int(center.y() + dy)
                        )
                        dist = (tip_pt - self.mouse_pos).manhattanLength()
                        if dist < nearest_dist and dist < self.arrow_tip_hover_thresh:
                            nearest_dist = dist
                            nearest_f = f_int
                            nearest_id = id_num
                            nearest_center = tip_pt
                            nearest_is_angle = True
                            nearest_angle_tip = tip_pt
                            nearest_angle_deg = angle_deg

                # Show label if something found
                if nearest_center is not None and nearest_f is not None and nearest_id is not None:
                    idstr = self.main_window.idnum_to_idstr.get(nearest_id, str(nearest_id)) \
                        if hasattr(self.main_window, 'idnum_to_idstr') else str(nearest_id)
                    label = f"#{int(idstr)+1}-{nearest_f}" + (" (angle)" if nearest_is_angle else "")
                    pos = nearest_center + QPoint(7, -7)
                    # Draw white outline by drawing text in 8 directions
                    painter.setPen(QPen(Qt.white, 4))
                    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                        painter.drawText(pos + QPoint(dx, dy), label)
                    painter.setPen(QPen(Qt.black, 1))
                    painter.drawText(pos, label)
                    if nearest_is_angle:
                        self.hovered_arrow = (nearest_id, nearest_f, nearest_angle_deg, nearest_angle_tip)
                    else:
                        self.hovered_arrow = None
                else:
                    self.hovered_arrow = None


        # --- DRAW CROSSHAIR (small at mouse position) ---
        if self.crosshair_enabled:
            center = self.mouse_pos
            length = 12  # total length of crosshair arms
            gap = 3      # gap at the very center for clarity
            pen = QPen(QColor(0, 0, 0), 2)
            painter.setPen(pen)
            painter.drawLine(center.x() - length, center.y(), center.x() - gap, center.y())
            painter.drawLine(center.x() + gap, center.y(), center.x() + length, center.y())
            painter.drawLine(center.x(), center.y() - length, center.x(), center.y() - gap)
            painter.drawLine(center.x(), center.y() + gap, center.x(), center.y() + length)

        # --- DRAW HELPERLINES (full width/height at mouse position) ---
        if self.helperlines_enabled:
            painter.setPen(QPen(QColor(0, 0, 0), 2))  # solid black, 3 px thick
            painter.drawLine(self.mouse_pos.x(), 0, self.mouse_pos.x(), self.height())
            painter.drawLine(0, self.mouse_pos.y(), self.width(), self.mouse_pos.y())

        self.draw_mouse_loop(painter)

        painter.end()


