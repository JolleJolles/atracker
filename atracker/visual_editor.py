import os
import sys
import time
import math
import pandas as pd
from collections import defaultdict
from pythutils.mediautils import get_vid_params

from atracker.process_image import ProcessImage
from atracker.utils import *

# Suppress unwanted macOS IMKClient messages
os.environ["QT_MAC_DISABLE_FOREIGN_WINDOWS"] = "1"

import sys, math, cv2, numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QHBoxLayout, QVBoxLayout, QButtonGroup, QGridLayout,
    QGroupBox, QComboBox, QCheckBox, QPushButton, QSlider, QSpinBox, QRadioButton, QScrollArea, QPushButton
)
from PyQt5.QtGui import QPainter, QPen, QColor, QPolygon, QImage, QPixmap, QBrush
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
            # Only use mask_qimg as zones overlay if it is actually a zones mask, else start blank
            if mask_qimg is not None and mask_qimg.width() == self.orig_width and mask_qimg.height() == self.orig_height and mask_qimg.format() == QImage.Format_ARGB32:
                self.zones_overlay = mask_qimg
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
            painter.setOpacity(0.8)
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
                painter.drawEllipse(center, 6, 6)

        elif mode == "point":
            if self.main_window and self.main_window.opmode_combo.currentText().lower() != "timepoints":
                for pt in self.points_orig:
                    center = self.convertToDisplay(pt)
                    color = QColor(self.drawing_color)
                    color.setAlphaF(self.main_window.point_opacity)
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QBrush(color))
                    painter.drawEllipse(center, 6, 6)

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
                start_frame = max(0, cur_frame - visible_range)
                end_frame = cur_frame + visible_range
                frames_sorted = sorted([f for f in frame_dict if start_frame <= f <= end_frame])
                points_in_range = [(f, frame_dict[f]) for f in frames_sorted]

                # --- 1. Draw all points except current frame (if highlight is ON) ---
                if self.main_window.show_points_checkbox.isChecked():
                    for f, pt in points_in_range:
                        if f == cur_frame and self.main_window.highlight_current_checkbox.isChecked():
                            continue  # Skip current frame, will draw highlight below
                        center = self.convertToDisplay(pt)
                        color = self.main_window.tp_id_colors[id_num % len(self.main_window.tp_id_colors)]
                        color = QColor(color)
                        color.setAlphaF(self.main_window.point_opacity)
                        painter.setPen(Qt.NoPen)
                        painter.setBrush(QBrush(color))
                        painter.drawEllipse(center, 6, 6)
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
                            painter.drawEllipse(center, 7, 7)  # slightly bolder
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


class PyQt5ShapeDrawerWindow(QMainWindow):
    _last_esc_press_time = 0  # Class variable shared across all instances

    def __init__(self, file=None, background_file=None, threshold_dict=None, mask_qimg=None, mode="default", 
                 total_frames=1, width=1280, height=960, roi=None):
        super().__init__()

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

        # --- Operation Mode ---
        self.opmode_group = self.makeGroupBox("Operation Mode")
        self.opmode_combo = QComboBox()
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
        self.tp_current_id = 1
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
        central = QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)


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

    def on_point_opacity_changed(self, value):
        self.point_opacity = value / 100.0
        self.update()  # or self.proxyUpdate() if you have it

    def undoLastDelete(self):
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

        cur_frame = self.current_frame_idx
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
        cur_frame = self.current_frame_idx
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
        cur_frame = self.current_frame_idx
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
        w = self.drawing_widget
        mouse_pt = w.convertToOriginal(w.mouse_pos)
        current_str = f"Current position: ({mouse_pt.x()}, {mouse_pt.y()})"
        if w.last_click_orig:
            last_str = f"Last click: ({w.last_click_orig.x()}, {w.last_click_orig.y()})"
        else:
            last_str = "Last click:"
        metric = ""
        opmode = self.opmode_combo.currentText().lower()
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

        if self.opmode_combo.currentText().lower().startswith("thresholding"):
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
        self.current_frame_idx = value
        if self.cap:
            self.timer.stop()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, value)
            self.nextFrame()
        else:
            self.frame_spin.setValue(self.current_frame_idx + 1)
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
            self.video_slider.blockSignals(True)
            self.video_slider.setValue(self.current_frame_idx + 1)
            self.video_slider.blockSignals(False)
            # Always update background image
            qimg = cvMatToQImage(frame)
            self.drawing_widget.setBackgroundImage(qimg)
            if self.opmode_combo.currentText().lower() in ["thresholding", "thresholding color"]:
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

    def onOpModeChanged(self, index):
        opmode = self.opmode_combo.itemText(index).lower()

        # Clear shapes and overlay if needed
        self.drawing_widget.clearCurrentShape()
        self.drawing_widget.shapes.clear()
        if opmode not in ["thresholding", "thresholding color"]:
            self.drawing_widget.drawing_overlay = None

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

        # Trigger redraw
        self.drawing_widget.update()
        self.proxyUpdate()

    def updateThresholdingImage(self, frame=None):
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

        opmode = self.opmode_combo.currentText().lower()

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
            self.drawing_widget.addCurrentShapeIfNeeded()
            result = None
            opmode = self.opmode_combo.currentText().lower()

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
            opmode = self.opmode_combo.currentText().lower()
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
            if self.opmode_combo.currentText().lower() == "timepoints" and self.tp_total_ids > 1:
                new_val = (self.current_id_box.value() - 2) % self.tp_total_ids + 1  # subtract 2 because value() is 1-based
                self.current_id_box.setValue(new_val)
                print(f"Switched to ID: {new_val}")

        elif key == Qt.Key_O:
            if self.opmode_combo.currentText().lower() == "timepoints" and self.tp_total_ids > 1:
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
            if self.opmode_combo.currentText().lower() == "zones" and self.drawing_widget.shapes:
                painter = QPainter(self.drawing_widget.zones_overlay)
                painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
                current_color = self.drawing_widget.zone_colors[self.drawing_widget.zone_color_index % len(self.drawing_widget.zone_colors)]
                painter.setBrush(QBrush(QColor(current_color)))
                painter.setPen(Qt.NoPen)

                for mode, shape_data in self.drawing_widget.shapes:
                    if mode == "rectangle":
                        x1, y1 = shape_data[0]
                        x2, y2 = shape_data[2]
                        painter.drawRect(QRect(x1, y1, x2 - x1, y2 - y1))
                    elif mode == "polygon":
                        points = [QPoint(x, y) for x, y in shape_data]
                        painter.drawPolygon(QPolygon(points))
                    elif mode == "circle":
                        (cx, cy), radius = shape_data
                        painter.drawEllipse(QPoint(cx, cy), radius, radius)
                    elif mode == "ellipse":
                        (cx, cy), (rx, ry) = shape_data
                        painter.drawEllipse(QPoint(cx, cy), rx, ry)
                painter.end()
                self.drawing_widget.zone_color_index += 1
                self.drawing_widget.shapes.clear()
                self.drawing_widget.update()
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

def load_tracking_data_from_df(df, firstframe=None, lastframe=None):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if 'frame' not in df.columns or 'id' not in df.columns:
        raise ValueError("CSV must have columns: frame, id")

    if firstframe is not None:
        df = df[df['frame'] >= firstframe]
    if lastframe is not None:
        df = df[df['frame'] <= lastframe]

    frame_data = defaultdict(lambda: {'c': {}, 'h': {}, 't': {}, 'a': {}})

    for i, row in df.iterrows():
        id_val = int(row['id'])
        frame = int(row['frame'])

        if 'cx' in df.columns and 'cy' in df.columns:
            if pd.notnull(row['cx']) and pd.notnull(row['cy']):
                frame_data[id_val]['c'][frame] = QPoint(int(row['cx']), int(row['cy']))
        if 'hx' in df.columns and 'hy' in df.columns:
            if pd.notnull(row['hx']) and pd.notnull(row['hy']):
                frame_data[id_val]['h'][frame] = QPoint(int(row['hx']), int(row['hy']))
        if 'tx' in df.columns and 'ty' in df.columns:
            if pd.notnull(row['tx']) and pd.notnull(row['ty']):
                frame_data[id_val]['t'][frame] = QPoint(int(row['tx']), int(row['ty']))
        if 'angle' in df.columns:
            if pd.notnull(row['angle']):
                frame_data[id_val]['a'][frame] = float(row['angle'])

    for id_val in frame_data:
        for ptype in frame_data[id_val]:
            new_dict = {}
            for f, pt in frame_data[id_val][ptype].items():
                new_dict[int(f)] = pt
            frame_data[id_val][ptype] = new_dict
    return frame_data

def annotation_gui(data_file=None, media_file=None, background_file=None, mask_file=None, mode="default", threshold_dict={}, 
                   firstframe=None, lastframe=None, fileaction="overwrite", width=1280, height=960, roi=None):
    """
    Launches the interactive drawing interface.
    """

    # Defaults
    df = None
    points_by_frame = {}
    unique_idstrs = []
    idstr_to_idnum = {}
    idnum_to_idstr = {}
    total_frames = 1

# If no data_file is provided, but there is a media_file, auto-create a .csv path
    if data_file is None and media_file:
        base, _ = os.path.splitext(os.path.expanduser(media_file))
        data_file = base + ".csv"
    
    if data_file and os.path.exists(os.path.expanduser(data_file)):
        df = load_and_convert_tracking_dataframe(data_file, firstframe, lastframe)

        # If file exists but is empty, create a minimal structure
        if df is None or df.shape[0] == 0:
            df = pd.DataFrame(columns=["frame", "IDstr", "x", "y"])
            points_by_frame = {}
            unique_idstrs = []
            # idstr_to_idnum and idnum_to_idstr stay empty {}
            total_frames = 1
        else:
            # --- Ensure IDstr column exists ---
            if "IDstr" not in df.columns:
                df["IDstr"] = ""

            # Map unique string IDs to int (0, 1, ...)
            unique_idstrs = sorted(
                [s for s in df["IDstr"].unique() if pd.notna(s)],
                key=str
            )
            idstr_to_idnum = {s: i for i, s in enumerate(unique_idstrs)}   # "str" → int
            idnum_to_idstr = {i: s for s, i in idstr_to_idnum.items()}     # int → "str"

            df["ID"] = df["IDstr"].map(idstr_to_idnum).fillna(-1).astype(int)

            tp_total_ids = len(unique_idstrs)

            # Determine total_frames safely
            if "frame" in df.columns and df["frame"].notna().any():
                maxframe = df["frame"].max()
                total_frames = int(maxframe) if pd.notna(maxframe) else 1
            else:
                total_frames = 1

            # Build points_by_frame with idnum as the key!
            points_by_frame = build_points_by_frame(df) if len(df) > 0 else {}

            print(f"Loaded {len(df)} points")
    else:
        # No file present → start empty
        df = pd.DataFrame(columns=["frame", "IDstr", "x", "y"])
        points_by_frame = {}
        unique_idstrs = []
        # idstr_to_idnum and idnum_to_idstr already initialised as {}
        total_frames = 1

    if firstframe is not None and lastframe is not None:
        try:
            total_frames = max(1, int(lastframe) - int(firstframe) + 1)
        except:
            total_frames = 1

    media_arg = os.path.expanduser(media_file) if media_file else None

    # Check if background file exists (if provided)
    if background_file:
        background_file = os.path.expanduser(background_file)
        if not os.path.exists(os.path.expanduser(background_file)):
            print(f"Background file not found: {background_file}")
            return None
    
    # Load mask or zones overlay if provided
    mask_qimg = None
    if mask_file is not None:
        if not os.path.isfile(mask_file):
            raise FileNotFoundError(f"Mask/zones file not found or invalid: {mask_file}")

        if mode.lower() == "zones":
            zones = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
            if zones is not None:
                mask_qimg = numpy_to_qimage(zones)  # RGB or RGBA image
            else:
                print("Could not load zones image.")
        else:
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask_qimg = numpy_to_qimage(mask, force_grayscale=True)
            else:
                print("Could not load mask image.")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    default_thresholds = {
        "blur": 9,
        "erode": 1,
        "blur2": 1,
        "threshold": 50,
        "min_area": 100,
        "max_area": 20000,
        "hue_lo": 30,
        "hue_hi": 90,
        "sat_lo": 50,
        "sat_hi": 255,
        "val_lo": 50,
        "val_hi": 255
    }

    thresholds = {**default_thresholds, **(threshold_dict or {})}
    window = PyQt5ShapeDrawerWindow(file=media_arg, 
                                    background_file=background_file, 
                                    mask_qimg=mask_qimg, 
                                    threshold_dict=thresholds, 
                                    mode=mode, 
                                    total_frames=total_frames,
                                    width=width,
                                    height=height,
                                    roi=roi)
    window.show()
    window.raise_()          # Bring window to front
    window.activateWindow()  # Give it focus

    # Attach ID mapping to window for UI logic (optional, but handy for colored labels etc)
    if unique_idstrs:
        window.idnum_to_idstr = idnum_to_idstr
        window.idstr_to_idnum = idstr_to_idnum

    if points_by_frame:
        window.drawing_widget.points_by_frame = points_by_frame
        window.tp_total_ids = tp_total_ids
        window.input_num_ids.setValue(tp_total_ids)
        window.tp_current_id = 0
        window.current_id_box.setValue(1)
        window.current_id_box.setMinimum(1)
        window.current_id_box.setMaximum(tp_total_ids)

    # Set the starting mode 
    idx = window.opmode_combo.findText(mode.lower())
    if idx >= 0:
        window.opmode_combo.setCurrentIndex(idx)
        
    QTimer.singleShot(0, window.drawing_widget.setFocus)
    app.exec_()
    result = window.drawing_widget.final_output
    
    # Format point output (for video)
    if result is not None and result[0] == "point" and window.is_video:
        pts_dict = {}
        for frame, pt in window.drawing_widget.points_by_frame.items():
            pts_dict[frame] = (pt.x(), pt.y())
        result = ("point", pts_dict)

    # Format timepoints output (DataFrame)
    elif result is not None and result[0] == "timepoints":
        df_out = result[1]  # Already a DataFrame!
        if "angle" in df_out.columns:
            df_out["angle"] = pd.to_numeric(df_out["angle"], errors="coerce").round(1)
        #print("Saving DataFrame:\n", df_out)
        result = ("timepoints", df_out)
        if data_file:
            base, ext = os.path.splitext(os.path.expanduser(data_file))
            outpath = data_file
            if fileaction == "newfile":
                i = 2
                while os.path.exists(f"{base}{i}{ext}"):
                    i += 1
                outpath = f"{base}{i}{ext}"
            df_out.to_csv(outpath, index=False)
            print(f"Saved timepoints to: {outpath}")
        else:
            # Should never hit this since we set data_file above
            print("No data_file provided. Not saving results.")
        result = ""
    
    if result == "exit":
        return "exit"
    
    return result

def manual_tracker(media_file=None, background_file=None, mask_file=None, mode="timepoints",
                   threshold_dict={}, firstframe=1, lastframe=None,
                   fileaction="overwrite", data_file=None, width=1280, height=960):
           
    return annotation_gui(
        media_file=media_file,
        background_file=background_file,
        mask_file=mask_file,
        mode=mode,
        threshold_dict=threshold_dict,
        firstframe=firstframe,
        lastframe=lastframe,
        fileaction=fileaction,
        data_file=data_file,
        width=width,
        height=height
    )

if __name__ == '__main__':
    result = annotation_gui(
        media_file="~/Desktop/sample_video_col.mp4",
        background_file="~/Desktop/sample_video_bg.jpg",
        mode="timepoints",
        threshold_dict={'blur': 14, 'erode': 6, 'blur2': 6, 'threshold': 9, 'min_area': 475, 'max_area': 1748},
        data_file="/Users/Jolle/Desktop/annotated_timepoints.csv",
        fileaction="newfile"
    )
