#! /usr/bin/env python

import os
import sys
import math
import time
import cv2
import colorsys
import numpy as np
import pandas as pd
from collections import defaultdict
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QHBoxLayout, QVBoxLayout,
    QButtonGroup, QGridLayout, QGroupBox, QComboBox, QCheckBox, QPushButton,
    QSlider, QSpinBox, QRadioButton, QScrollArea,
)
from PyQt5.QtGui import QPainter, QPen, QColor, QPolygon, QImage, QPixmap, QBrush
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal, QTimer


# ---------- Image / Qt Utilities ----------

def cvMatToQImage(frame):
    if len(frame.shape) == 2:
        h, w = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGBA)
    elif len(frame.shape) == 3:
        h, w, ch = frame.shape
        if ch == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        elif ch == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
    else:
        raise ValueError("Unsupported frame format for QImage conversion.")
    return QImage(frame.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()


def qimage_to_numpy(qimg, format=QImage.Format_RGBA8888, to_bgr=True):
    qimg = qimg.convertToFormat(format)
    width = qimg.width()
    height = qimg.height()
    ptr = qimg.bits()
    ptr.setsize(qimg.byteCount())
    if format == QImage.Format_Grayscale8:
        return np.array(ptr).reshape(height, width)
    arr = np.array(ptr).reshape(height, width, 4)
    if to_bgr:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return arr


def numpy_to_qimage(arr, force_grayscale=False):
    if force_grayscale:
        if len(arr.shape) == 3 and arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        if len(arr.shape) != 2:
            raise ValueError("Expected a 2D array for grayscale.")
        h, w = arr.shape
        return QImage(arr.data, w, h, w, QImage.Format_Grayscale8).copy()
    if len(arr.shape) == 2:
        h, w = arr.shape
        return QImage(arr.data, w, h, w, QImage.Format_Grayscale8).copy()
    elif len(arr.shape) == 3 and arr.shape[2] == 3:
        arr_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        h, w, ch = arr_rgb.shape
        return QImage(arr_rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    else:
        raise ValueError("Unsupported array shape for QImage conversion.")


def qimg_to_grayscale(qimg):
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


def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
        colors.append(QColor(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))
    return colors


def draw_arrow_head(painter, tip, size=10, angle_rad=0.0, angle_offset=math.pi / 8):
    for offset in [+angle_offset, -angle_offset]:
        x = tip.x() - int(size * math.cos(angle_rad + offset))
        y = tip.y() - int(size * math.sin(angle_rad + offset))
        painter.drawLine(tip, QPoint(x, y))


def build_points_by_frame(df):
    frame_data = defaultdict(lambda: {'c': {}, 'h': {}, 't': {}, 'a': {}})
    for _, row in df.iterrows():
        id_val = int(row['ID'])
        frame = int(row['frame'])
        if pd.notnull(row.get('cx')) and pd.notnull(row.get('cy')):
            frame_data[id_val]['c'][frame] = QPoint(int(row['cx']), int(row['cy']))
        if pd.notnull(row.get('hx')) and pd.notnull(row.get('hy')):
            frame_data[id_val]['h'][frame] = QPoint(int(row['hx']), int(row['hy']))
        if pd.notnull(row.get('tx')) and pd.notnull(row.get('ty')):
            frame_data[id_val]['t'][frame] = QPoint(int(row['tx']), int(row['ty']))
        if pd.notnull(row.get('angle')):
            frame_data[id_val]['a'][frame] = float(row['angle'])
    return frame_data
