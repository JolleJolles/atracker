#! /usr/bin/env python

import re
import cv2
import math
import signal
import colorsys
import subprocess
import threading
from typing import Callable

import numpy as np
from collections import defaultdict
from screeninfo import get_monitors

import pandas as pd
from PyQt5.QtGui import QImage, QColor
from PyQt5.QtCore import QPoint


def cvMatToQImage(frame):
    """Convert a BGR or grayscale OpenCV image to QImage (RGBA)."""
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
    """Convert a QImage to a NumPy array. Supports grayscale and RGBA."""
    qimg = qimg.convertToFormat(format)
    width = qimg.width()
    height = qimg.height()
    ptr = qimg.bits()
    ptr.setsize(qimg.byteCount())
    if format == QImage.Format_Grayscale8:
        arr = np.array(ptr).reshape(height, width)
        return arr
    else:
        arr = np.array(ptr).reshape(height, width, 4)
        if to_bgr:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        return arr


def numpy_to_qimage(arr, force_grayscale=False):
    """Convert a NumPy array (grayscale or BGR) to QImage."""
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


def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
        colors.append(QColor(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))
    return colors


def get_screen_dimensions():
    """Fallback method to get screen dimensions using system_profiler."""
    try:
        output = subprocess.check_output(["system_profiler", "SPDisplaysDataType"])
        output = output.decode("utf-8")
        for line in output.split("\n"):
            if "Resolution" in line:
                match = re.search(r"(\d+)\s*x\s*(\d+)", line)
                if match:
                    width, height = map(int, match.groups())
                    return (width, height)
        print("No resolution line found in system_profiler output.")
        return (None, None)
    except Exception as e:
        print(f"Error using fallback method: {e}")
        return (None, None)


def get_screen_resolution():
    """Try screeninfo first, fallback to system_profiler if necessary."""
    try:
        monitor = get_monitors()[0]
        return (monitor.width, monitor.height)
    except (ImportError, IndexError) as e:
        return get_screen_dimensions()


def create_hue_gradient(height=10):
    hue_gradient = np.zeros((height, 180, 3), dtype=np.uint8)
    hue_gradient[..., 0] = np.arange(0, 180)
    hue_gradient[..., 1] = 255
    hue_gradient[..., 2] = 255
    hue_gradient_bgr = cv2.cvtColor(hue_gradient, cv2.COLOR_HSV2BGR)
    return hue_gradient_bgr


def draw_arrow_head(painter, tip, size=10, angle_rad=0.0, angle_offset=math.pi / 8):
    """Draw two lines from 'tip' at ±angle_offset from 'angle_rad'."""
    for offset in [+angle_offset, -angle_offset]:
        x = tip.x() - int(size * math.cos(angle_rad + offset))
        y = tip.y() - int(size * math.sin(angle_rad + offset))
        painter.drawLine(tip, QPoint(x, y))


def build_points_by_frame(df):
    """
    Converts standardized tracking DataFrame into points_by_frame dict:
    {id: {'c': {frame: QPoint}, 'h': {...}, ...}}
    Always uses df['ID'] as the ID key (int, 0-based).
    """
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


def start_powermate_listener(
    vendor_id: int = 1917,
    product_id: int = 1040,
    update_key_function: Callable[[int], None] = None,
    running_flag: threading.Event = None,
    leftkey: str = "e",
    rightkey: str = "r",
    pressedleftkey: str = "w",
    pressedrightkey: str = "t"
) -> threading.Thread:
    """Starts a thread to listen for PowerMate input and updates the key value."""
    import hid
    import time

    def powermate_loop():
        try:
            device = hid.Device(vendor_id, product_id)
            print("PowerMate connected!")
            pressed = False
            while running_flag.is_set():
                try:
                    data = device.read(64)
                    if data:
                        press_noticed = data[0] == 0x01
                        rotation = data[1]
                        if press_noticed:
                            pressed = not pressed
                        if pressed and rotation == 0xff:
                            key = pressedleftkey
                        if pressed and rotation == 0x01:
                            key = pressedrightkey
                        if not pressed and rotation == 0xff:
                            key = leftkey
                        if not pressed and rotation == 0x01:
                            key = rightkey
                        update_key_function(ord(key))
                except OSError:
                    print(f"Read error")
                    if not running_flag.is_set():
                        break
                time.sleep(0.01)
        except Exception as e:
            print(f"Error in PowerMate loop: {e}")
        finally:
            if 'device' in locals():
                device.close()
                print("PowerMate disconnected.")

    listener_thread = threading.Thread(target=powermate_loop, daemon=True)
    listener_thread.start()
    return listener_thread


def stop_powermate_listener(listener_thread: threading.Thread, running_flag: threading.Event):
    """Stops the PowerMate listener thread."""
    print("Stopping PowerMate listener...")
    running_flag.clear()
    if listener_thread and listener_thread.is_alive():
        listener_thread.join()
    print("PowerMate listener stopped.")
