"""Load saved data for one editor purpose without managing window navigation."""
import os

import cv2
import pandas as pd
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QInputDialog

from atracker.helpers.data import load_and_convert_tracking_dataframe


def load_purpose_data(window, fi, opmode):
    if opmode == "mask":
        window.drawing_widget.mask_image = QImage(window.orig_width, window.orig_height, QImage.Format_Grayscale8)
        window.drawing_widget.mask_image.fill(255)
    elif opmode == "zones":
        window.drawing_widget.zones_overlay = None
    elif opmode == "timepoints":
        window.drawing_widget.points_by_frame = {}
        window.input_num_ids.setValue(1)
        fi["id_labels"] = {}
    if opmode == "mask":
        mask_path = fi.get("mask_path")
        if mask_path and os.path.isfile(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                from ._utils import numpy_to_qimage
                window.drawing_widget.mask_image = numpy_to_qimage(mask, force_grayscale=True)
    elif opmode == "zones":
        zones_path = fi.get("zones_path")
        if zones_path and os.path.isfile(zones_path):
            zones = cv2.imread(zones_path, cv2.IMREAD_UNCHANGED)
            if zones is not None:
                from ._utils import numpy_to_qimage
                window.drawing_widget.zones_overlay = numpy_to_qimage(zones).convertToFormat(QImage.Format_ARGB32)
    elif opmode == "roi":
        roi = fi.get("roi")
        if roi:
            (x0, y0), (x1, y1) = roi
            window.drawing_widget.shapes = [("rectangle", ((x0, y0), (x1, y0), (x1, y1), (x0, y1)))]
    elif opmode == "framelimits":
        start = fi.get("frame_start")
        stop = fi.get("frame_stop")
        if start:
            window.flim_start_spin.setValue(int(start))
        if stop:
            window.flim_stop_spin.setValue(int(stop))
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
                window.drawing_widget.points_by_frame = pbf
                if pbf:
                    window.input_num_ids.setValue(len(pbf))
                    window.current_id_box.setValue(1)
                    window.updateCurrentID(1)
                    window.drawing_widget.tp_current_id = 0
    elif opmode in ("thresholding", "thresholding color"):
        options = fi.get("threshold_options", {})
        selected = window._threshold_selection.get(opmode)
        if selected in options:
            pass
        elif len(options) > 1:
            selected, ok = QInputDialog.getItem(self, "Threshold configuration",
                "Configuration to edit:", list(options), 0, False)
            if not ok:
                selected = None
        elif options:
            selected = next(iter(options))
        else:
            selected = None
        fi["threshold_type"] = selected
        if selected is not None:
            window._threshold_selection[opmode] = selected
        params = options.get(selected, fi.get("threshold_dict", {}))
        # New configurations must start from defaults, not the previous video's settings.
        defaults = {"blur": 9, "erode": 1, "blur2": 1, "threshold": 50,
            "min_area": 100, "max_area": 20000, "gamma": 1.0,
            "hue_lo": 30, "hue_hi": 90, "sat_lo": 50, "sat_hi": 255,
            "val_lo": 50, "val_hi": 255}
        params = window._threshold_drafts.get(selected, {**defaults, **params})
        params = {**fi.get("threshold_defaults", {}), **params}
        window._restore_state({"thresh_params": params})
    window.drawing_widget.update()
