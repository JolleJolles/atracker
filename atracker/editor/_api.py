#! /usr/bin/env python

import os
import sys
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer, QPoint
from PyQt5.QtGui import QImage

from ._utils import build_points_by_frame, numpy_to_qimage
from ._window import PyQt5ShapeDrawerWindow
from atracker.helpers.data import load_and_convert_tracking_dataframe

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

def annotation_gui(data_file=None, media_file=None, background_file=None, mask_file=None, mode="default",
                   threshold_dict={}, firstframe=None, lastframe=None, fileaction="overwrite",
                   width=1280, height=960, roi=None, start_fullscreen=False, _state=None):
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
    window._set_timeline_limits({"frame_start": firstframe, "frame_stop": lastframe})
    window.show()
    if start_fullscreen:
        window.showFullScreen()
        window.was_fullscreen = True
    window.raise_()
    window.activateWindow()

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
    if _state is not None:
        _state["was_fullscreen"] = window.was_fullscreen
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


def editor_gui(file_infos, purpose="mask", save_callback=None):
    """
    Launch the multi-file interactive editor.

    Parameters
    ----------
    file_infos : list[dict]
        List of dicts with keys: video_path, background_path, mask_path,
        zones_path, roi, frame_start, frame_stop, tracked_csv,
        threshold_dict, ind, video_name, dirs.
    purpose : str
        Initial purpose: "mask", "roi", "zones", "framelimits",
        "timepoints", "measure", "thresholding". Default "mask".
    save_callback : callable | None
        Called as save_callback(file_idx, ind, purpose, data) when the
        user clicks Store.
    """
    from ._window import PyQt5ShapeDrawerWindow, _PURPOSE_MAP, _MULTI_FILE_PURPOSES

    if not file_infos:
        return

    default_thresholds = {
        "blur": 9, "erode": 1, "blur2": 1, "threshold": 50,
        "min_area": 100, "max_area": 20000,
        "hue_lo": 30, "hue_hi": 90, "sat_lo": 50, "sat_hi": 255, "val_lo": 50, "val_hi": 255,
    }

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")

    fi = file_infos[0]
    thresholds = {**default_thresholds, **(fi.get("threshold_dict") or {})}

    window = PyQt5ShapeDrawerWindow(
        file_infos=file_infos,
        file_idx=0,
        save_callback=save_callback,
        threshold_dict=thresholds,
    )

    # Set initial purpose in the combo
    purpose_label_map = {
        "mask": "Mask",
        "roi": "ROI",
        "zones": "Zones",
        "framelimits": "Frame limits",
        "timepoints": "Coordinate data",
        "measure": "Measurement",
        "thresholding": "Thresholding",
    }
    target_label = purpose_label_map.get(purpose.lower(), purpose.capitalize())
    idx = window.opmode_combo.findText(target_label)
    if idx < 0:
        # Fallback: case-insensitive search
        for i in range(window.opmode_combo.count()):
            if window.opmode_combo.itemText(i).lower() == purpose.lower():
                idx = i
                break
    if idx >= 0:
        window.opmode_combo.setCurrentIndex(idx)

    # Auto-load data for the initial file and purpose
    window._auto_load_purpose_data(fi, window.opmode_combo.currentText())

    window.show()
    window.raise_()
    window.activateWindow()
    QTimer.singleShot(0, window.drawing_widget.setFocus)
    app.exec_()
    return window.drawing_widget.final_output

if __name__ == '__main__':
    result = annotation_gui(
        media_file="~/Desktop/sample_video_col.mp4",
        background_file="~/Desktop/sample_video_bg.jpg",
        mode="timepoints",
        threshold_dict={'blur': 14, 'erode': 6, 'blur2': 6, 'threshold': 9, 'min_area': 475, 'max_area': 1748},
        data_file="/Users/Jolle/Desktop/annotated_timepoints.csv",
        fileaction="newfile"
    )
