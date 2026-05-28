#! /usr/bin/env python

import os
import cv2
import json
import shutil
import tempfile
import numpy as np
import pandas as pd

from box import Box
from pythutils.mediautils import get_vid_params
from pythutils.sysutils import lineprint

from .tracker import Tracker
from .media import bg_extract
from .visual_editor import annotation_gui


def _make_config(fps=25, simple=True, show_tracking=False, create_vid=True,
                 overwrite=True, frame_start=1, frame_stop=99999):
    """Build a minimal Box config compatible with Tracker."""
    return Box({
        "exp": Box({"fps": fps, "realdims": None}),
        "track": Box({
            "startframe": frame_start,
            "stopframe": frame_stop,
            "keep_frames": 10,
            "simple": simple,
            "orientfrombw": False,
            "create_vid": create_vid,
            "create_dat": True,
            "overwrite": overwrite,
            "strict": False,
            "linkdisthreshold": 100,
            "mergedmindist": 20,
            "contour_mode": "static",
            "shape_area_tol": 0.25,
            "shape_history_len": 500,
        }),
        "vis": Box({
            "show_tracking": show_tracking,
            "vid_displaysize": 1,
            "frame_disstep": 100,
            "traj_length": 4,
            "waitkey": 1,
            "idcol": True,
            "contour_col": "(255, 0, 0)",
            "centre_col": "(255, 255, 255)",
            "front_col": "(0, 0, 0)",
            "orient_col": "(0, 0, 0)",
            "traj_col": "(0, 255, 255)",
            "centre_lwidth": 13,
            "orient_lwidth": 2,
            "orient_tip": 0.15,
            "orient_length": 15,
            "traj_minthick": 6.4,
            "traj_maxthick": 9,
            "traj_opacity": 0.5,
            "mask_opacity": 0.15,
            "box_opacity": 0.7,
            "contnrs": False,
            "trajs_below": False,
        }),
        "orient": Box({"delwindow": 10}),
        "bgextract": Box({"bg_frames": 25}),
    })


def save_atcache(path, img_bg, img_mask, threshinfo, roi):
    """
    Save tracking setup to a single .atcache file.

    Stores background image, mask image, threshold settings, and ROI in one
    compressed file. Load it later with load_atcache() or pass the path to
    track_video(load_settings=...) to skip the setup GUIs.

    Parameters
    ----------
    path : str
        Output path. The .atcache extension is appended if missing.
    img_bg : np.ndarray
        Background image (BGR uint8).
    img_mask : np.ndarray or None
        Mask image (grayscale uint8), or None if no mask.
    threshinfo : dict
        Threshold settings dict (as returned by the threshold GUI).
    roi : tuple
        ((x1, y1), (x2, y2)) region-of-interest coordinates.
    """
    if not path.endswith(".atcache"):
        path += ".atcache"
    mask_arr = img_mask if img_mask is not None else np.array([], dtype=np.uint8)
    np.savez_compressed(
        path,
        bg=img_bg,
        mask=mask_arr,
        threshinfo=np.frombuffer(json.dumps(threshinfo).encode(), dtype=np.uint8),
        roi=np.frombuffer(json.dumps([list(roi[0]), list(roi[1])]).encode(), dtype=np.uint8),
    )
    lineprint(f"Settings saved: {path}")


def load_atcache(path):
    """
    Load tracking setup from a .atcache file.

    Parameters
    ----------
    path : str
        Path to the .atcache file.

    Returns
    -------
    dict with keys: img_bg, img_mask, threshinfo, roi
        img_bg    : np.ndarray (BGR uint8)
        img_mask  : np.ndarray (grayscale uint8) or None
        threshinfo: dict
        roi       : ((x1, y1), (x2, y2)) or None
    """
    if not path.endswith(".atcache"):
        path += ".atcache"
    data = np.load(path)
    img_mask = data["mask"] if data["mask"].size > 0 else None
    threshinfo = json.loads(data["threshinfo"].tobytes().decode())
    roi_raw = json.loads(data["roi"].tobytes().decode())
    roi = (tuple(roi_raw[0]), tuple(roi_raw[1])) if roi_raw else None
    return {
        "img_bg": data["bg"],
        "img_mask": img_mask,
        "threshinfo": threshinfo,
        "roi": roi,
    }


def track_video(
    video_path,
    method="bw",
    n_objects=1,
    draw_roi=False,
    draw_mask=False,
    set_threshold=True,
    simple=True,
    show_tracking=False,
    create_vid=True,
    output=None,
    overwrite=True,
    frame_start=None,
    frame_stop=None,
    fps=None,
    save_settings=None,
    load_settings=None,
):
    """
    Track a single video without any folder setup or overview file.

    Outputs only a CSV (and optional _TR.mp4) in the same directory as the
    video. All intermediate files (background, mask, thresholds) are kept
    in memory and a temporary directory — nothing else is written to disk
    unless save_settings is given.

    Parameters
    ----------
    video_path : str
        Path to the video file to track.
    method : str, default "bw"
        Detection method. "bw" uses background subtraction.
        Any other string (e.g. "red", "blue") uses HSV colour thresholding.
    n_objects : int, default 1
        Number of objects to track.
    draw_roi : bool, default False
        Open GUI to draw the region of interest. If False, the full frame is used.
    draw_mask : bool, default False
        Open GUI to draw an exclusion mask (e.g. a shelter or refuge area).
    set_threshold : bool, default True
        Open threshold calibration GUI on random frames.
    simple : bool, default True
        If True, extract centroid only. If False, extract full shape data
        (head, tail, skeleton, orientation, curvature).
    show_tracking : bool, default False
        Show live tracking window while processing.
    create_vid : bool, default True
        Write a _TR.mp4 tracking overlay video alongside the output CSV.
    output : str or None, default None
        Directory for the output CSV and video. Defaults to the video's directory.
    overwrite : bool, default True
        If False and a CSV already exists, skip tracking.
    frame_start : int or None
        First frame to track. Defaults to 1.
    frame_stop : int or None
        Last frame to track. Defaults to end of video.
    fps : int or None
        Override the video's stored frame rate.
    save_settings : str or None
        If given, saves background image, mask, thresholds, and ROI to this
        path as a single .atcache file. Useful for re-tracking with the same
        setup without repeating the GUI steps.
    load_settings : str or None
        Path to a .atcache file previously created by save_settings. Skips
        background extraction and GUI steps for any settings found in the file.
        Individual steps (draw_roi, draw_mask, set_threshold) can still be set
        True to override loaded values.

    Returns
    -------
    str
        Path to the output CSV file.
    """
    video_path = os.path.abspath(video_path)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    video_dir = os.path.dirname(video_path)
    video_name = os.path.basename(video_path)
    video_base = os.path.splitext(video_name)[0]
    output_dir = os.path.abspath(output) if output else video_dir

    # --- Video parameters ---
    _fps, width, height, fcount = get_vid_params(video_path)
    vid_fps = fps or _fps
    fs = frame_start or 1
    fe = frame_stop or int(fcount)

    # --- Load settings bundle if provided ---
    loaded = {}
    if load_settings:
        loaded = load_atcache(load_settings)
        lineprint(f"Loaded settings from {load_settings}")

    # --- Background (in memory) ---
    img_bg = loaded.get("img_bg")
    if img_bg is None:
        lineprint("Extracting background..", end=" ")
        img_bg = bg_extract(video_path, start=fs, stop=fe, framenr=25)
        if img_bg is None:
            raise RuntimeError("Background extraction failed.")
        print("done")

    # --- All temp files go in one directory; cleaned up unconditionally ---
    tmp_dir = tempfile.mkdtemp(prefix=".attracker_", dir=video_dir)
    work_dir = os.path.join(tmp_dir, "_work")
    os.makedirs(work_dir)

    try:
        # Write background to temp dir so annotation_gui and Tracker can read it
        bg_name = f"{video_base}_bg.jpg"
        bg_path = os.path.join(tmp_dir, bg_name)
        cv2.imwrite(bg_path, img_bg)

        # --- ROI ---
        loaded_roi = loaded.get("roi")
        if draw_roi or loaded_roi is None:
            if draw_roi:
                lineprint("Draw ROI in the GUI.")
                result = annotation_gui(
                    media_file=video_path, background_file=bg_path,
                    mode="roi", firstframe=fs, lastframe=fe,
                )
                if result is None or result == "exit":
                    raise RuntimeError("ROI drawing cancelled.")
                pt1, pt2 = result[1]
            else:
                pt1 = (0, 0)
                pt2 = (width, height)
        else:
            pt1, pt2 = loaded_roi
        roi_str = str((pt1, pt2))

        # --- Mask (in memory) ---
        img_mask = loaded.get("img_mask")
        if draw_mask:
            lineprint("Draw exclusion mask in the GUI.")
            result = annotation_gui(
                media_file=video_path, background_file=bg_path,
                mode="mask", firstframe=fs, lastframe=fe, roi=(pt1, pt2),
            )
            if result is not None and result != "exit" and isinstance(result[1], np.ndarray):
                img_mask = result[1]
            else:
                lineprint("No mask drawn.")

        # Write mask to temp dir if we have one
        mask_name = None
        if img_mask is not None:
            mask_name = f"{video_base}_mask.jpg"
            cv2.imwrite(os.path.join(tmp_dir, mask_name), img_mask)

        # --- Thresholds (in memory) ---
        threshinfo = loaded.get("threshinfo") or {}
        thresh_types = [method] if method != "bw" else ["bw"]

        if set_threshold:
            for ttype in thresh_types:
                mode = "thresholding" if ttype.startswith("bw") else "thresholding color"
                lineprint(f"Calibrate threshold for '{ttype}' — close window when done.")
                result = annotation_gui(
                    media_file=video_path, background_file=bg_path,
                    mode=mode,
                    threshold_dict=threshinfo.get(ttype, {}),
                    firstframe=fs, lastframe=fe,
                )
                if result is not None and result != "exit" and isinstance(result[1], dict) and result[1]:
                    threshinfo[ttype] = result[1]
                else:
                    lineprint(f"No threshold set for '{ttype}', using defaults.")

        # --- Optionally save settings bundle ---
        if save_settings:
            save_atcache(save_settings, img_bg, img_mask, threshinfo, (pt1, pt2))

        # --- Copy video into temp dir (original is never moved) ---
        video_copy = os.path.join(tmp_dir, video_name)
        shutil.copy2(video_path, video_copy)

        # --- Build minimal dirs ---
        dirs = {
            "originals": tmp_dir,   # video copy + bg + mask live here
            "temp": work_dir,        # video is moved here during active tracking
            "tracked": output_dir,   # CSV and TR.mp4 go here
        }

        # --- Build minimal overview row ---
        overview = pd.DataFrame([{
            "video": video_base,
            "fps": vid_fps,
            "fcount": fcount,
            "resolution": f"{width}x{height}",
            "frame_start": fs,
            "frame_stop": fe,
            "roi": roi_str,
            "conv": np.nan,
            "bgimg": bg_name,
            "maskimg": mask_name if mask_name else np.nan,
            "thresh_types": ",".join(thresh_types),
            "objects": n_objects,
            "exclude": np.nan,
        }], index=[0])

        # --- Config ---
        config = _make_config(
            fps=vid_fps, simple=simple, show_tracking=show_tracking,
            create_vid=create_vid, overwrite=overwrite,
            frame_start=fs, frame_stop=fe,
        )

        # --- Run tracking ---
        T = Tracker(
            pools=1,
            inds=[0],
            trackfiles=[video_copy],
            dirs=dirs,
            overview=overview,
            config=config,
            threshinfo=threshinfo,
            start=frame_start,
            stop=frame_stop,
            threshtype=",".join(thresh_types),
            objects=n_objects,
            checkconschange=False,
            suffix="",
            overwrite=overwrite,
        )
        T.setuptracking(0, video_copy)

    finally:
        # Temp dir cleanup — rescue any video copy that may have got stuck
        if os.path.exists(work_dir):
            for f in os.listdir(work_dir):
                src = os.path.join(work_dir, f)
                dst = os.path.join(tmp_dir, f)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    output_csv = os.path.join(output_dir, video_base + ".csv")
    if os.path.exists(output_csv):
        lineprint(f"Tracking complete → {output_csv}")
    else:
        lineprint("Tracking finished (no CSV produced — check threshold settings).")
    return output_csv
