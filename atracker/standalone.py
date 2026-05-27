#! /usr/bin/env python

import os
import cv2
import shutil
import tempfile
import numpy as np
import pandas as pd
import yaml

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
):
    """
    Track a single video without any folder setup or overview file.

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
        Open GUI to calibrate detection threshold on random frames. If False,
        any previously saved threshold settings alongside the video are reused,
        or sensible defaults are used.
    simple : bool, default True
        If True, extract centroid only. If False, extract full shape data
        (head, tail, skeleton, orientation, curvature).
    show_tracking : bool, default False
        Show live tracking window while processing.
    create_vid : bool, default True
        Write a tracking overlay video (_TR.mp4) alongside the output CSV.
    output : str or None, default None
        Directory where the output CSV (and optional video) are saved.
        Defaults to the same directory as the input video.
    overwrite : bool, default True
        If False and a CSV already exists, skip tracking.
    frame_start : int or None
        First frame to track. Defaults to 1.
    frame_stop : int or None
        Last frame to track. Defaults to end of video.
    fps : int or None
        Override the video's stored frame rate.

    Returns
    -------
    str
        Path to the output CSV file.
    """
    video_path = os.path.abspath(video_path)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    video_dir = os.path.dirname(video_path)
    video_base = os.path.splitext(os.path.basename(video_path))[0]

    output_dir = os.path.abspath(output) if output else video_dir

    # --- Get video parameters ---
    vid_params = get_vid_params(video_path)
    vid_fps = fps or vid_params.get("fps", 25)
    fcount = vid_params.get("fcount", 99999)
    width = vid_params.get("width", 0)
    height = vid_params.get("height", 0)

    fs = frame_start or 1
    fe = frame_stop or int(fcount)

    # --- Background image ---
    bg_name = f"{video_base}_bg.jpg"
    bg_path = os.path.join(video_dir, bg_name)
    if not os.path.exists(bg_path):
        lineprint("Extracting background image..")
        img_bg = bg_extract(video_path, start=fs, stop=fe, framenr=25)
        if img_bg is None:
            raise RuntimeError("Background extraction failed.")
        cv2.imwrite(bg_path, img_bg)
    else:
        lineprint(f"Background image found: {bg_name}")

    # --- ROI ---
    if draw_roi:
        lineprint("Draw ROI: click and drag to define the tracking region.")
        result = annotation_gui(media_file=video_path, background_file=bg_path, mode="roi",
                                firstframe=fs, lastframe=fe)
        if result is None or result == "exit":
            raise RuntimeError("ROI drawing cancelled.")
        roi_str = str(result[1])
        pt1, pt2 = result[1]
    else:
        pt1 = (0, 0)
        pt2 = (width, height)
        roi_str = str((pt1, pt2))

    # --- Mask ---
    mask_name = None
    if draw_mask:
        lineprint("Draw mask: mark areas to exclude from tracking.")
        result = annotation_gui(media_file=video_path, background_file=bg_path, mode="mask",
                                firstframe=fs, lastframe=fe, roi=roi_str)
        if result is not None and result != "exit" and isinstance(result[1], np.ndarray):
            mask_name = f"{video_base}_mask.jpg"
            cv2.imwrite(os.path.join(video_dir, mask_name), result[1])
            lineprint(f"Mask saved: {mask_name}")
        else:
            lineprint("No mask drawn.")

    # --- Thresholds ---
    thresh_types = [method] if method != "bw" else ["bw"]
    threshinfo_path = os.path.join(video_dir, f"{video_base}_threshinfo.yml")
    threshinfo = {}

    if os.path.exists(threshinfo_path):
        with open(threshinfo_path) as f:
            threshinfo = yaml.safe_load(f) or {}

    if set_threshold:
        for ttype in thresh_types:
            mode = "thresholding" if ttype.startswith("bw") else "thresholding color"
            lineprint(f"Set threshold for method '{ttype}'. Close window when satisfied.")
            result = annotation_gui(
                media_file=video_path,
                background_file=bg_path,
                mode=mode,
                threshold_dict=threshinfo.get(ttype, {}),
                firstframe=fs,
                lastframe=fe,
            )
            if result is not None and result != "exit" and isinstance(result[1], dict) and result[1]:
                threshinfo[ttype] = result[1]
                with open(threshinfo_path, "w") as f:
                    yaml.safe_dump(threshinfo, f, default_flow_style=False)
                lineprint(f"Threshold settings saved to {threshinfo_path}")
            else:
                lineprint(f"No threshold set for '{ttype}', using defaults.")
    elif not threshinfo:
        lineprint("No threshold file found and set_threshold=False; using empty defaults.")

    # --- Build minimal dirs ---
    tmp_dir = tempfile.mkdtemp(prefix="_atracker_", dir=video_dir)
    try:
        dirs = {
            "originals": video_dir,
            "temp": tmp_dir,
            "tracked": output_dir,
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
        }])
        overview.index = [0]

        # --- Config ---
        config = _make_config(
            fps=vid_fps,
            simple=simple,
            show_tracking=show_tracking,
            create_vid=create_vid,
            overwrite=overwrite,
            frame_start=fs,
            frame_stop=fe,
        )

        # --- Run tracking ---
        T = Tracker(
            pools=1,
            inds=[0],
            trackfiles=[video_path],
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
        T.setuptracking(0, video_path)

    finally:
        if os.path.exists(tmp_dir):
            # Move back any video file that may still be in temp (e.g. after a crash)
            for f in os.listdir(tmp_dir):
                src = os.path.join(tmp_dir, f)
                dst = os.path.join(video_dir, f)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
            shutil.rmtree(tmp_dir, ignore_errors=True)

    output_csv = os.path.join(output_dir, video_base + ".csv")
    if os.path.exists(output_csv):
        lineprint(f"Tracking complete. Output: {output_csv}")
    else:
        lineprint("Tracking finished (no CSV produced — check settings).")

    return output_csv
