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

from .track import Tracker
from .helpers.media import bg_extract
from .process import Processor
from .editor import annotation_gui


def _make_config(advanced=False, show_tracking=False, create_vid=True, overwrite=True):
    """Build a minimal Box config compatible with Tracker."""
    return Box({
        "track": Box({
            "advanced": advanced,
            "orientfrombw": False,
            "create_vid": create_vid,
            "create_dat": True,
            "overwrite": overwrite,
            "link_dist": 100,
            "merge_dist": 20,
            "size_filter": False,
            "size_filter_tol": 0.25,
            "size_filter_memory": 500,
            "check_flicker": False,
            "skip_frames": 0,
            "max_framedist": 200,
            "track_merges": False,
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
    })


def _atrk_path(path):
    """Normalise a user-supplied path to a canonical .atrk path."""
    # Strip numpy's auto-appended .npz if present
    if path.endswith(".npz"):
        path = path[:-4]
    if not path.endswith(".atrk"):
        path += ".atrk"
    return path


def save_atcache(path, img_bg, img_mask, threshinfo, roi):
    """
    Save tracking setup to a single .atrk file.

    Stores background image, mask image, threshold settings, and ROI in one
    compressed file. Load it later with load_atcache() or pass the path to
    track_video(load_settings=...) to skip the setup GUIs.

    Parameters
    ----------
    path : str
        Output path. The .atrk extension is appended if missing.
    img_bg : np.ndarray
        Background image (BGR uint8).
    img_mask : np.ndarray or None
        Mask image (grayscale uint8), or None if no mask.
    threshinfo : dict
        Threshold settings dict (as returned by the threshold GUI).
    roi : tuple
        ((x1, y1), (x2, y2)) region-of-interest coordinates.
    """
    path = _atrk_path(path)
    mask_arr = img_mask if img_mask is not None else np.array([], dtype=np.uint8)
    np.savez_compressed(
        path,
        bg=img_bg,
        mask=mask_arr,
        threshinfo=np.frombuffer(json.dumps(threshinfo).encode(), dtype=np.uint8),
        roi=np.frombuffer(json.dumps([list(roi[0]), list(roi[1])]).encode(), dtype=np.uint8),
    )
    # numpy appends .npz automatically; rename back to the clean .atrk path
    if os.path.exists(path + ".npz") and not os.path.exists(path):
        os.rename(path + ".npz", path)
    lineprint(f"Settings saved: {path}")


def load_atcache(path):
    """
    Load tracking setup from a .atrk file.

    Parameters
    ----------
    path : str
        Path to the .atrk file (or legacy .atcache path).

    Returns
    -------
    dict with keys: img_bg, img_mask, threshinfo, roi
        img_bg    : np.ndarray (BGR uint8)
        img_mask  : np.ndarray (grayscale uint8) or None
        threshinfo: dict
        roi       : ((x1, y1), (x2, y2)) or None
    """
    path = _atrk_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Settings file not found: {path}")
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
    save_settings : bool or str, default False
        Save background, mask, thresholds, and ROI to a .atrk file after
        setup. True saves alongside the video as {video_base}.atrk; a string
        is treated as an explicit output path.
    load_settings : bool or str, default False
        Load a previously saved .atrk file to skip setup GUIs. True looks for
        {video_base}.atrk next to the video; a string is an explicit path.
        Individual GUI steps (draw_roi, draw_mask, set_threshold) can still be
        set True to override specific loaded values.

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

    # Resolve True/False for save_settings / load_settings to actual paths
    default_atrk = os.path.join(video_dir, video_base + ".atrk")
    if save_settings is True:
        save_settings = default_atrk
    if load_settings is True:
        load_settings = default_atrk

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
        if method != "bw":
            thresh_types = [method]
        elif threshinfo:
            # Derive thresh type from saved threshinfo keys rather than defaulting to "bw"
            thresh_types = list(threshinfo.keys())
        else:
            thresh_types = ["bw"]

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
            advanced=not simple,
            show_tracking=show_tracking,
            create_vid=create_vid,
            overwrite=overwrite,
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


def process_video(
    csvfile,
    videofile=None,
    outdir=None,
    fps=25,
    roi=None,
    conv=1,
    maskfile=None,
    wallfile=None,
    zonefile=None,
    orientfrombw=False,
    overwrite=False,
    fulldata=False,
    convert=True,
    changefps=None,
    mask_margin=15,
    max_traj_gap=50,
    min_traj_len=10,
    roi_edge_margin=10,
    interp_gap_com=500,
    interp_gap_orient=100,
    smoothwin=10,
    orient_min_speed=1,
    interpolate=True,
    compute_movement=True,
    compute_distances=True,
):
    """
    Post-process a single tracked CSV file without an ATracker folder setup.

    Runs the same pipeline as AT.process() but takes explicit paths instead of
    relying on an overview file.  The output is written to the same directory as
    the CSV (or *outdir* if specified) with a ``_F.csv`` suffix.

    Parameters
    ----------
    csvfile : str
        Path to the tracked CSV produced by track_video() or AT.track().
    videofile : str or None
        Original video file. When provided, fps, resolution, and frame count are
        read from it; roi defaults to the full frame.
    outdir : str or None
        Directory for the output CSV. Defaults to the directory of *csvfile*.
    fps : float, default 25
        Frame rate used for speed calculations. Ignored when *videofile* is given.
    roi : tuple or None
        ``((x1, y1), (x2, y2))`` region of interest. Defaults to the full frame
        when *videofile* is given, or is inferred from coordinate ranges otherwise.
    conv : float, default 1
        Pixel-to-real-world conversion factor (mm/pixel). 1 = no conversion.
    maskfile : str or None
        Path to a mask image (exclusion zone).
    wallfile : str or None
        Path to a wall image (physical boundary for distance calculations).
    zonefile : str or None
        Path to a zone image (named areas for distance calculations).
    orientfrombw : bool, default False
        Use the bw tracking angle column as orientation instead of head/tail coords.
    overwrite : bool, default False
        Overwrite an existing processed file.
    fulldata : bool, default False
        Extend output to every frame in the video window (needs *videofile* or a
        reliable frame count). When False, output covers only detected frames.
    convert : bool, default True
        Apply the *conv* factor to convert pixel coordinates.
    changefps : int or None
        Resample output to a lower frame rate.
    mask_margin : int, default 15
        Pixels from the mask boundary: detections within are removed, gaps are
        not interpolated.
    max_traj_gap : int, default 50
        Frame gap above which a break starts a new trajectory.
    min_traj_len : int, default 10
        Minimum trajectory length in frames; shorter bursts are discarded.
    roi_edge_margin : int, default 10
        Pixels from the ROI edge: head/tail blanked; ROI exits not interpolated.
    interp_gap_com : int, default 500
        Maximum gap (frames) to interpolate centroid data over.
    interp_gap_orient : int, default 100
        Maximum gap (frames) to interpolate head/tail and orientation over.
    smoothwin : int, default 10
        Savitzky–Golay smoothing window in frames (1 = no smoothing).
    orient_min_speed : float, default 1
        Minimum speed above which heading is used as fallback for orientation.
    interpolate : bool, default True
        Interpolate gaps in centroid, head/tail, and orientation data.
    compute_movement : bool, default True
        Compute movement variables: displacement, speed, heading, turn rates.
    compute_distances : bool, default True
        Compute distance measures: ROI edge, mask, walls, zones.

    Returns
    -------
    str
        Path to the output ``_F.csv`` file.
    """
    csvfile = os.path.abspath(csvfile)
    if not os.path.exists(csvfile):
        raise FileNotFoundError(f"CSV not found: {csvfile}")

    csv_dir = os.path.dirname(csvfile)
    csv_base = os.path.splitext(os.path.basename(csvfile))[0]
    out_dir = os.path.abspath(outdir) if outdir else csv_dir

    # --- Video metadata ---
    fcount = 99999
    resolution = None
    if videofile is not None:
        videofile = os.path.abspath(videofile)
        _fps, width, height, fcount = get_vid_params(videofile)
        fps = _fps if _fps and _fps > 0 else fps
        resolution = (width, height)
        if roi is None:
            roi = ((0, 0), (width, height))

    # Infer ROI from data if still unknown
    if roi is None:
        _data = pd.read_csv(csvfile)
        cx_max = _data["cx"].max() if "cx" in _data and _data["cx"].notna().any() else 1000
        cy_max = _data["cy"].max() if "cy" in _data and _data["cy"].notna().any() else 1000
        roi = ((0, 0), (int(cx_max * 1.1) + 1, int(cy_max * 1.1) + 1))

    if resolution is None:
        resolution = roi[1]

    # --- Build minimal overview row ---
    # Store image paths as absolute paths — valid_img_path uses os.path.join which
    # ignores the originals_dir prefix when the stored value is already absolute.
    overview = pd.DataFrame([{
        "video": csv_base,
        "fps": fps,
        "fcount": fcount,
        "resolution": str(resolution),
        "frame_start": np.nan,
        "frame_stop": np.nan,
        "roi": str(roi),
        "conv": conv,
        "maskimg": os.path.abspath(maskfile) if maskfile else np.nan,
        "wallimg": os.path.abspath(wallfile) if wallfile else np.nan,
        "zoneimg": os.path.abspath(zonefile) if zonefile else np.nan,
        "bgimg": np.nan,
        "ID": np.nan,
        "objects": 1,
        "exclude": np.nan,
    }], index=[0])
    overview["roi"] = [roi]  # Processor reads roi as a pre-parsed tuple

    dirs = {
        "originals": csv_dir,  # unused — image paths are absolute
        "processed": out_dir,
        "tracked": csv_dir,
        "temp": csv_dir,
        "todo": csv_dir,
    }

    config = _make_config()

    lineprint(f"Processing {os.path.basename(csvfile)}..")

    P = Processor(
        dirs=dirs,
        config=config,
        overview=overview,
        trackedfiles=[csvfile],
        orientfrombw=orientfrombw,
        overwrite=overwrite,
        fulldata=fulldata,
        convert=convert,
        changefps=changefps,
        mask_margin=mask_margin,
        max_traj_gap=max_traj_gap,
        min_traj_len=min_traj_len,
        roi_edge_margin=roi_edge_margin,
        interp_gap_com=interp_gap_com,
        interp_gap_orient=interp_gap_orient,
        smoothwin=smoothwin,
        orient_min_speed=orient_min_speed,
        interpolate=interpolate,
        compute_movement=compute_movement,
        compute_distances=compute_distances,
    )
    P.setup(csvfile, None)

    out_csv = os.path.join(out_dir, csv_base + "_F.csv")
    if os.path.exists(out_csv):
        lineprint(f"Processing complete → {out_csv}")
    else:
        lineprint("Processing finished (no output produced).")
    return out_csv
