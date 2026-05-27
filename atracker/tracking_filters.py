#! /usr/bin/env python

import cv2
import numpy as np
from collections import deque

from pythutils.mediautils import crop


def point_near_mask(x, y, mask, edge_margin=50):
    """Return True if (x, y) is within edge_margin pixels of any masked (dark) pixel."""
    xc = int(np.clip(x, 0, mask.shape[1] - 1))
    yc = int(np.clip(y, 0, mask.shape[0] - 1))
    y_min = max(0, yc - edge_margin)
    y_max = min(mask.shape[0], yc + edge_margin)
    x_min = max(0, xc - edge_margin)
    x_max = min(mask.shape[1], xc + edge_margin)
    return bool(np.any(mask[y_min:y_max, x_min:x_max] < 128))


def filter_tracking_jumps(ids, coms, frame_nr, last_valid, max_framedist=200, pr_comm="", mask=None):
    filtered_coms = []
    for idx, id in enumerate(ids):
        c = coms[idx]
        if c is None or not isinstance(c, (tuple, list)) or any([ci != ci for ci in c]):
            filtered_coms.append((np.nan, np.nan))
            continue

        cur_near_mask = mask is not None and point_near_mask(c[0], c[1], mask)

        if id in last_valid:
            prev_x, prev_y, prev_frame, prev_near_mask = last_valid[id]
            gap = frame_nr - prev_frame
            allowed_jump = max_framedist * gap  # scales linearly — no cap, so fish can always be re-acquired
            dist = np.linalg.norm([c[0] - prev_x, c[1] - prev_y])
            if dist > allowed_jump:
                if prev_near_mask and cur_near_mask:
                    pass  # accept: fish traversed the mask region
                else:
                    filtered_coms.append((np.nan, np.nan))
                    continue

        filtered_coms.append((c[0], c[1]))
        last_valid[id] = (c[0], c[1], frame_nr, cur_near_mask)
    return filtered_coms, last_valid


def filter_contour_shape(ids, areas, frame_nr, shape_history, min_history=10, area_tol=0.25, pr_comm=""):
    """
    Per-ID rolling area consistency filter.
    Rejects detections whose area falls outside [area_tol, 1/area_tol] × rolling median.
    E.g. area_tol=0.25 accepts areas between 25% and 400% of the median.
    Only activates once min_history accepted frames are available for that ID.
    Returns accept list (bool per ID). Does NOT update shape_history.
    """
    accept = []
    for i, id in enumerate(ids):
        area = areas[i]
        if id not in shape_history or len(shape_history[id]) < min_history:
            accept.append(True)
        else:
            med_area = np.median(list(shape_history[id]))
            ratio = area / med_area if med_area > 0 else 1.0
            accept.append(area_tol <= ratio <= (1.0 / area_tol))
    return accept


def update_shape_history(ids, areas, filtered_coms, shape_history, history_len=500):
    """Update shape_history with areas for IDs that passed all filters (non-NaN cx)."""
    for i, id in enumerate(ids):
        if np.isfinite(filtered_coms[i][0]):
            shape_history.setdefault(id, deque(maxlen=history_len)).append(areas[i])


def estimate_flicker_baseline(cap, img_bg, pt1, pt2, n_frames=100, multiplier=20):
    """Sample frames and compute median mean-diff as flicker baseline."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_inds = np.linspace(0, total - 1, n_frames, dtype=int)
    means = []
    for i in sample_inds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = crop(frame, pt1, pt2)
        diff = cv2.subtract(img_bg, frame)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        means.append(np.mean(gray))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    baseline = np.median(means)
    return baseline * multiplier


def check_threshtypes(thresh_types):
    if isinstance(thresh_types, (np.floating, float)):
        thresh_types = ["bw"]
    if thresh_types[0] == "(":
        thresh_types = thresh_types[1:len(thresh_types) - 1]
    if "," in thresh_types:
        thresh_types = thresh_types.split(',')
    if not isinstance(thresh_types, list):
        thresh_types = [thresh_types]
    return thresh_types


def dic_exclnan(dic, var1, var2=None):
    if var2 is None:
        var2 = var1
    inds = [i for i, j in enumerate(dic[var2]) if j == j]
    vals = [dic[var1][i] for i, j in enumerate(dic[var2]) if j == j]
    return inds, vals
