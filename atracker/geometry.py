#! /usr/bin/env python

import os
import cv2
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from shapely.geometry import Point, Polygon

from pythutils.mathutils import ptsToDist, points_to_angle


def dist_to_target(xs, ys, target, method="point", **kwargs):
    if method == "point":
        return dist_to_point(xs, ys, *target)
    elif method == "rect":
        return dist_to_rect(xs, ys, *target)
    elif method == "poly":
        return dist_to_poly(xs, ys, target)
    elif method == "mask":
        return dist_to_mask(xs, ys, target, kwargs.get("conv", 1.0))
    else:
        raise ValueError("Unknown method")


def dist_to_point(xs, ys, cx, cy):
    return np.hypot(xs - cx, ys - cy)


def calc_borderdist(pt, roi):
    x, y = pt
    xmin, ymin = roi[0]
    xmax, ymax = roi[1]
    return dist_to_rect(np.array([x]), np.array([y]), xmin, xmax, ymin, ymax)[0]


def calc_borderdistdf(df, roi):
    cx = df['cx'].to_numpy()
    cy = df['cy'].to_numpy()
    xmin, ymin = roi[0]
    xmax, ymax = roi[1]
    return dist_to_rect(cx, cy, xmin, xmax, ymin, ymax)


def dist_to_rect(xs, ys, xmin, xmax, ymin, ymax):
    dx = np.maximum(np.maximum(xmin - xs, 0), xs - xmax)
    dy = np.maximum(np.maximum(ymin - ys, 0), ys - ymax)
    outside_dist = np.hypot(dx, dy)
    inside = (xs >= xmin) & (xs <= xmax) & (ys >= ymin) & (ys <= ymax)
    if np.any(inside):
        min_dist_inside = np.minimum.reduce([
            xs[inside] - xmin,
            xmax - xs[inside],
            ys[inside] - ymin,
            ymax - ys[inside]
        ])
        outside_dist[inside] = -min_dist_inside
    return outside_dist


def dist_to_poly(xs, ys, poly_coords):
    poly = Polygon(poly_coords)
    pts = [Point(x, y) for x, y in zip(xs, ys)]
    dists = np.array([poly.exterior.distance(pt) for pt in pts])
    inside = np.array([poly.contains(pt) for pt in pts])
    dists[inside] = -dists[inside]
    return dists


def dist_to_mask(xs, ys, mask, conv=1.0):
    """
    Signed distance to mask: negative if inside, positive if outside, 0 on edge.
    xs, ys: coordinates in mask pixel space.
    mask: binary (uint8) mask.
    conv: pixel to mm conversion.
    """
    h, w = mask.shape
    xs_ = np.round(xs).astype(int)
    ys_ = np.round(ys).astype(int)
    valid = (xs_ >= 0) & (xs_ < w) & (ys_ >= 0) & (ys_ < h)
    dists = np.full(xs.shape, np.nan)
    inside = np.zeros(xs.shape, dtype=bool)
    inside[valid] = mask[ys_[valid], xs_[valid]] > 0

    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours and len(contours[0]) > 0:
        edge_pts = np.vstack(contours).squeeze()
        tree = KDTree(edge_pts)
    else:
        edge_pts = None
        tree = None

    outside_idx = np.where(~inside & valid)[0]
    if len(outside_idx):
        ys_mask, xs_mask = np.where(mask > 0)
        if len(xs_mask):
            tree_mask = KDTree(np.column_stack([xs_mask, ys_mask]))
            for idx in outside_idx:
                pt = [xs_[idx], ys_[idx]]
                dists[idx] = tree_mask.query(pt)[0] * conv

    inside_idx = np.where(inside & valid)[0]
    if len(inside_idx) and edge_pts is not None:
        for idx in inside_idx:
            pt = [xs_[idx], ys_[idx]]
            dists[idx] = -tree.query(pt)[0] * conv
    elif len(inside_idx):
        dists[inside_idx] = 0

    return dists


def is_axis_aligned_rectangle(coords):
    coords = np.asarray(coords)
    if coords.shape[0] != 4:
        return False
    xs, ys = coords[:, 0], coords[:, 1]
    return (len(np.unique(xs)) == 2 and len(np.unique(ys)) == 2)


def valid_img_path(fileinfo, key, originals_dir):
    """Returns full image path if column exists, is not nan/empty, and file exists, else None."""
    val = getattr(fileinfo, key, None)
    if val is not None and isinstance(val, str) and val.strip() and val.strip().lower() != "nan":
        fpath = os.path.join(originals_dir, val)
        if os.path.isfile(fpath):
            return fpath
    return None


def get_coord(x, y, angle, length, astuple=False):
    x2 = np.array(x + np.sin(np.radians(angle)) * length)
    y2 = np.array(y - np.cos(np.radians(angle)) * length)
    if astuple:
        coord = [np.nan if np.isnan(a) else (int(a), int(b)) for a, b in zip([x2], [y2])][0]
        return coord
    else:
        return x2, y2


def adjpt(pt, xypad, add=True):
    if add:
        pt = (pt[0] + xypad[0], pt[1] + xypad[1])
    else:
        pt = (pt[0] - xypad[0], pt[1] - xypad[1])
    return pt


def fix_roi(roi, resolution):
    try:
        pt1 = max(0, roi[0][0]), max(0, roi[0][1])
        pt2 = min(resolution[0], roi[1][0]), min(resolution[1], roi[1][1])
    except:
        pt1, pt2 = ((0, 0), (resolution))
    return ((pt1, pt2))


def nextvec(prevpoint, currpoint, delay):
    displ = ptsToDist(prevpoint, currpoint) / delay
    angle = points_to_angle(prevpoint, currpoint)
    return get_coord(0, 0, angle, displ, astuple=True) if displ > 0.5 else (0, 0)


def series_to_point_tuple(df, cols):
    """Returns a tuple of NumPy arrays (x, y) from a DataFrame's specified columns."""
    if len(cols) != 2:
        raise ValueError("cols must be a list or tuple of exactly two column names")
    x = pd.to_numeric(df[cols[0]], errors='coerce').to_numpy()
    y = pd.to_numeric(df[cols[1]], errors='coerce').to_numpy()
    return (x, y)


def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]


def geom_tocoord(geom):
    if geom.geom_type in ["GeometryCollection", "MultiLineString", "MultiPoint"]:
        if len(list(geom.geoms)) == 0:
            return np.nan
        geom = geom.geoms[0]
    if geom.geom_type == "Polygon":
        coord = geom.exterior.coords.xy
    else:
        if len(np.array(geom.coords)) == 0:
            return np.nan
        coord = geom.coords.xy
    return int(coord[0][0]), int(coord[1][0])


def convert(x, y, conv, height, flip=True, roi=None, already_relative=False, decimals=3):
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    conv = float(conv)

    if roi is not None:
        x0, y0 = roi[0]
        x1, y1 = roi[1]
        if not already_relative:
            x = x - float(x0)
            y = y - float(y0)
        if flip:
            y = (float(y1) - float(y0)) - y
    else:
        if flip:
            y = float(height) - y

    x = np.round(x * conv, decimals)
    y = np.round(y * conv, decimals)
    return x, y
