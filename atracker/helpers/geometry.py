#! /usr/bin/env python

import math
import os
import cv2
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import shapely
from shapely.geometry import Point, Polygon

from pythutils.mathutils import ptsToDist, points_to_angle, angle_to_vec, get_weights


# ---------------------------------------------------------------------------
# Geometry and distance functions
# ---------------------------------------------------------------------------

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
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    poly = Polygon(poly_coords)
    pts = shapely.points(xs, ys)
    dists = shapely.distance(poly.exterior, pts)
    inside = shapely.contains_xy(poly, xs, ys)
    dists[inside] = -dists[inside]
    return dists


def dist_to_zone(xs, ys, coords, conv=1.0):
    """Signed distance to a zone (point, rect, or polygon). Returns None if coords is empty."""
    if not coords:
        return None
    if len(coords) == 1:
        return dist_to_point(xs, ys, coords[0][0], coords[0][1]) * conv
    arr = np.asarray(coords)
    if len(coords) == 4 and is_axis_aligned_rectangle(coords):
        return dist_to_rect(xs, ys,
                            arr[:, 0].min(), arr[:, 0].max(),
                            arr[:, 1].min(), arr[:, 1].max()) * conv
    if len(coords) >= 3:
        return dist_to_poly(xs, ys, coords) * conv
    return None


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


# ---------------------------------------------------------------------------
# Angle functions (merged from angles.py)
# ---------------------------------------------------------------------------

def minangle(angle, angle2):
    angle = angle2 - angle
    angle = abs((angle + 180) % 360 - 180)
    return angle


def hflipangle(angle):
    vx, vy = angle_to_vec(angle)
    angle = points_to_angle((vx, vy * -1))
    return angle


def hvflipangle(angle):
    if angle < 0:
        angle += 180.0
    else:
        angle -= 180.0
    return angle


def compare_angle(angle, angle2):
    vect1 = (np.sin(np.radians(angle)), np.cos(np.radians(angle)))
    vect2 = (np.sin(np.radians(angle2)), np.cos(np.radians(angle2)))
    angle = hvflipangle(angle) if np.inner(vect1, vect2) < 0 else angle
    return angle


def anglediff(angle1, angle2):
    return (angle1 - angle2 + 180) % 360 - 180


def get_anglediff(angle):
    if type(angle) is not pd.core.series.Series:
        angle = pd.Series(angle)
    anglediff = angle - angle.shift(periods=1) if len(angle) > 1 else angle
    anglediff.loc[anglediff < -180] = 360 - abs(anglediff.loc[anglediff < -180])
    anglediff.loc[anglediff > 180] = (360 - anglediff.loc[anglediff > 180]) * -1
    return list(anglediff)


def orientchecker(curr_frame, start_frame, prev_frame, delay, centre_coord, con_angle,
                  extdists, convexity, prev_angle, avg_vel, vel_thresh, prev_coord,
                  min_convex=0.7, min_extdist_ratio=0.1, max_anglechange=30):
    convex_ok = convexity > min_convex
    extdist_ratio = 1 - (min(extdists) / max(extdists))
    distratio_ok = extdist_ratio >= min_extdist_ratio
    delay_ok = (curr_frame - prev_frame) <= delay

    if None in prev_coord:
        vel_ok = False
    else:
        if not delay_ok:
            vel_ok = False
        else:
            if avg_vel is None:
                vel_ok = False
            else:
                vel = ptsToDist(centre_coord, prev_coord) / (curr_frame - prev_frame)
                vel_ok = avg_vel >= vel_thresh and vel >= vel_thresh
                move_angle = points_to_angle(prev_coord, centre_coord, flip=True)

    if convex_ok and distratio_ok:
        if extdists[0] < extdists[1]:
            new_angle = con_angle
        else:
            new_angle = hvflipangle(con_angle)

    elif not distratio_ok and vel_ok:
        move_angle = points_to_angle(prev_coord, centre_coord, flip=True)
        new_angle = compare_angle(con_angle, move_angle)
        if minangle(prev_angle, new_angle) > max_anglechange:
            new_angle = hvflipangle(new_angle)

    elif convex_ok and not distratio_ok and not vel_ok and delay_ok and not np.isnan(prev_angle):
        new_angle = compare_angle(con_angle, prev_angle)
        if minangle(prev_angle, new_angle) > max_anglechange:
            new_angle = hvflipangle(new_angle)

    else:
        new_angle = np.nan

    return new_angle


def fixheadtail(series, areamultiplier=1.75, distmultiplier=10,
                seqlenthreshold=40, thresharea=None):
    from .trajectory import calcudiff, get_anglediff as _get_anglediff

    data = series.copy()
    swappedinds = []
    if thresharea is None:
        thresharea = np.nanmedian(data.area)

    data["headdiff"] = round(calcudiff(data.fx, data.fy, period=1))
    data["taildiff"] = round(calcudiff(data.tx, data.ty, period=1))
    data["headangle"] = round(calcudiff(data.fx, data.fy, period=1, angle=True))
    data["tailangle"] = round(calcudiff(data.tx, data.ty, period=1, angle=True))
    tx, ty, fx, fy = (data["tx"].copy(), data["ty"].copy(), data["fx"].copy(), data["fy"].copy())
    data["fxsw"], data["fysw"] = (data.fx, data.fy)
    inds = np.arange(0, len(data.fx), 2)
    data.loc[inds, "fxsw"], data.loc[inds, "fysw"] = (data.loc[inds, "tx"], data.loc[inds, "ty"])
    data["swapdiff"] = round(calcudiff(data.fxsw, data.fysw, period=1))
    data["anglediff"] = get_anglediff(data.orient)

    indstocheck = data.index[(data["headdiff"] > data["swapdiff"]) | (data["taildiff"] > data["swapdiff"])]
    if len(indstocheck) > 0:
        for i, ind in enumerate(indstocheck):
            if ind in swappedinds:
                continue
            if i == len(indstocheck) - 1:
                if len(indstocheck) == 1:
                    pass
                elif (max(data.index) - ind) < seqlenthreshold:
                    data.loc[range(ind, max(data.index) + 1), ["head", "fx", "fy", "tail", "tx", "ty"]] = np.nan
                else:
                    data.loc[range(max(ind - 10, 0), ind + 10), ["head", "fx", "fy", "tail", "tx", "ty"]] = np.nan
                continue
            if data.loc[ind, "taildiff"] < (np.nanmedian(data["taildiff"]) * distmultiplier):
                data.loc[ind, ["head", "fx", "fy"]] = np.nan
                if data.loc[ind, "anglediff"] < 20:
                    continue
            if data.loc[ind, "headdiff"] < (np.nanmedian(data["headdiff"]) * distmultiplier):
                data.loc[ind, ["tail", "fx", "fy"]] = np.nan
                if data.loc[ind, "anglediff"] < 20:
                    continue
            if abs(anglediff(data.loc[ind, "headangle"], data.loc[ind, "tailangle"])) < 90:
                continue
            seqlen = indstocheck[i + 1] - indstocheck[i]
            if (seqlen) > seqlenthreshold:
                data.loc[range(ind, ind + 10), ["head", "fx", "fy", "tail", "tx", "ty"]] = np.nan
                continue
            if data.loc[ind, "orienttoexcl"] == 1:
                while True:
                    ind = ind + 1
                    if ind > (len(data) - 1) or data.loc[ind, "orienttoexcl"] != 1:
                        break
            toswap = range(ind + 1, indstocheck[i + 1] + 1)
            if data.loc[indstocheck[i + 1], "orienttoexcl"] == 1:
                data.loc[toswap, ["head", "fx", "fy", "tail", "tx", "ty"]] = np.nan
            data.loc[toswap, "fx"] = tx.loc[toswap]
            data.loc[toswap, "fy"] = ty.loc[toswap]
            data.loc[toswap, "tx"] = fx.loc[toswap]
            data.loc[toswap, "ty"] = fy.loc[toswap]
            swappedinds += toswap

    return data, swappedinds
