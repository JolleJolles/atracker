#! /usr/bin/env python

import math
import numpy as np
import pandas as pd

from pythutils.mathutils import points_to_angle, angle_to_vec, get_weights, ptsToDist


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
