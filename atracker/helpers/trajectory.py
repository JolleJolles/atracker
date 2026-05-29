#! /usr/bin/env python

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype
from scipy.signal import savgol_filter
from scipy.spatial import KDTree
from shapely.geometry import Point, Polygon

from pythutils.mathutils import get_weights

from .geometry import calc_borderdist


def differentiate(val, period=1):
    val = np.asarray(val, dtype=np.float64)
    diff = np.full_like(val, np.nan)
    if len(val) > period:
        diff[period:] = val[period:] - val[:-period]
    return diff


def calcudiff(x, y, period=1, angle=False):
    x = pd.to_numeric(x, errors='coerce').to_numpy()
    y = pd.to_numeric(y, errors='coerce').to_numpy()
    xdiff = differentiate(x, period)
    ydiff = differentiate(y, period)
    if angle:
        return np.arctan2(xdiff, ydiff) * 180 / np.pi
    else:
        return np.sqrt(xdiff**2 + ydiff**2)


def weightedavg(datlist, minlen=10):
    weights = get_weights(length=len(datlist))
    datlist = [val for i, val in enumerate(datlist) if val is not None and val == val]
    weights = [weights[i] for i, val in enumerate(datlist) if val is not None and val == val]
    weightedavg = round(np.average(datlist, weights=weights), 1) if len(datlist) >= minlen else np.nan
    return weightedavg


def getavgvel(coordlist, forward=True):
    x, y = zip(*coordlist)
    vellist = list(calcudiff(pd.Series(x), pd.Series(y), period=1))
    if not forward:
        vellist = vellist[::-1]
    avgvel = weightedavg(vellist, 3) if len(vellist) > 3 else np.nan
    return avgvel


def getindsections(fullinds, win=0, gap=1):
    """Get list of tracking sections with starting index, stopping index, and length."""
    inds = [fullinds[0]] + [fullinds[i] for i in list(range(1, len(fullinds))) if (fullinds[i] - fullinds[i - 1] > gap)]
    relinds = [list(fullinds).index(i) for i in inds] + [len(fullinds)]
    fullinds2 = [min(fullinds) - 1] + fullinds
    lengths = [fullinds2[relinds[i + 1]] - fullinds[relinds[i]] + 1 for i, val in enumerate(relinds[1:])]
    indsecs = [[ind, ind + lengths[i] - 1, lengths[i]] for i, ind in enumerate(inds)]
    if win > 0:
        indsecs = [[indsec[0] - win, indsec[1] + win, indsec[2] + win + win] for indsec in indsecs]
    for i, _ in enumerate(indsecs):
        if indsecs[i][0] < 0:
            indsecs[i][0] = 0
    return indsecs


def getalones(dataset, var, win):
    """Get lists of tracked sections shorter than specified window."""
    temp = dataset.dropna(subset=[var]).copy()
    if len(temp) == 0:
        alones = []
    elif len(temp) <= win:
        alones = temp.index.values
    else:
        inds = getindsections(list(temp.index.values))
        alinds = []
        for i, ind in enumerate(inds):
            if len(inds) == 1:
                if ind[2] < win:
                    alinds = alinds + [ind]
            else:
                if ind[2] < win:
                    if i == 0:
                        if (inds[i + 1][0] - ind[1]) > win:
                            alinds = alinds + [ind]
                    else:
                        if i < (len(inds) - 1):
                            if ((ind[0] - inds[i - 1][1])) > win and (inds[i + 1][0] - ind[1]) > win:
                                alinds = alinds + [ind]
                        else:
                            if (ind[0] - inds[i - 1][1]) > win:
                                alinds = alinds + [ind]
        alones = [item for i in alinds for item in list(range(i[0], i[0] + i[2]))]
    return alones


def findmissinginds(series, col, checkzero=False):
    """Output list of all indices with rows that contain missing values."""
    var = series[pd.isnull(list(series[col]))]
    missinds = [val for k, val in enumerate(var.index) if val > k]
    if checkzero:
        missinds = [val for k, val in enumerate(var.index) if val == k] + missinds if 0 in var.index else missinds
    return missinds


def checknearmask(series, coordcols, mask, indsecs, masksecs, win=5, nearmaskdis=50):
    nearlist = []
    awaylist = []

    for i, indsec in enumerate(indsecs):
        if (min(series.index.values) + 1) >= indsec[0]:
            beflen = win
        else:
            subset = series.loc[masksecs[i][0]:indsecs[i][0] - 1, coordcols].dropna()
            array = np.stack((
                pd.to_numeric(subset[coordcols[0]], errors='coerce'),
                pd.to_numeric(subset[coordcols[1]], errors='coerce')
            ), axis=1)
            array = array[~np.isinf(array).any(axis=1)]
            befdis, _ = KDTree(mask).query(array)
            beflen = len([n for n in befdis if n < nearmaskdis or n == np.inf])

        if max(series.index.values) == indsec[1]:
            aftlen = win
        else:
            subset = series.loc[indsecs[i][1] + 1:masksecs[i][1], coordcols].dropna()
            array = np.stack((
                pd.to_numeric(subset[coordcols[0]], errors='coerce'),
                pd.to_numeric(subset[coordcols[1]], errors='coerce')
            ), axis=1)
            array = array[~np.isinf(array).any(axis=1)]
            aftdis, _ = KDTree(mask).query(array)
            aftlen = len([n for n in aftdis if n < nearmaskdis or n == np.inf])

        if beflen > 0 and aftlen > 0:
            nearlist.append(i)
        else:
            awaylist.append(i)

    return nearlist, awaylist


def fillmissing(series, colpair, mask, roi, win=5, nearmaskdis=25, edgedis=10, lenthresh=100):
    """
    Fill-in missing data for specific numeric columns in a dataset.

    Missing data near a mask is left unfilled (object presumably behind mask).
    Missing data caused by disappearance from view is also left unfilled.
    Remaining missing values are filled by linear interpolation.
    """
    if roi is not None:
        roi = ((1, 1), (roi[1][0] - roi[0][0], roi[1][1] - roi[0][1]))

    missinds = findmissinginds(series, colpair[0], checkzero=False)
    missing = len(missinds)
    if missing > 0:
        indsecs = getindsections(missinds)
        indsecs = [ind for ind in indsecs if ind[0] >= (min(series.index.values) + 1) and ind[1] != max(series.index.values)]
        missinds = [item for i in indsecs for item in list(range(i[0], i[0] + i[2]))]
        missing = len(missinds)

        if missing > 0:
            diffsecs = getindsections(missinds, win=1)
            if mask not in (None, []):
                masksecs = getindsections(missinds, win=win)
                nearlist, awaylist = checknearmask(series, colpair, mask, indsecs, masksecs, nearmaskdis=nearmaskdis)
                indsecs = [indsec for i, indsec in enumerate(indsecs) if (indsec[2] < 5 and i in nearlist) or i in awaylist]
                diffsecs = [diffsec for i, diffsec in enumerate(diffsecs) if (diffsec[2] < (5 + 2) and i in nearlist) or i in awaylist]
                missing = sum([item[2] for item in indsecs])

        if missing > 0 and "cx" in colpair and roi is not None:
            dellist = []
            for i, indsec in enumerate(indsecs):
                befxy = (series.at[indsec[0] - 1, colpair[0]], series.at[indsec[0] - 1, colpair[1]])
                aftxy = (series.at[indsec[1] + 1, colpair[0]], series.at[indsec[1] + 1, colpair[1]])
                befout = calc_borderdist(befxy, roi)
                aftout = calc_borderdist(aftxy, roi)
                if befout < edgedis and aftout < edgedis:
                    series.loc[indsec[0]:indsec[1], "inroi"] = 0
                    dellist.append(i)
            indsecs = [ind for i, ind in enumerate(indsecs) if i not in dellist]
            diffsecs = [ind for i, ind in enumerate(diffsecs) if i not in dellist]
            missing = sum([item[2] for item in indsecs])

    if missing > 0:
        for i, indsec in enumerate(indsecs):
            if indsec[2] > lenthresh:
                missing -= indsec[2]
            else:
                for col in colpair:
                    diffsec = diffsecs[i]
                    newvals = np.around(np.linspace(series[col][diffsec[0]], series[col][diffsec[1]], diffsec[2]), 3)[1:diffsec[2] - 1]
                    series.loc[indsec[0]:indsec[1], col] = newvals

    return series, missing


def process_trajectories(series, mask=None, cover=True, win=5,
                         trajgap=50, inmaskdis=10, mintrajlength=10,
                         erase_coords=True, interpolate=True):
    """
    Process trajectories: detects mask-covered segments, removes data near mask,
    assigns trajectory IDs, interpolates gaps, and removes short trajectories.
    """
    series["inmask"] = 0
    series["traj"] = np.nan
    series["cx"] = pd.to_numeric(series["cx"], errors="coerce")
    series["cy"] = pd.to_numeric(series["cy"], errors="coerce")

    removed_mask = np.zeros(len(series), dtype=bool)
    if cover and mask and len(mask) > 0:
        polygon = Polygon(mask)
        valid = series[["cx", "cy"]].dropna().copy()
        points = list(valid.itertuples(index=True, name=None))
        locs = [(x[1], x[2]) for x in points]
        indices = [x[0] for x in points]
        distances, _ = KDTree(mask).query(locs)
        inside = [polygon.contains(Point(x, y)) for x, y in locs]
        removed_mask[indices] = (np.array(inside) | (distances < inmaskdis))
        series.loc[removed_mask, "inmask"] = 1
        if erase_coords:
            series.loc[removed_mask, ["cx", "cy"]] = np.nan

    nonmiss = series.index[series["cx"].notna() & (series["inroi"] != 0)]

    def _getindsections(idxs, gap=1):
        if not len(idxs): return []
        idxs = sorted(idxs)
        breaks = [0] + [i + 1 for i in range(len(idxs) - 1) if idxs[i + 1] - idxs[i] > gap] + [len(idxs)]
        return [(idxs[start], idxs[end - 1]) for start, end in zip(breaks[:-1], breaks[1:])]

    sections = _getindsections(nonmiss, gap=trajgap)

    for i, (start, end) in enumerate(sections, start=1):
        series.loc[start:end, "traj"] = i
        if interpolate:
            inds = series.loc[start:end].index
            for col in ["cx", "cy"]:
                y = series.loc[inds, col]
                if y.isna().any():
                    x = y.index
                    filled = y.interpolate(method='linear', limit_direction='both')
                    series.loc[x, col] = filled

    nremoved = 0
    for trajid, group in series.groupby("traj"):
        if pd.isna(trajid):
            continue
        if len(group) < mintrajlength:
            series.loc[group.index, ["cx", "cy", "traj"]] = np.nan
            nremoved += len(group)

    valid_trajids = [i for i in series.traj.unique() if not pd.isna(i)]
    for new_id, old_id in enumerate(valid_trajids, start=1):
        series.loc[series.traj == old_id, "traj"] = new_id

    ntraj = len(valid_trajids)
    return series, ntraj, int(removed_mask.sum())


def smooth(series, trajs=None, columns=("cx", "cy"), smoothwin=5, polyorder=3):
    for col in columns:
        if col in series.columns and not is_float_dtype(series[col].dtype):
            series[col] = pd.to_numeric(series[col], errors="coerce").astype("float64")

    if trajs is None:
        if len(series) > 6:
            for column in columns:
                if column not in series.columns:
                    continue
                nonan = series[column].dropna()
                nonanlen = len(nonan)
                win = smoothwin if nonanlen > smoothwin else nonanlen
                if win > 6:
                    win = win - 1 if win % 2 == 0 else win
                    if win <= polyorder:
                        win = polyorder + 2 + ((polyorder + 2) % 2)
                        if win > nonanlen:
                            continue
                    filt = savgol_filter(nonan.to_numpy(dtype=float), polyorder=polyorder, window_length=win)
                    series.loc[nonan.index, column] = filt
    else:
        for t in trajs:
            sub = series.loc[series.traj == t]
            if len(sub) > 6:
                for column in columns:
                    if column not in series.columns:
                        continue
                    nonan = sub[column].dropna()
                    nonanlen = len(nonan)
                    win = smoothwin if nonanlen > smoothwin else nonanlen
                    if win > 6:
                        win = win - 1 if win % 2 == 0 else win
                        if win <= polyorder:
                            win = polyorder + 2 + ((polyorder + 2) % 2)
                            if win > nonanlen:
                                continue
                        filt = savgol_filter(nonan.to_numpy(dtype=float), polyorder=polyorder, window_length=win)
                        series.loc[nonan.index, column] = filt

    return series
