#! /usr/bin/env python

import cv2
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

from pythutils.drawutils import namedcols
from pythutils.datutils import contour_to_tuple
from pythutils.mathutils import ptsToDist

from .geometry import get_coord


def intproper(x):
    i, f = divmod(x, 1)
    return int(i + ((f >= 0.5) if (x > 0) else (f > 0.5)))


def concom(contour):
    M = cv2.moments(contour)
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])
    return (cX, cY)


def filledconcom(contour):
    x, y = zip(*contoarray(contour))
    cX = intproper(np.mean(x))
    cY = intproper(np.mean(y))
    return (cX, cY)


def contoarray(contour):
    return np.array([list(pt[0]) for pt in contour])


def arraytocon(array):
    return np.array([[list([pt][0])] for pt in array])


def roifromcon(contour, pan=5):
    x = [i[0][0] for i in contour]
    y = [i[0][1] for i in contour]
    tl = (max(1, min(x) - pan), max(1, min(y) - pan))
    br = (max(x) + pan, max(y) + pan)
    w = br[0] - tl[0]
    h = br[1] - tl[1]
    return tl, br, w, h


def concoords(contour, filled=True, array=True):
    tl, br, w, h = roifromcon(contour)
    canvas = np.zeros((h, w), np.uint8)
    cv2.polylines(canvas, [contour - tl], 0, 255, 1)
    canvas = cv2.blur(canvas, (3, 3))
    canvas = cv2.erode(canvas, np.ones((3, 3), np.uint8))
    con, _ = cv2.findContours(canvas, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
    if not filled:
        out = contoarray(con[0] + tl) if array else con[0] + tl
    else:
        canvas = np.zeros((h, w), np.uint8)
        cv2.fillPoly(canvas, [con[0]], 255)
        out = [(x + tl[0], y + tl[1]) for y, row in enumerate(canvas) for x, col in enumerate(row) if col == 255]
        if not array:
            out = arraytocon(out)
    return out


def consplit(prevarrays, mergedarray):
    dists, ids = KDTree(prevarrays[0]).query(mergedarray)
    dists2, ids2 = KDTree(prevarrays[1]).query(mergedarray)

    distlist = [[i, dists[i], ids[i], dists2[i], ids2[i]] for i, _ in enumerate(dists)]
    mindis = [abs(distlist[i][1] - distlist[i][3]) for i, _ in enumerate(distlist)]
    distlist = [row + [mindis[i]] for i, row in enumerate(distlist)]
    distlist.sort(key=lambda x: int(x[5]), reverse=True)
    conids = []
    prevratio = len(prevarrays[0]) / (len(prevarrays[0]) + len(prevarrays[1]))
    for i, _ in enumerate(distlist):
        connrs = [1, 2] if distlist[i][1] < distlist[i][3] else [2, 1]
        if connrs[0] == 2:
            prevratio = 1 - prevratio
        lenthresh = int(prevratio * len(mergedarray))
        conids += [connrs[0] if conids.count(connrs[0]) <= lenthresh else connrs[1]]

    newcon1 = [[list(mergedarray[distlist[i][0]]), distlist[i][0]] for i, j in enumerate(conids) if j == 1]
    newcon1.sort(key=lambda x: int(x[1]))
    newcon1 = arraytocon([i[0] for i in newcon1])
    newcon1 = concoords(newcon1, True, False)
    newcon1 = arraytocon([list(pt[0]) for pt in newcon1 if tuple(pt[0]) in mergedarray])
    newcon1 = concoords(newcon1, False, False)

    newcon2 = [[list(mergedarray[distlist[i][0]]), distlist[i][0]] for i, j in enumerate(conids) if j == 2]
    newcon2.sort(key=lambda x: int(x[1]))
    newcon2 = arraytocon([i[0] for i in newcon2])
    newcon2 = concoords(newcon2, True, False)
    newcon2 = arraytocon([list(pt[0]) for pt in newcon2 if tuple(pt[0]) in mergedarray])
    newcon2 = concoords(newcon2, False, False)

    newcons = [newcon1] if len(newcon2) == 0 else (newcon1, newcon2)
    return np.array(newcons)


def get_head(x, y, angle, length, contour):
    fbx, fby = get_coord(x, y, angle, length * 0.5)
    contuple = contour_to_tuple(contour)
    _, pixid = KDTree(contuple).query((fbx, fby))
    fx, fy = contuple[pixid]
    return fx, fy


def con_lathom(skeleton, contour):
    distmat = cdist(skeleton, [list(i[0]) for i in contour])
    minmat = [min(row) for row in distmat]
    lathom = round((np.max(minmat) - np.min(minmat)) / len(minmat), 3)
    return lathom


def draw_coordlist(img, coordlist, col="orange"):
    cols, rows = list(zip(*coordlist))
    img[rows, cols] = list(namedcols(col, BRG=False))
    return img


def getouterpts(img):
    horpts = [pt for x in range(1, img.shape[1]) for pt in [(x, 1), (x, img.shape[0])]]
    verpts = [pt for y in range(1, img.shape[0]) for pt in [(1, y), (img.shape[1], y)]]
    return sorted(horpts + verpts, key=lambda k: [k[0], k[1]])


def loadmask(maskfile):
    try:
        img_mask = cv2.imread(maskfile, 0)
        kernel = np.ones((5, 5), np.uint8)
        img_mask = cv2.erode(img_mask, kernel)
        img_mask = cv2.dilate(img_mask, kernel)
    except:
        img_mask = None
    return img_mask


def coordsfrommask(maskfile, epsilon=4.0):
    """
    Given a mask file (filename or numpy array), return contours and simplified (x, y) coordinates.
    epsilon: simplification in pixels — increase for simpler shapes.
    """
    if isinstance(maskfile, str):
        img_mask = cv2.imread(maskfile, 0)
    elif isinstance(maskfile, np.ndarray):
        img_mask = maskfile
        if len(img_mask.shape) == 3:
            img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
    else:
        return None, None

    try:
        img_maskinv = cv2.erode(img_mask, np.ones((5, 5), np.uint8))
        img_maskinv = cv2.threshold(img_maskinv, 10, 255, cv2.THRESH_BINARY_INV)[1]
        maskconts, _ = cv2.findContours(img_maskinv, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        maskcoords = []
        for cnt in maskconts:
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            maskcoords.extend([tuple(pt[0]) for pt in approx])
    except Exception as e:
        print("Mask error:", e)
        maskconts = None
        maskcoords = None
    return maskconts, maskcoords


def coordsfromzones(imgfile, palette_hues=None, tol=20, min_sat=200, min_val=200, epsilon=2.0):
    """For each zone (color) in the image, return a simplified polygon as a list of (x, y) tuples."""
    if palette_hues is None:
        palette_hues = list(range(0, 360, 36))
    img = cv2.imread(imgfile)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    zone_coords = {}
    for i, h in enumerate(palette_hues):
        lower = np.array([(h - tol) // 2, min_sat, min_val])
        upper = np.array([(h + tol) // 2, 255, 255])
        mask = cv2.inRange(img_hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            coords = [tuple(pt[0]) for pt in approx]
            if len(coords) > 2 and coords[0] == coords[-1]:
                coords = coords[:-1]
            if len(coords) >= 3:
                zone_coords[i + 1] = coords
    return zone_coords
