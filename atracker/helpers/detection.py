#! /usr/bin/env python

import cv2
import numpy as np

from pythutils.mathutils import points_to_angle

from .contours import concom
from .body import body_settings, measure_body
from .filters import dic_exclnan
from .detection_settings import gamma_correct, aspect_limits


def warp_barcode_patch(gray_img, contour, size=15):
    pts = contour[:, 0, :].astype(np.float32)
    if len(pts) != 4:
        return None
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    patch = cv2.warpPerspective(gray_img, M, (size, size))
    return patch


class ProcessImage:

    def __init__(self, img, img_bg=None, img_mask=None, threshtype="bw", blur=9,
        erode=1, blur2=1, threshold=50, min_area=100, max_area=10000,
        colmin=None, colmax=None, body_mode="centroid",
        tags=None, tag_ids=None, barcode_size=15, barcode_tol=1,
        flicker_threshold=None, min_aspect_ratio=1.4, max_aspect_ratio=10, gamma=1.0,
        head_rule="near_widest", skeleton_points=10):

        self.img = np.asarray(img)
        self.img_bg = np.asarray(img_bg)
        self.gamma = float(gamma)
        self.img = gamma_correct(self.img, self.gamma)
        if img_bg is not None:
            self.img_bg = gamma_correct(self.img_bg, self.gamma)
        self.img_mask = img_mask
        self.threshtype = threshtype
        self.blur = max(1, blur | 1)
        self.blur2 = max(1, blur2 | 1)
        erode = max(1, erode | 1)
        self.erode = np.ones((erode, erode), np.uint8)
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area
        self.colmin = colmin
        self.colmax = colmax
        self.body_mode, self.head_rule, self.skeleton_points = body_settings(body_mode, head_rule, skeleton_points)
        self.flicker_threshold = flicker_threshold
        self.flicker = False
        self.min_aspect_ratio, self.max_aspect_ratio = aspect_limits(
            {"min_aspect_ratio": min_aspect_ratio, "max_aspect_ratio": max_aspect_ratio})

        # Barcoding
        self.tags = tags  # array of barcode templates
        self.tag_ids = tag_ids  # corresponding IDs
        self.barcode_size = barcode_size  # warp target size
        self.barcode_tol = barcode_tol  # max Hamming distance allowed


    def preprocess_bw_mode(self):

        """Shared thresholding for bw and barcode modes."""
          
        self.img = cv2.subtract(self.img_bg, self.img)  # only darker-than-background survives
        
        if self.flicker_threshold is not None:
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            if np.mean(gray) > self.flicker_threshold:
                self.flicker = True
                self.img_thresh = np.zeros(gray.shape, np.uint8)
                return
        
        if self.img_mask is not None:
            if self.img_mask.shape[:2] != self.img.shape[:2]:
                self.img_mask = cv2.resize(self.img_mask, (self.img.shape[1], self.img.shape[0]), interpolation=cv2.INTER_AREA)
            self.img = cv2.bitwise_and(self.img, self.img, mask=self.img_mask)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        self.img_thresh = cv2.GaussianBlur(self.img, (self.blur, self.blur), 3)
        self.img_thresh = cv2.erode(self.img_thresh, self.erode, 3)
        self.img_thresh = cv2.GaussianBlur(self.img_thresh, (self.blur2, self.blur2), 3)
        self.img_thresh = cv2.threshold(self.img_thresh, self.threshold, 255, cv2.THRESH_BINARY)[1]        
        
    def process(self):

        """Detect silhouettes and extract the independently selected body detail."""

        if self.threshtype.startswith("bw"):
            self.preprocess_bw_mode()
        elif self.threshtype == "barcode":
            self.preprocess_bw_mode()
            return self.process_barcode_mode()
        else:
            self.img_thresh = cv2.GaussianBlur(self.img, (self.blur, self.blur), 0)
            self.img_thresh = cv2.cvtColor(self.img_thresh, cv2.COLOR_BGR2HSV)
            self.img_thresh = cv2.inRange(self.img_thresh, self.colmin, self.colmax)
            self.img_thresh = cv2.erode(self.img_thresh, None, iterations=2)
            self.img_thresh = cv2.dilate(self.img_thresh, None, iterations=2)

        # Extract contours
        allcons,_ = cv2.findContours(self.img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        # Compute areas once, sort contours by area descending
        con_areas = [cv2.contourArea(c) for c in allcons]
        order = sorted(range(len(allcons)), key=lambda i: con_areas[i], reverse=True)
        self.allcons = [allcons[i] for i in order]
        sorted_areas = [con_areas[i] for i in order]

        # Create conlist
        varlist = ("id","objbox","area","aspect_ratio","length","com",
        "head","tail","angle","skeleton","contour",
        "length_method","center_source","body_valid","body_straightness","head_assignment","head_asymmetry")
        self.conlist = {var:[np.nan]*len(self.allcons) for var in varlist}

        # Go through the contours and extract contour information
        for ind, (contour, area) in enumerate(zip(self.allcons, sorted_areas)):
            self.processcon(ind, contour, precomputed_area=area)

        # Subset to only contours within dimensions
        inds,_ = dic_exclnan(self.conlist,"com")
        for key in self.conlist.keys():
            self.conlist[key] = [self.conlist[key][i] for i in inds]

        return self.img_thresh, self.allcons, self.conlist


    def process_barcode_mode(self):
        """
        Barcode detection mode using the same preprocessing steps as 'bw',
        followed by contour filtering and barcode matching.
        """

        # Preprocessing: same as 'bw'
        self.img = cv2.absdiff(self.img, self.img_bg)
        if self.img_mask is not None:
            if self.img.shape != self.img_mask.shape:
                self.img_mask = cv2.resize(self.img_mask, (self.img.shape[1], self.img.shape[0]), interpolation=cv2.INTER_AREA)
            self.img = cv2.bitwise_and(self.img, self.img, mask=self.img_mask)

        self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        self.img_thresh = cv2.GaussianBlur(self.img, (self.blur, self.blur), 3)
        self.img_thresh = cv2.erode(self.img_thresh, self.erode, 3)
        self.img_thresh = cv2.GaussianBlur(self.img_thresh, (self.blur2, self.blur2), 3)
        self.img_thresh = cv2.threshold(self.img_thresh, self.threshold, 255, cv2.THRESH_BINARY)[1]

        # Contour detection
        contours, _ = cv2.findContours(self.img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Barcode results storage
        varlist = ("barcode_id", "barcode_conf", "com", "contour")
        self.conlist = {var: [] for var in varlist}
        self.allcons = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_area < area < self.max_area):
                continue

            approx = cv2.approxPolyDP(cnt, 0.05 * cv2.arcLength(cnt, True), True)
            if len(approx) != 4:
                continue

            # Warp contour region to square patch
            patch = warp_barcode_patch(self.img, approx, size=self.barcode_size)
            if patch is None:
                continue

            patch_bin = (patch >= self.threshold)
            best_dist = np.inf
            best_id = None

            for tag, tag_id in zip(self.tags, self.tag_ids):
                tag_resized = cv2.resize(tag.astype(np.uint8) * 255, (self.barcode_size, self.barcode_size), interpolation=cv2.INTER_NEAREST)
                tag_bin = (tag_resized >= 128)
                dist = np.sum(patch_bin != tag_bin)

                if dist < best_dist:
                    best_dist = dist
                    best_id = tag_id

            if best_dist <= self.barcode_tol:
                # Centroid of contour
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    center = (cx, cy)
                else:
                    center = (0, 0)

                self.conlist["barcode_id"].append(best_id)
                self.conlist["barcode_conf"].append(best_dist)
                self.conlist["com"].append(center)
                self.conlist["contour"].append(cnt)
                self.allcons.append(cnt)

        return self.img_thresh, self.allcons, self.conlist

    
    def processcon(self, ind, contour, precomputed_area=None):

        # Get contour area (reuse precomputed value from sort if available)
        self.conlist["contour"][ind] = contour
        area = int(precomputed_area) if precomputed_area is not None else int(cv2.contourArea(contour))
        self.conlist["area"][ind] = area

        rotrecb = cv2.minAreaRect(contour)
        short, long = sorted(rotrecb[1])
        aspect_ratio = long / short if short > 0 else 0
        self.conlist["aspect_ratio"][ind] = round(aspect_ratio,2)

        # Continue only with contours of the right size and shape
        if self.min_area < area < self.max_area and self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio:
            # Get object centroid
            self.conlist["com"][ind] = concom(contour)
            measurements = measure_body(contour, self.conlist["com"][ind],
                self.body_mode, self.head_rule, self.skeleton_points)
            for key, value in measurements.items():
                self.conlist[key][ind] = value
            if self.body_mode == "skeleton" and measurements["body_valid"]:
                self.conlist["angle"][ind] = (float(points_to_angle(
                    measurements["com"], measurements["head"], flip=True))
                    if self.head_rule != "unassigned" else np.nan)
