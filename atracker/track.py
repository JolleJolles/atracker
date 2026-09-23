#! /usr/bin/env python

import os
import cv2
import sys
import time
import queue
import shutil
import threading
import numpy as np
import pandas as pd
import multiprocessing
from collections import deque
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from pythutils.sysutils import lineprint, Suppressor, removeline
from pythutils.fileutils import listfiles
from pythutils.mediautils import check_media, crop
from pythutils.mathutils import points_to_angle

from ast import literal_eval

from .helpers.geometry import get_coord, adjpt, geom_tocoord, fix_roi, hvflipangle
from .helpers.contours import concom, concoords, consplit, con_lathom, coordsfrommask, coordsfromzones
from .helpers.trajectory import getavgvel
from .helpers.filters import (filter_tracking_jumps, filter_contour_shape,
                               update_shape_history, check_threshtypes,
                               dic_exclnan, estimate_flicker_baseline)
from .helpers.media import videowriter, make_even, framechecks
from .helpers.data import subdic, eval_func_tuple
from .helpers.detection import ProcessImage
from .visualise import Visualiser

class KeyboardInterruptError(Exception): pass

class AsyncVideoWriter:
    def __init__(self, vidout):
        self.vidout = vidout
        self.q = queue.Queue(maxsize=64)
        self.thread = threading.Thread(target=self._writer, daemon=True)
        self.thread.start()

    def _writer(self):
        while True:
            frame = self.q.get()
            if frame is None:  # poison pill to stop
                break
            self.vidout.append_data(frame)

    def write(self, frame):
        self.q.put(frame)

    def close(self):
        self.q.put(None)  # signal stop
        self.thread.join()  # wait for all frames to be written
        self.vidout.close()

class Tracker:

    def __init__(self, pools, inds, trackfiles, dirs, overview, config,
                 threshinfo, start, stop, threshtype, objects,
                 checkconschange, suffix, max_framedist=200, overwrite=None, check_flicker=False,
                 skip_frames=0):

        self.pools = pools
        self.inds = inds
        self.trackfiles = trackfiles
        self.dirs = dirs
        self.overview = overview
        self.config = config
        self.threshinfo = threshinfo
        self.ustart = start
        self.ustop = stop
        self.suffix = "" if len(suffix)==0 else "_"+suffix
        self.advanced = bool(getattr(self.config.track, 'advanced', False))
        self.orientfrombw = bool(getattr(self.config.track, 'orientfrombw', False))
        self.linkdisthreshold = getattr(self.config.track, 'link_dist', getattr(self.config.track, 'linkdisthreshold', 100))
        self.mergedmindist = getattr(self.config.track, 'merge_dist', getattr(self.config.track, 'mergedmindist', 20))
        _cm = getattr(self.config.track, 'contour_mode', 'static')
        self.size_filter = bool(getattr(self.config.track, 'size_filter', _cm == 'dynamic'))
        self.keep_frames = 10
        self.tracked = 0
        self.threshtype_override = threshtype
        self.objects_override = objects
        self.checkconschange = checkconschange
        self.max_framedist = max_framedist
        self.overwrite = overwrite if overwrite is not None else self.config.track.overwrite
        self.check_flicker = check_flicker
        self.skip_frames = skip_frames

    def setuptracking(self, ind, trackfile):

        self.ind = ind
        if self.ind in self.inds:
            self.inds.remove(self.ind)
        self.trackfile = trackfile
        self.filebase = os.path.basename(trackfile)
        self.filebasezero = os.path.splitext(self.filebase)[0]
        self.filename = self.filebasezero
        self.tempfile = os.path.join(self.dirs["temp"], self.filebase)
        if "region" in self.overview:
            regions = self.overview[self.overview.video == self.filebasezero]["region"]
            if len(regions) > 1:
                self.region = self.overview.loc[ind]["region"]
                self.filename = self.filename + "_R" + str(int(self.region))
        conv = self.overview.loc[ind]["conv"]
        conv = 1 if conv!=conv else float(conv)
        self.linkdisthreshold = self.linkdisthreshold / conv
        self.trackedfile = os.path.join(self.dirs["temp"], self.filename+self.suffix+"_TR.mp4")
        self.trackeddata = os.path.join(self.dirs["tracked"], self.filename+self.suffix+".csv")

        pr_procid = multiprocessing.current_process()._identity
        pr_procid = "{P"+str(pr_procid[0])+"}" if len(pr_procid)>0 else ""
        pr_ind = "["+str(self.ind)+"] "
        self.pr_comm = pr_procid+pr_ind+self.filename+" "

        # Copy file information to tracker instance
        # ! Here we also get the self.thresh_types from the overview
        fileinfo = self.overview.loc[ind].squeeze()
        for name, values in fileinfo.items():
            values = int(values) if type(values) == np.int64 else values
            self.__dict__.update([(name, values)])
        if not isinstance(self.roi, str) or not self.roi.strip():
            lineprint(self.pr_comm + "roi is missing in overview, skipping..")
            return self.inds
        self.pt1, self.pt2 = literal_eval(self.roi)
        self.vidw  = self.pt2[0] - self.pt1[0]
        self.vidh =  self.pt2[1] - self.pt1[1]
        self.frame_start = self.frame_start if self.frame_start==self.frame_start else 1
        self.frame_start = self.ustart if self.ustart is not None else self.frame_start
        self.frame_stop = self.frame_stop if self.frame_stop==self.frame_stop else int(self.fcount)
        self.frame_stop = self.ustop if self.ustop is not None else self.frame_stop
        self.fps = self.fps if hasattr(self, "fps") and pd.notnull(self.fps) and self.fps > 0 else 25

        # Check if all threshtypes are in threshinfo
        ## check_threshtypes turns empty cells (which are nan) into "bw" threshtype
        nopass = False
        if self.threshtype_override is not None:
            self.thresh_types = check_threshtypes(self.threshtype_override)
        else:
            self.thresh_types = check_threshtypes(self.thresh_types)
        for t in self.thresh_types:
            if t not in self.threshinfo:
                lineprint(self.pr_comm + f" threshtype '{t}' not in threshinfo.. Skipping this file.")
                return self.inds  # <-- exit early

        # Skip if already tracked and overwrite is off
        if not self.overwrite:
            tracked_path = os.path.join(self.dirs["tracked"], self.filename + self.suffix + ".csv")
            if os.path.exists(tracked_path):
                lineprint(self.pr_comm + " already tracked, skipping..")
                return self.inds

        # Check required background image
        if isinstance(self.bgimg, float) or self.bgimg is None:
            lineprint(self.pr_comm + "bgimg cannot be found, skipping..")
            return self.inds

        # Clean up previous tracking output if overwriting
        if self.overwrite:
            try:
                if os.path.exists(self.trackedfile):
                    os.remove(self.trackedfile)
                if os.path.exists(self.trackeddata):
                    os.remove(self.trackeddata)
            except Exception as e:
                lineprint(self.pr_comm + f"warning while cleaning tracked files: {e}")


        # Confirm original video exists
        if not os.path.exists(self.trackfile):
            #lineprint(self.pr_comm + f"video file not found at {self.trackfile}, skipping..")
            return self.inds

        # Try moving video to temp folder
        try:
            shutil.move(self.trackfile, self.tempfile)
        except Exception as e:
            lineprint(self.pr_comm + f"could not move video to temp folder: {e}")
            return self.inds

        # Check if moved file is valid media
        with Suppressor():
            self.mediaok = check_media(self.tempfile, internal=True)

        if not self.mediaok:
            if os.path.exists(self.tempfile):
                if not os.path.exists(self.trackedfile):
                    self.inds.append(self.ind)
                lineprint(self.pr_comm + "media check failed, skipping video..")
            else:
                lineprint(self.pr_comm + "video could not be found after moving, skipping..")
            return self.inds


        # Set up video and related resources
        self.cap = cv2.VideoCapture(self.tempfile)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_start - 1)
        
        self.img_bg = cv2.imread(os.path.join(self.dirs["originals"], self.bgimg))
        self.img_bg = crop(self.img_bg, self.pt1, self.pt2)

        self.img_mask = None
        if isinstance(self.maskimg, str):
            img_mask = cv2.imread(os.path.join(self.dirs["originals"], self.maskimg), 0)
            img_mask = crop(img_mask, self.pt1, self.pt2)
            kernel = np.ones((5, 5), np.uint8)
            self.img_mask = cv2.erode(img_mask, kernel)
            self.img_mask = cv2.dilate(self.img_mask, kernel)
            self.img_mask = cv2.resize(self.img_mask, (self.vidw, self.vidh), interpolation=cv2.INTER_AREA)

        # Precompute scene overlay contours for visualization
        self.mask_contours = None
        if self.img_mask is not None:
            self.mask_contours, _ = coordsfrommask(self.img_mask)
        self.wall_contours = None
        if hasattr(self, "wallimg") and isinstance(self.wallimg, str):
            _wall_img = cv2.imread(os.path.join(self.dirs["originals"], self.wallimg))
            if _wall_img is not None:
                _wall_img = crop(_wall_img, self.pt1, self.pt2)
                _wall_img = cv2.resize(_wall_img, (self.vidw, self.vidh), interpolation=cv2.INTER_AREA)
                self.wall_contours, _ = coordsfrommask(_wall_img)
        self.zone_coords = None
        if hasattr(self, "zoneimg") and isinstance(self.zoneimg, str):
            _zone_img = cv2.imread(os.path.join(self.dirs["originals"], self.zoneimg))
            if _zone_img is not None:
                _zone_img = crop(_zone_img, self.pt1, self.pt2)
                _zone_img = cv2.resize(_zone_img, (self.vidw, self.vidh), interpolation=cv2.INTER_AREA)
                self.zone_coords = coordsfromzones(_zone_img, all_parts=True)

        # Set up video window (optional GUI)
        if self.config.vis.show_tracking and self.pools < 2:
            cv2.namedWindow("X", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("X", 1, 1)
            cv2.namedWindow(self.filename, cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(self.filename, 100, 0)

        # Final parameters and flags
        self.traj_length = int(self.fps * self.config.vis.traj_length)

        if self.objects_override is not None:
            self.objects = self.objects_override
        else:
            self.objects = self.overview.loc[self.overview.video == self.filebasezero, "objects"].iloc[0] if "objects" in self.overview else 1

        if self.objects != self.objects:  # handle NaN
            self.objects = 1
        else:
            self.objects = int(self.objects)

        self.threshcolors = False
        if len(self.thresh_types) > 1:
            self.objects = len(self.thresh_types)
            self.threshcolors = True

        # Launch the tracker
        self.tracksingle()

        return self.inds



    def _check_tomove(self):

        """Checks if video will be tracked again and moves file accordingly"""

        tracked = listfiles(self.dirs["tracked"], type = ".mp4", keepext = False)
        trackfiles = [os.path.splitext(os.path.basename(i))[0] for i in self.trackfiles]
        return len(tracked) < trackfiles.count(self.filebasezero)


    def exit(self):
        if self.config.vis.show_tracking:
            try:
                cv2.destroyWindow(self.filename)
            except:
                pass
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            for i in range(5):
                cv2.waitKey(1)
        if self.config.track.create_vid:
            self.vidout.close()
        self.cap.release()


    def conschange(self):

        # Did the number of contours decrease?
        contourchange = len(self.conlist["com"]) / len(self.prevlist["id"])

        # Is there one much larger contour?
        maxareachange = max(self.conlist["area"]) / max(self.prevlist["area"])

        # We have a potential merge, but check locations
        if contourchange < 1 and (maxareachange > 1.5 or 1 in self.prevlist["consmerged"]):
            maxsize = max(self.conlist["area"])
            self.curr_ind = [i for i,j in enumerate(self.conlist["area"]) if j==maxsize][0]
            curr_loc = [self.conlist["com"][self.curr_ind]]
            # get list of distances of the merged contour to all contours the previous frame
            dists = cdist(np.array(self.prevlist["com"]), np.array(curr_loc))
            self.prevclose_inds = [i for i,dis in enumerate(dists) if dis < self.mergedmindist]
            if len(self.prevclose_inds)>1:
                self.conlist["consmerged"][self.curr_ind] = 1


    def splitter(self):

        # Prepare current and previous contours
        mergedarray = concoords(self.conlist["contour"][self.curr_ind])
        prevcons = [self.prevlist["contour"][i] for i in self.prevclose_inds][:2]
        previds = [self.prevlist["id"][i] for i in self.prevclose_inds][:2]
        prevarray1 = concoords(prevcons[0])
        prevarray2 = concoords(prevcons[1])
        prevarrays = np.array((prevarray1,prevarray2))

        # Predict current contours position
        counter=0
        for i,ind in enumerate(self.prevclose_inds):
            counter += 1
            id = self.prevlist["id"][ind]
            v = self.movedat[id]["vel"][-1]
            h = self.movedat[id]["head"][-1]
            # don't add more than two previous arrays to be considered
            if counter<3 and v==v and h==h:
                coord = [get_coord(0, 0, h, v, astuple=True)]
                prevarrays[i] = np.array(prevarrays[i])+coord

        # Run the splitting function
        newcons = consplit(prevarrays, mergedarray)

        # Remove merged contour from conlist
        mergedind = [i for i,j in enumerate(self.conlist["consmerged"]) if j==1][0]
        for key in self.conlist.keys():
            del self.conlist[key][mergedind]

        # Add newly split contours to conlist
        for i,newcon in enumerate(newcons):
            for key in self.conlist.keys():
                if key == "consmerged":
                    self.conlist[key].append(1)
                elif key == "contour":
                    self.conlist[key].append(newcon)
                elif key == "com":
                    self.conlist[key].append(concom(newcon))
                elif key == "area":
                    self.conlist[key].append(cv2.contourArea(newcon))
                else:
                    self.conlist[key].append(np.nan)


    def linkIDs(self):
        """
        Assigns IDs to contours by linking them to previously tracked objects based on spatial proximity.

        For grayscale ('bw') thresholding, IDs are numeric and tracked individually.
        For color-based thresholding, a fixed color label (string) is used.
        """

        if self.thresh_type.startswith("bw"):

            # Determine whether to link to previous contours
            if len(self.movedat) == 0 or self.threshcolors:
                # No previous data or color thresholding present: assign default numeric IDs
                IDsfinal = [i + 1 if i + 1 in range(1, self.objects + 1) else np.nan for i in range(len(self.conlist["com"]))]
            else:
                # Only use previously tracked grayscale (numeric) IDs
                bwmovedat = {k: v for k, v in self.movedat.items() if isinstance(k, (int, np.integer))}
                previds = list(bwmovedat.keys())
                coms = []

                for id in bwmovedat:
                    # Get last known (valid) center of mass
                    nonnans = [(i, j) for i, j in enumerate(bwmovedat[id]["com"]) if j[0] == j[0]]
                    if len(nonnans) > 0:
                        lastvalidind, com = nonnans[0]
                        coms.append(com)
                    else:
                        previds.remove(id)  # Drop ID with no valid history

                # Link new contours to previous ones using Hungarian matching
                if len(coms) > 0:
                    # Calculate frame-based adaptive distance threshold
                    disthreshold = int(self.linkdisthreshold * (self.frame_nr - bwmovedat[id]["frame"][lastvalidind]))
                    dismat = cdist(np.array(coms), np.array(self.conlist["com"]))

                    # Optimal assignment
                    IDs_ind, curr_allocated = linear_sum_assignment(dismat)

                    # Keep only assignments under distance threshold
                    assigned_pairs = [
                        (previds[IDs_ind[i]], curr_allocated[i])
                        for i in range(len(curr_allocated))
                        if dismat[IDs_ind[i]][curr_allocated[i]] < disthreshold
                    ]
                    IDs = [id for id, _ in assigned_pairs]
                    curr_allocated = [j for _, j in assigned_pairs]
                else:
                    IDs = []
                    curr_allocated = []

                # Final allocation list
                IDsfinal = [np.nan] * len(self.conlist["com"])
                IDs_available = [i for i in range(1, int(self.objects) + 1) if i not in IDs]

                for i, com in enumerate(self.conlist["com"]):
                    if i in curr_allocated:
                        # Use assigned ID
                        IDsfinal[i] = IDs[next(k for k, l in enumerate(curr_allocated) if l == i)]
                    else:
                        # Fallback: try to reuse unassigned ID if close enough
                        if len(coms) > 0 and len(IDs_available) > 0:
                            dists = cdist([com], coms)[0]
                            best_idx = np.argmin(dists)
                            best_id = previds[best_idx]
                            if dists[best_idx] < disthreshold and best_id in IDs_available:
                                IDsfinal[i] = best_id
                                IDs_available.remove(best_id)
                            else:
                                IDsfinal[i] = IDs_available[0]
                                IDs_available = IDs_available[1:]
                        else:
                            # No valid history for any ID: re-acquire using available IDs
                            if len(IDs_available) > 0:
                                IDsfinal[i] = IDs_available.pop(0)
                            else:
                                IDsfinal[i] = np.nan

        else:
            if self.thresh_type not in self.movedat:
                # First appearance: assign to current contour
                IDsfinal = [self.thresh_type] + [np.nan] * (len(self.conlist["com"]) - 1)
            else:
                # Find closest current contour to previous COM
                nonnans = [(i, j) for i, j in enumerate(self.movedat[self.thresh_type]["com"]) if j[0] == j[0]]
                if len(nonnans) > 0:
                    lastvalidind, com = nonnans[0]
                    dismat = cdist(np.array([com]), np.array(self.conlist["com"]))
                    _, curr_allocated = linear_sum_assignment(dismat)
                else:
                    curr_allocated = []

                IDsfinal = [np.nan] * len(self.conlist["com"])
                if len(curr_allocated) > 0:
                    IDsfinal[curr_allocated[0]] = self.thresh_type

        # Store result
        self.conlist["id"] = IDsfinal


    def tracksingle(self):
        # Capture file paths locally so pooled workers don't overwrite each other
        trackedfile = self.trackedfile
        trackeddata = self.trackeddata
        tempfile = self.tempfile
        filename = self.filename
        create_vid = self.config.track.create_vid
        create_dat = self.config.track.create_dat

        # Set up video writer
        if create_vid:
            self.vidout = AsyncVideoWriter(videowriter(trackedfile, self.vidw, self.vidh, self.fps))
            try:
                    if self.vidout is None or (hasattr(self.vidout, 'isOpened') and not self.vidout.isOpened()):
                        raise ValueError("Video writer failed")
            except Exception:
                create_vid = False
                lineprint(self.pr_comm + "Warning: video writer could not be created. Video output disabled.")

        newline = False if self.pools<2 else True
        lineprint(self.pr_comm+"tracking started..", newline=newline)
        sys.stdout.flush()
        t1 = time.time()
        key = None
        self.fulldat = {"frame":[]}
        self.movedat = {}
        self.last_valid = {}
        self.shape_history = {}
        traj_history = {}   # {id: deque(maxlen=traj_length)} — only valid (cx, cy)
        tracked_ids = set()

        vis = Visualiser(objects=self.objects,
                         thresh_types=self.thresh_types,
                         threshcolors=self.threshcolors,
                         orientfrombw=self.orientfrombw,
                         mask_contours=self.mask_contours,
                         wall_contours=self.wall_contours,
                         zone_coords=self.zone_coords,
                         config=self.config)

        try:
            if self.check_flicker:
                self.flicker_threshold = estimate_flicker_baseline(
                    self.cap, self.img_bg, self.pt1, self.pt2)
                lineprint(f"Flicker threshold set at: {self.flicker_threshold:.2f}")
            
            while self.cap.isOpened():
                frameOK, self.img = self.cap.read()
                stop, skip, self.frame_nr = framechecks(self.cap, frameOK, None, self.frame_stop,
                                         self.config.vis.frame_disstep, self.ind)
                if stop:
                    break
                if skip:
                    continue
                if self.skip_frames > 0 and self.frame_nr % (self.skip_frames + 1) != 0:
                    continue
                
                # Create images to work with
                self.img = crop(self.img, self.pt1, self.pt2)
                self.img_draw = self.img.copy()

                # Get standard conlists
                _frame_info = []
                for t,self.thresh_type in enumerate(self.thresh_types):
                    ti = self.threshinfo[self.thresh_type]
                    if "blur2" not in ti:
                        ti["blur2"] = 1
                    min_ar = ti.get("min_aspect_ratio", 1.4)
                    max_ar = ti.get("max_aspect_ratio", 10)
                    if self.thresh_type.startswith("bw"):
                        PI = ProcessImage(self.img, self.img_bg, self.img_mask, self.thresh_type,
                            ti["blur"], ti["erode"], ti["blur2"], ti["threshold"], ti["min_area"], ti["max_area"],
                            simple=not self.advanced, flicker_threshold=self.flicker_threshold if self.check_flicker else None,
                            min_aspect_ratio=min_ar, max_aspect_ratio=max_ar)
                    else:
                        if "hue_lo" in ti:
                            colmin = (ti["hue_lo"], ti["sat_lo"], ti["val_lo"])
                            colmax = (ti["hue_hi"], ti["sat_hi"], ti["val_hi"])
                        else:
                            colmin = literal_eval(ti["colmin"])
                            colmax = literal_eval(ti["colmax"])
                        PI = ProcessImage(self.img, self.img_bg, self.img_mask, self.thresh_type,
                            ti["blur"], min_area=ti["min_area"], max_area=ti["max_area"],
                            colmin=colmin, colmax=colmax,
                            simple=True, flicker_threshold=self.flicker_threshold if self.check_flicker else None,
                            min_aspect_ratio=min_ar, max_aspect_ratio=max_ar)
                    self.img_thresh, self.allcons, self.conlist = PI.process()
                    if PI.flicker:
                        continue

                    # Check for merged contours and link IDs over time
                    self.conlist["consmerged"] = [0]*len(self.conlist["com"])
                    if self.thresh_type.startswith("bw"):
                        if len(self.conlist["com"])>0:
                            if hasattr(self, "prevlist"):
                                if self.checkconschange and len(self.prevlist["id"])>0:
                                    self.conschange()
                            if 1 in self.conlist["consmerged"]:
                                self.splitter()
                            self.linkIDs()
                    else:
                        if len(self.conlist["com"])>0:
                            self.linkIDs()

                    # Add data to fulldat
                    inds, ids = dic_exclnan(self.conlist, "id")
                    self.fulldat["frame"].extend([self.frame_nr]*len(ids))
                    self.fulldat.setdefault("id",[]).extend(ids)
                    self.fulldat.setdefault("area",[]).extend([self.conlist["area"][i] for i in inds])
                    self.fulldat.setdefault("aspect_ratio",[]).extend([self.conlist["aspect_ratio"][i] for i in inds])
                    if self.objects > 1 and self.thresh_type.startswith("bw"):
                        self.fulldat.setdefault("consmerged",[]).extend([self.conlist["consmerged"][i] for i in inds])

                    # --- Live jump filtering: remove impossible jumps per ID ---
                    coms = [self.conlist["com"][i] for i in inds]
                    filtered_coms, self.last_valid = filter_tracking_jumps(
                        ids, coms, self.frame_nr, self.last_valid, max_framedist=self.max_framedist,
                        pr_comm=self.pr_comm, mask=self.img_mask)

                    # --- Dynamic shape filter: reject contours that deviate from per-ID rolling area ---
                    areas = [self.conlist["area"][i] for i in inds]
                    if self.size_filter:
                        _area_tol = getattr(self.config.track, 'size_filter_tol', getattr(self.config.track, 'shape_area_tol', 0.25))
                        shape_accept = filter_contour_shape(
                            ids, areas, self.frame_nr, self.shape_history, area_tol=_area_tol, pr_comm=self.pr_comm)
                        for i, ok in enumerate(shape_accept):
                            if not ok:
                                filtered_coms[i] = (np.nan, np.nan)
                    _history_len = getattr(self.config.track, 'size_filter_memory', getattr(self.config.track, 'shape_history_len', 500))
                    update_shape_history(ids, areas, filtered_coms, self.shape_history, history_len=_history_len)

                    # Collect overlay info for this thresh type
                    if create_vid or self.config.vis.show_tracking:
                        aspects = [self.conlist["aspect_ratio"][i] for i in inds]
                        # Contours that failed processcon (valid area but no COM assigned)
                        excl = [(self.conlist["area"][j], self.conlist["aspect_ratio"][j])
                                for j in range(len(self.conlist["com"]))
                                if not isinstance(self.conlist["com"][j], tuple)
                                and self.conlist["area"][j] > 0]
                        excl.sort(reverse=True)
                        _frame_info.append(f"cons:{len(self.allcons)} id:{len(inds)}")
                        for i, id in enumerate(ids):
                            tag = "" if np.isfinite(filtered_coms[i][0]) else " [-]"
                            _frame_info.append(f"ID{id}: area={areas[i]:.0f} ar={aspects[i]:.2f}{tag}")
                        if excl:
                            parts = [f"{a:.0f}/{ar:.1f}" for a, ar in excl[:3]]
                            _frame_info.append(f"excl: {', '.join(parts)}")

                    self.fulldat.setdefault("cx", []).extend([c[0] for c in filtered_coms])
                    self.fulldat.setdefault("cy", []).extend([c[1] for c in filtered_coms])

                    # Store head/tail coordinates if not simple
                    if self.advanced:
                        heads = [self.conlist["head"][i] for i in inds]
                        tails = [self.conlist["tail"][i] for i in inds]
                        self.fulldat.setdefault("hx", []).extend([h[0] if (h is not None and h == h) else np.nan for h in heads])
                        self.fulldat.setdefault("hy", []).extend([h[1] if (h is not None and h == h) else np.nan for h in heads])
                        self.fulldat.setdefault("tx", []).extend([t[0] if (t is not None and t == t) else np.nan for t in tails])
                        self.fulldat.setdefault("ty", []).extend([t[1] if (t is not None and t == t) else np.nan for t in tails])
                        self.fulldat.setdefault("angle", []).extend([self.conlist["angle"][i] for i in inds])
                    elif self.orientfrombw:
                        # If orientfrombw, also extend angle column
                        self.fulldat.setdefault("angle", []).extend([self.conlist["angle"][i] for i in inds])

                    # Add data to movedat
                    self.prevlist = self.conlist
                    allids = np.unique(ids+list(self.movedat.keys())) if self.thresh_type.startswith("bw") else ids
                    for i,id in enumerate(ids):
                        self.movedat.setdefault(id, {})
                        self.movedat[id].setdefault("frame", deque(maxlen=self.keep_frames)).appendleft(self.frame_nr)
                        com = filtered_coms[i]  # NaN if jump was rejected, so linkIDs falls back to last real position
                        self.movedat[id].setdefault("com", deque(maxlen=self.keep_frames)).appendleft(com)
                        self.movedat[id].setdefault("vel", deque(maxlen=self.keep_frames)).appendleft(getavgvel(self.movedat[id]["com"]))
                        head = points_to_angle(self.movedat[id]["com"][-1],self.movedat[id]["com"][0],flip=True)
                        self.movedat[id].setdefault("head", deque(maxlen=self.keep_frames)).appendleft(head)
                        #self.movedat[id]["contour"] = self.conlist["contour"][inds[next(i for i,id in enumerate(ids))]] if id in ids else []

                    # Update per-ID trajectory deques (O(1) per fish, avoids O(N) fulldat scans in drawing)
                    for i, id in enumerate(ids):
                        tracked_ids.add(id)
                        cx, cy = filtered_coms[i]
                        if np.isfinite(cx):
                            if id not in traj_history:
                                traj_history[id] = deque(maxlen=self.traj_length)
                            traj_history[id].appendleft((cx, cy))

                    # Thresh_type drawing
                    #---------------------------------
                    if self.config.track.create_vid or self.config.vis.show_tracking:
                        for key in self.conlist.keys():
                            self.conlist[key] = [self.conlist[key][i] for i in inds]
                        vis.draw_thresh_pass(
                            self.img_draw, self.img, self.img_thresh,
                            self.conlist, self.allcons, ids, filtered_coms,
                            traj_history, tracked_ids, self.thresh_type)

                # Linking of bw and color threshtypes
                if self.threshcolors and self.orientfrombw:
                    vis.draw_orient_link(self.img_draw, self.fulldat, self.frame_nr)

                if create_vid or self.config.vis.show_tracking:
                    vis.draw_scene_overlays(self.img_draw)

                # Final drawing
                #---------------------------------
                img_mask_draw = self.img_mask if isinstance(self.maskimg, str) else None
                vis.draw_info_overlay(self.img_draw, img_mask_draw, self.frame_nr, _frame_info)

                # Write video to file
                if self.config.track.create_vid:
                    frame_rgb = cv2.cvtColor(self.img_draw, cv2.COLOR_BGR2RGB)
                    even_w, even_h = make_even(self.vidw), make_even(self.vidh)
                    if (frame_rgb.shape[1], frame_rgb.shape[0]) != (even_w, even_h):
                        frame_rgb = cv2.resize(frame_rgb, (even_w, even_h))
                    self.vidout.write(frame_rgb)

                # Display video
                if self.config.vis.show_tracking:
                    w = int(self.config.vis.vid_displaysize * self.vidw)
                    h = int(self.config.vis.vid_displaysize * self.vidh)
                    cv2.imshow(self.filename, cv2.resize(self.img_draw, (w, h)))
                    key = cv2.waitKey(self.config.vis.waitkey) & 0xff
                    if key == 27 or key == ord('s'):
                        break

                if key == 27 or key == ord('s'):
                    break

            # Tracking of video is finished
            #---------------------------------
            self.exit()

            if key == 27:
                shutil.move(tempfile, self.trackfile)
                try:
                    os.remove(trackedfile)
                except:
                    pass
                lineprint("\nUser quit and escaped, video put back and output deleted")
            else:
                if create_dat:
                    finaldat = pd.DataFrame(self.fulldat)
                    finaldat.to_csv(trackeddata, index=False)
                if create_vid and os.path.exists(trackedfile):
                    shutil.move(trackedfile, os.path.join(self.dirs["tracked"], filename+self.suffix+"_TR.mp4"))
                else:
                    lineprint(self.pr_comm + "Tracked video not created, skipping move.")
                shutil.move(tempfile, os.path.join(self.dirs["originals"], self.filebase))
                if key == ord('s'):
                    lineprint("User quit and saved, tracking output stored")

            self.tracked += 1
            timediff = time.time() - t1
            speed = str(round((self.frame_nr-self.frame_start)/float(timediff),1))
            removeline()
            left_msg = f"; {len(self.inds)} left.." if self.pools < 2 else ".."
            lineprint(self.pr_comm+"tracking completed in "+"%.2f" % timediff+"s at "+speed+"fps; "+left_msg)
            if self.fulldat.get("id") and self.fulldat.get("area"):
                summary_df = pd.DataFrame({"id": self.fulldat["id"], "area": self.fulldat["area"],
                                           "aspect_ratio": self.fulldat["aspect_ratio"],
                                           "cx": self.fulldat["cx"]})
                valid = summary_df[summary_df["cx"].notna() & summary_df["id"].notna()]
                if len(valid) > 0:
                    stats = valid.groupby("id").agg({"area": "median", "aspect_ratio": "median"})
                    parts = [f"ID{i}: area={r['area']:.0f} ar={r['aspect_ratio']:.2f}"
                             for i, r in stats.iterrows()]
                    lineprint(self.pr_comm + "  " + ", ".join(parts))

        except KeyboardInterrupt:
            raise KeyboardInterruptError()
        except Exception as e:
            lineprint(f"Error on [{self.ind}] {self.filename}: {type(e).__name__}: {e}")
            raise
