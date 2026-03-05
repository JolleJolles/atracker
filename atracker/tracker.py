#! /usr/bin/env python

import os
import cv2
import sys
import time
import shutil
import numpy as np
import pandas as pd
import multiprocessing
from collections import deque
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from pythutils.sysutils import lineprint, Suppressor, removeline
from pythutils.fileutils import listfiles
from pythutils.mediautils import check_media, crop
from pythutils.drawutils import namedcols, draw_text, draw_traj, uniqcols

from .utils import *
from .process_image import ProcessImage

class KeyboardInterruptError(Exception): pass

class Tracker:

    def __init__(self, pools, inds, trackfiles, dirs, overview, config,
                 threshinfo, start, stop, custhreshtypes, cusobjects,
                 checkconschange, suffix, max_framedist=200, overwrite=None):

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
        self.simple = self.config.track.simple
        self.orientfrombw = self.config.track.orientfrombw
        self.linkdisthreshold = 100 if "linkdisthreshold" not in self.config.track else self.config.track.linkdisthreshold
        self.mergedmindist = 20 if "mergedmindist" not in self.config.track else self.config.track.mergedmindist
        self.tracked = 0
        self.custhreshtypes = custhreshtypes
        self.cusobjects = cusobjects
        self.checkconschange = checkconschange
        self.max_framedist = max_framedist
        self.overwrite = overwrite if overwrite is not None else self.config.track.overwrite

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
        self.pt1, self.pt2 = literal_eval(self.roi)
        self.vidw  = self.pt2[0] - self.pt1[0]
        self.vidh =  self.pt2[1] - self.pt1[1]
        self.frame_start = self.frame_start if self.frame_start==self.frame_start else self.config.track.startframe
        self.frame_start = self.ustart if self.ustart is not None else self.frame_start
        self.frame_stop = self.frame_stop if self.frame_stop==self.frame_stop else int(self.fcount)
        self.frame_stop = self.ustop if self.ustop is not None else self.frame_stop
        self.fps = self.fps if hasattr(self, "fps") and pd.notnull(self.fps) and self.fps > 0 else self.config.exp.fps

        # Check if all threshtypes are in threshinfo
        ## check_threshtypes turns empty cells (which are nan) into "bw" threshtype
        nopass = False
        if self.custhreshtypes is not None:
            self.thresh_types = check_threshtypes(self.custhreshtypes)
        else:
            self.thresh_types = check_threshtypes(self.thresh_types)
        for t in self.thresh_types:
            if t not in self.threshinfo:
                lineprint(self.pr_comm + f" threshtype '{t}' not in threshinfo.. Skipping this file.")
                return self.inds  # <-- exit early

        # Skip if already tracked and overwrite is off
        if not self.overwrite:
            tracked_path = os.path.join(self.dirs["tracked"], self.filename + self.suffix + "_TR.mp4")
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

        # Set up video writer
        if self.config.track.create_vid:
            self.vidout = videowriter(self.trackedfile, self.vidw, self.vidh, self.fps)
            try:
                if self.vidout is None or (hasattr(self.vidout, 'isOpened') and not self.vidout.isOpened()):
                    raise ValueError("Video writer failed")
            except Exception:
                self.config.track.create_vid = False
                lineprint(self.pr_comm + "Warning: video writer could not be created. Video output disabled.")

        # Set up video window (optional GUI)
        if self.config.vis.show_tracking and self.pools < 2:
            cv2.namedWindow("X", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("X", 1, 1)
            cv2.namedWindow(self.filename, cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(self.filename, 100, 0)

        # Final parameters and flags
        self.traj_length = int(self.fps * self.config.vis.traj_length)
        self.orcheckwindow = self.fps * self.config.orient.delwindow

        if self.cusobjects is not None:
            self.objects = self.cusobjects
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
        if not any(thresh.startswith("bw") for thresh in self.thresh_types):
            self.thresh_types.insert(0, "bw")

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

        newline = False if self.pools<2 else True
        lineprint(self.pr_comm+"tracking started..", newline=newline)
        sys.stdout.flush()
        t1 = time.time()
        key = None
        self.fulldat = {"frame":[]}
        self.movedat = {}
        self.last_valid = {}

        try:
            while self.cap.isOpened():
                frameOK, self.img = self.cap.read()
                stop, skip, self.frame_nr = framechecks(self.cap, frameOK, None, self.frame_stop,
                                         self.config.vis.frame_disstep, self.ind)
                if stop:
                    break
                if skip:
                    continue

                # Create images to work with
                self.img = crop(self.img, self.pt1, self.pt2)
                self.img_draw = self.img.copy()

                # Get standard conlists
                for t,self.thresh_type in enumerate(self.thresh_types):
                    sys.stdout.flush()
                    ti = self.threshinfo[self.thresh_type]
                    if "blur2" not in ti:
                        ti["blur2"] = 1
                    if self.thresh_type.startswith("bw"):
                        PI = ProcessImage(self.img, self.img_bg, self.img_mask, self.thresh_type,
                            ti["blur"], ti["erode"], ti["blur2"], ti["threshold"], ti["min_area"], ti["max_area"],
                            simple=self.simple)
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
                            simple=True)
                    self.img_thresh, self.allcons, self.conlist = PI.process()

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
                    self.fulldat.setdefault("consmerged",[]).extend([self.conlist["consmerged"][i] for i in inds])

                    # --- Live jump filtering: remove impossible jumps per ID ---
                    coms = [self.conlist["com"][i] for i in inds]
                    filtered_coms, self.last_valid = filter_tracking_jumps(
                        ids, coms, self.frame_nr, self.last_valid, max_framedist=300, max_gap=10, pr_comm=self.pr_comm)
                    self.fulldat.setdefault("cx", []).extend([c[0] for c in filtered_coms])
                    self.fulldat.setdefault("cy", []).extend([c[1] for c in filtered_coms])

                    # Store head/tail coordinates if not simple
                    if not self.simple:
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
                        self.movedat[id].setdefault("frame", deque(maxlen=10)).appendleft(self.frame_nr)
                        com = self.conlist["com"][inds[next(i for i,j in enumerate(ids) if j==id)]] if id in ids else (np.nan,np.nan)
                        self.movedat[id].setdefault("com", deque(maxlen=10)).appendleft(com)
                        self.movedat[id].setdefault("vel", deque(maxlen=10)).appendleft(getavgvel(self.movedat[id]["com"]))
                        head = points_to_angle(self.movedat[id]["com"][-1],self.movedat[id]["com"][0],flip=True)
                        self.movedat[id].setdefault("head", deque(maxlen=10)).appendleft(head)
                        #self.movedat[id]["contour"] = self.conlist["contour"][inds[next(i for i,id in enumerate(ids))]] if id in ids else []

                    # Thresh_type drawing
                    #---------------------------------
                    if self.config.track.create_vid or self.config.vis.show_tracking:

                        # Draw trajectories
                        tframes = list(range(self.frame_nr-self.traj_length+1,self.frame_nr+1))
                        cols = uniqcols(len(np.unique(self.fulldat["id"])))
                        for i,id in enumerate(np.unique(self.fulldat["id"]) if self.thresh_type.startswith("bw") else [self.thresh_type]):
                            idinds = [i for i,j in enumerate(self.fulldat["id"]) if j == id]
                            trajdat = [
                                (self.fulldat["cx"][k], self.fulldat["cy"][k])
                                for k in idinds if self.fulldat["frame"][k] in tframes
                            ]
                            # Filter to only consecutive pairs that are both valid
                            trajdat_valid = [pt for pt in trajdat if (
                                pt is not None and
                                isinstance(pt, (tuple, list)) and
                                all([np.isfinite(x) for x in pt])
                            )]
                            trajcol = namedcols(self.thresh_type) if not self.thresh_type.startswith("bw") else cols[i] if self.config.vis.idcol else eval(self.config.vis.traj_col)
                            if len(trajdat_valid) >= 2:
                                draw_traj(self.img_draw, trajdat_valid, trajcol,
                                      self.config.vis.traj_minthick,
                                      self.config.vis.traj_maxthick,
                                      self.config.vis.traj_opacity)

                        # hide trajectories behind contours
                        if self.config.vis.trajs_below:
                            self.img_draw[self.img_thresh == 255] = self.img[self.img_thresh == 255]

                        # Draw all contours
                        cv2.drawContours(self.img_draw, self.allcons, -1, namedcols("red"), 1)

                        # Subset conlist to ID'ed contours
                        cl = self.conlist
                        for key in cl.keys():
                            cl[key] = [cl[key][i] for i in inds]

                        # Draw contours and centroids within size range
                        col = 128 if not self.thresh_type.startswith("bw") else eval(self.config.vis.contour_col) if not cl["consmerged"] else 128
                        cv2.drawContours(self.img_draw, cl["contour"], -1, col, 1)

                        # Draw more complex information
                        for i,j in enumerate(cl["id"]):
                            #cv2.polylines(self.img_draw, np.array([cl["skeleton"][i]]), False, namedcols("orange"), 1)
                            if cl["skeleton"][i]==cl["skeleton"][i]:
                                self.img_draw = draw_coordlist(self.img_draw, cl["skeleton"][i], namedcols("orange"))
                            if cl["tail"][i]==cl["tail"][i]:
                                cv2.circle(self.img_draw, cl["tail"][i], 0, namedcols("red"), 6)
                            #cv2.circle(self.img_draw, cl["fed"][i], 0, namedcols("yellow"), 6)
                            if cl["head"][i]==cl["head"][i]:
                                arrowtip = get_coord(cl["head"][i][0], cl["head"][i][1], cl["angle"][i], 13, True)
                                cv2.arrowedLine(self.img_draw, cl["head"][i], arrowtip, eval(self.config.vis.orient_col), 1, tipLength = 0.4)
                                cv2.circle(self.img_draw, cl["head"][i], 0, namedcols("lightgreen"), 6)
                            if cl["com"][i]==cl["com"][i]:
                                idcol = namedcols(self.thresh_type) if not self.thresh_type.startswith("bw") else eval(self.config.vis.centre_col)
                                cv2.circle(self.img_draw, cl["com"][i], 0, idcol, self.config.vis.centre_lwidth)
                                if self.thresh_type.startswith("bw"):
                                    draw_text(self.img_draw, str(j), (cl["com"][i][0]-4, cl["com"][i][1]-4), 0.3, "black", 0, 1)

                # Linking of bw and color threshtypes
                if self.threshcolors and self.orientfrombw:
                    currframe_inds = [i for i,f in enumerate(self.fulldat["frame"]) if f==self.frame_nr]
                    ids = [id for i,id in enumerate(self.fulldat["id"]) if i in currframe_inds]

                    if "angle" not in self.fulldat:
                        self.fulldat["angle"] = [None] * len(self.fulldat["frame"])

                    if len(ids) > 0 and (len(ids) == 2*len([i for i in ids if type(i)==str])):
                        # Get coms as tuples from cx, cy
                        coms = [(self.fulldat["cx"][i], self.fulldat["cy"][i]) for i in currframe_inds]
                        colids, colcoms = zip(*[(ids[i], com) for i, com in enumerate(coms) if type(ids[i])==str])
                        bwids, bwcoms = zip(*[(ids[i], com) for i, com in enumerate(coms) if type(ids[i])!=str])
                        dismat = cdist(colcoms, bwcoms)
                        _, bw_inds = linear_sum_assignment(dismat)
                        for i, id in enumerate(colids):
                            angle = int(points_to_angle(colcoms[i], bwcoms[bw_inds[i]], flip=True))
                            target_index = currframe_inds[ids.index(id)]
                            if target_index >= len(self.fulldat["angle"]):
                                self.fulldat["angle"].extend([None] * (target_index - len(self.fulldat["angle"]) + 1))
                            self.fulldat["angle"][target_index] = angle
                            arrowtip = get_coord(colcoms[i][0], colcoms[i][1], angle, 5, True)
                            col = (255,255,255) if id in ["blue","black"] else (0,0,0)
                            cv2.arrowedLine(self.img_draw, colcoms[i], arrowtip, col, self.config.vis.orient_lwidth, tipLength = 0.4)

                # Final drawing
                #---------------------------------
                # Draw the mask
                if isinstance(self.maskimg, str):
                    img_masked = cv2.bitwise_and(self.img_draw, self.img_draw, mask = self.img_mask)
                    cv2.addWeighted(img_masked, self.config.vis.mask_opacity,
                    self.img_draw, 1-self.config.vis.mask_opacity, 0, self.img_draw)

                # Draw framenumber
                draw_text(self.img_draw, str(self.frame_nr), (0,0), 0.8, margin=5, bgcol="white")

                # Write video to file
                if self.config.track.create_vid:
                    frame_rgb = cv2.cvtColor(self.img_draw, cv2.COLOR_BGR2RGB)
                    even_w, even_h = make_even(self.vidw), make_even(self.vidh)
                    if (frame_rgb.shape[1], frame_rgb.shape[0]) != (even_w, even_h):
                        frame_rgb = cv2.resize(frame_rgb, (even_w, even_h))
                    self.vidout.append_data(frame_rgb)

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
                shutil.move(self.tempfile, self.trackfile)
                try:
                    os.remove(self.trackedfile)
                except:
                    pass
                lineprint("\nUser quit and escaped, video put back and output deleted")
            else:
                if self.config.track.create_dat:
                    finaldat = pd.DataFrame(self.fulldat)
                    finaldat.to_csv(self.trackeddata, index=False)
                if self.config.track.create_vid and os.path.exists(self.trackedfile):
                    shutil.move(self.trackedfile, os.path.join(self.dirs["tracked"], self.filename+self.suffix+"_TR.mp4"))
                else:
                    lineprint(self.pr_comm + "Tracked video not created, skipping move.")
                shutil.move(self.tempfile, os.path.join(self.dirs["originals"], self.filebase))
                if key == ord('s'):
                    lineprint("User quit and saved, tracking output stored")

            self.tracked += 1
            timediff = time.time() - t1
            speed = str(round((self.frame_nr-self.frame_start)/float(timediff),1))
            removeline()
            lineprint(self.pr_comm+"tracking completed in "+"%.2f" % timediff+"s at "+speed+"fps; "+str(len(self.inds))+" left..")

        except KeyboardInterrupt:
            raise KeyboardInterruptError()
