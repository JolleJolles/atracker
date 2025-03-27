#! /usr/bin/env python

from __future__ import print_function

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
from pythutils.mediautils import check_media, crop, videowriter
from pythutils.drawutils import namedcols, draw_text, draw_traj, uniqcols

from .utils import *
from .imgprocessor import ImgProcessor

class KeyboardInterruptError(Exception): pass

class Tracker:

    def __init__(self, pools, inds, trackfiles, dirs, overview, config,
                 treshinfo, start, stop, custreshtypes, cusobjects,
                 checkconschange, suffix):

        self.pools = pools
        self.inds = inds
        self.trackfiles = trackfiles
        self.dirs = dirs
        self.overview = overview
        self.config = config
        self.treshinfo = treshinfo
        self.ustart = start
        self.ustop = stop
        self.suffix = "" if len(suffix)==0 else "_"+suffix
        self.simple = self.config.track.simple
        self.orientfrombw = self.config.track.orientfrombw
        self.linkdistreshold = 100 if "linkdistreshold" not in self.config.track else self.config.track.linkdistreshold
        self.mergedmindist = 20 if "mergedmindist" not in self.config.track else self.config.track.mergedmindist
        self.tracked = 0
        self.custreshtypes = custreshtypes
        self.cusobjects = cusobjects
        self.checkconschange = checkconschange

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
            regions = self.overview[self.overview.video==self.filebasezero]["region"]
            if len(regions)>1:
                self.region = self.overview.loc[ind]["region"]
                self.filename = self.filename+"_R"+str(self.region)
        conv = self.overview.loc[ind]["conv"]
        conv = 1 if conv!=conv else float(conv)
        self.linkdistreshold = self.linkdistreshold / conv
        self.trackedfile = os.path.join(self.dirs["temp"], self.filename+self.suffix+"_TR.mp4")
        self.trackeddata = os.path.join(self.dirs["tracked"], self.filename+self.suffix+".csv")

        pr_procid = multiprocessing.current_process()._identity
        pr_procid = "{P"+str(pr_procid[0])+"}" if len(pr_procid)>0 else ""
        pr_ind = "["+str(self.ind)+"] "
        self.pr_comm = pr_procid+pr_ind+self.filename+" "

        # Copy file information to tracker instance
        # ! Here we also get the self.tresh_types from the overview
        fileinfo = self.overview.loc[ind].squeeze()
        for name, values in fileinfo.iteritems():
            values = int(values) if type(values) == np.int64 else values
            self.__dict__.update([(name, values)])
        self.pt1, self.pt2 = literal_eval(self.roi)
        self.vidw  = self.pt2[0] - self.pt1[0]
        self.vidh =  self.pt2[1] - self.pt1[1]
        self.frame_start = self.frame_start if self.frame_start==self.frame_start else self.config.track.startframe
        self.frame_start = self.ustart if self.ustart is not None else self.frame_start
        self.frame_stop = self.frame_stop if self.frame_stop==self.frame_stop else int(self.fcount)
        self.frame_stop = self.ustop if self.ustop is not None else self.frame_stop

        # Check if all treshtypes are in treshinfo
        ## check_treshtypes turns empty cells (which are nan) into "bw" treshtype
        nopass = False
        if self.custreshtypes is not None:
            self.tresh_types = check_treshtypes(self.custreshtypes)
        else:
            self.tresh_types = check_treshtypes(self.tresh_types)
        for t in self.tresh_types:
            if t not in self.treshinfo:
                lineprint(self.pr_comm+" treshtype "+t+" not in treshinfo..")
                nopass = True

        # Check file existance and status
        if not self.config.track.overwrite:
            if os.path.exists(os.path.join(self.dirs["tracked"], self.filename+self.suffix+"_TR.mp4")):
                lineprint(self.pr_comm+" already tracked, skipping..")
                nopass = True
        if not nopass:
            with Suppressor():
                self.mediaok = check_media(self.trackfile, internal=True)
            if not self.mediaok:
                if os.path.exists(self.tempfile):
                    if not os.path.exists(self.trackedfile):
                        self.inds = self.inds+[self.ind]
                    nopass = True
                else:
                    lineprint(self.pr_comm+"video cannot be found, skipping..")
                    nopass = True
            if type(self.bgimg) == float:
                lineprint(self.pr_comm+"bgimg cannot be found, skipping..")
                nopass = True

        if not nopass:
            if self.config.track.overwrite:
                try:
                    os.remove(self.trackedfile)
                    os.remove(self.trackeddata)
                except:
                    pass
            shutil.move(self.trackfile, self.tempfile)

            # Set up video and load bg and mask images
            self.cap = cv2.VideoCapture(self.tempfile)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_start-1)
            self.img_bg = cv2.imread(os.path.join(self.dirs["originals"], self.bgimg))
            self.img_bg = crop(self.img_bg, self.pt1, self.pt2)
            self.img_mask = None
            if isinstance(self.maskimg, str):
                img_mask = cv2.imread(os.path.join(self.dirs["originals"], self.maskimg), 0)
                img_mask = crop(img_mask, self.pt1, self.pt2)
                kernel = np.ones((5,5),np.uint8)
                self.img_mask = cv2.erode(img_mask, kernel)
                self.img_mask = cv2.dilate(self.img_mask, kernel)
                self.img_mask = cv2.resize(self.img_mask, (self.vidw,self.vidh), interpolation = cv2.INTER_AREA)

            # Set up video writer
            if self.config.track.create_vid:
                self.vidout = videowriter(self.trackedfile, self.vidw, self.vidh, self.config.exp.fps)

            # Set up video window
            if self.config.vis.show_tracking and self.pools<2:
                cv2.namedWindow("X", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("X", 1,1)
                cv2.namedWindow(self.filename, cv2.WINDOW_AUTOSIZE)
                cv2.moveWindow(self.filename, 100, 0)

            # Set some final parameters
            self.traj_length = int(self.fps * self.config.vis.traj_length)
            self.orcheckwindow = self.fps * self.config.orient.delwindow
            if self.cusobjects is not None:
                self.objects = self.cusobjects
            else:
                if "objects" not in self.overview:
                    self.objects = 1
                else:
                    self.objects = list(self.overview.loc[self.overview.video==self.filebasezero,"objects"])[0]
            if self.objects != self.objects:
                self.objects = 1
            else:
                self.objects = int(self.objects)
            self.treshcolors = False
            if len(self.tresh_types)>1:
                self.objects = len(self.tresh_types)
                self.treshcolors = True
            if not any(tresh.startswith("bw") for tresh in self.tresh_types):
               self.tresh_types.insert(0, "bw")

            # Start tracking
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
            self.vidout.release()
        self.cap.release()


    def conschange(self):

        # Did the number of contours decrease?
        contourchange = len(self.conlist["icom"]) / len(self.prevlist["id"])

        # Is there one much larger contour?
        maxareachange = max(self.conlist["area"]) / max(self.prevlist["area"])

        # We have a potential merge, but check locations
        if contourchange < 1 and (maxareachange > 1.5 or 1 in self.prevlist["consmerged"]):
            maxsize = max(self.conlist["area"])
            self.curr_ind = [i for i,j in enumerate(self.conlist["area"]) if j==maxsize][0]
            curr_loc = [self.conlist["icom"][self.curr_ind]]
            # get list of distances of the merged contour to all contours the previous frame
            dists = cdist(np.array(self.prevlist["icom"]), np.array(curr_loc))
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
                elif key == "icom":
                    self.conlist[key].append(concom(newcon))
                elif key == "area":
                    self.conlist[key].append(cv2.contourArea(newcon))
                else:
                    self.conlist[key].append(np.nan)


    def linkIDs(self):

        if self.tresh_type.startswith("bw"):
            if len(self.movedat) == 0 or self.treshcolors:
                IDsfinal = [i if i in range(self.objects) else np.nan for i in range(len(self.conlist["com"]))]
            else:
                bwmovedat = {k: v for k, v in self.movedat.items() if type(k) in [int,np.int64]}
                previds = list(bwmovedat.keys())
                coms = []
                for id in bwmovedat:
                    nonnans = [(i,j) for i,j in enumerate(bwmovedat[id]["com"]) if j[0]==j[0]]
                    if len(nonnans)>0:
                        lastvalidind,com = nonnans[0]
                        coms += [com]
                    else:
                        previds.remove(id)
                # divide the distance treshold for linking by the number of frames
                distreshold = int(self.linkdistreshold * (self.frame_nr - bwmovedat[id]["frame"][lastvalidind]))
                if len(coms)>0:
                    # cdist provides a distance matrix between all the distances of
                    # two lists of points (lets say A and B). The output is an array
                    # of lists with each row being an indice of A and each column an
                    # indice of B. So with three positions in A and two in B the matrix
                    # is 3 lists of 2, whereby the distance between the 2nd point of A
                    # and the 1st point of B will be dismat[1][0]
                    dismat = cdist(np.array(coms), np.array(self.conlist["icom"]))
                    # the linear sum assignment finds for each point in B the closest
                    # point in A
                    IDs_ind, curr_allocated = linear_sum_assignment(dismat)
                    curr_allocated = [j for i,j in enumerate(curr_allocated) if dismat[IDs_ind[i]][j] < distreshold]
                    IDs = [previds[i] for i in IDs_ind]
                else:
                    IDs = []
                    curr_allocated = []
                IDsfinal = [np.nan]*len(self.conlist["icom"])
                IDs_available = [i for i in range(int(self.objects)) if i not in IDs]
                for i,j in enumerate(self.conlist["icom"]):
                    # go through all newpoints and if point is in allocated IDs
                    if i in curr_allocated:
                        # get the ID corresponding to the current point by looking at the (first)
                        # instance where its index occurs in the allocation list
                        IDsfinal[i] = IDs[next(k for k,l in enumerate(curr_allocated) if l==i)]
                    else:
                        if len(IDs_available)>0:
                            IDsfinal[i] = IDs_available[0]
                            IDs_available = IDs_available[1:]
                        else:
                            IDsfinal[i]  = np.nan
        else:
            if self.tresh_type not in self.movedat:
                IDsfinal = [self.tresh_type] + [np.nan] * (len(self.conlist["com"])-1)
            else:
                nonnans = [(i,j) for i,j in enumerate(self.movedat[self.tresh_type]["com"]) if j[0]==j[0]]
                if len(nonnans)>0:
                    lastvalidind,com = nonnans[0]
                    dismat = cdist(np.array([com]), np.array(self.conlist["icom"]))
                    _,curr_allocated = linear_sum_assignment(dismat)
                else:
                    curr_allocated = []
                IDsfinal = [np.nan]*len(self.conlist["icom"])
                IDsfinal[curr_allocated[0]] = self.tresh_type

        self.conlist["id"] = IDsfinal


    def tracksingle(self):

        newline = False if self.pools<2 else True
        lineprint(self.pr_comm+"tracking started..", newline=newline)
        sys.stdout.flush()
        t1 = time.time()
        key = None
        self.fulldat = {"frame":[]}
        self.movedat = {}

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
                for t,self.tresh_type in enumerate(self.tresh_types):
                    sys.stdout.flush()
                    ti = self.treshinfo[self.tresh_type]
                    if "blur2" not in ti:
                        ti["blur2"] = 1
                    if self.tresh_type.startswith("bw"):
                        IP = ImgProcessor(self.img, self.img_bg, self.img_mask, self.tresh_type,
                            ti["blur"], ti["erosion"], ti["blur2"], ti["treshold"], ti["min_area"], ti["max_area"],
                            simple=self.simple)
                    else:
                        IP = ImgProcessor(self.img, self.img_bg, self.img_mask, self.tresh_type,
                            ti["blur"], min_area=ti["min_area"], max_area=ti["max_area"],
                            colmin=literal_eval(ti["colmin"]), colmax=literal_eval(ti["colmax"]),
                            simple=True)
                    self.img_tresh, self.allcons, self.conlist = IP.process()

                    # Check for merged contours and link IDs over time
                    self.conlist["consmerged"] = [0]*len(self.conlist["com"])
                    if self.tresh_type.startswith("bw"):
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
                    inds,ids = dic_exclnan(self.conlist,"id")
                    self.fulldat["frame"].extend([self.frame_nr]*len(ids))
                    self.fulldat.setdefault("id",[]).extend(ids)
                    vars = ["area","com","icom","consmerged"]
                    if self.orientfrombw:
                        vars = vars + ["angle"]
                    if not self.simple:
                            vars = vars + [key for key in ["head", "tail", "angle"] if key not in vars]
                    for key in vars:
                        self.fulldat.setdefault(key,[]).extend([self.conlist[key][i] for i in inds])

                    # Add data to movedat
                    self.prevlist = self.conlist
                    allids = np.unique(ids+list(self.movedat.keys())) if self.tresh_type.startswith("bw") else ids
                    for i,id in enumerate(ids):
                        self.movedat.setdefault(id, {})
                        self.movedat[id].setdefault("frame", deque(maxlen=10)).appendleft(self.frame_nr)
                        com = self.conlist["icom"][inds[next(i for i,j in enumerate(ids) if j==id)]] if id in ids else (np.nan,np.nan)
                        self.movedat[id].setdefault("com", deque(maxlen=10)).appendleft(com)
                        self.movedat[id].setdefault("vel", deque(maxlen=10)).appendleft(getavgvel(self.movedat[id]["com"]))
                        head = points_to_angle(self.movedat[id]["com"][-1],self.movedat[id]["com"][0],flip=True)
                        self.movedat[id].setdefault("head", deque(maxlen=10)).appendleft(head)
                        #self.movedat[id]["contour"] = self.conlist["contour"][inds[next(i for i,id in enumerate(ids))]] if id in ids else []

                    # Tresh_type drawing
                    #---------------------------------
                    if self.config.track.create_vid or self.config.vis.show_tracking:

                        # Draw trajectories
                        tframes = list(range(self.frame_nr-self.traj_length+1,self.frame_nr+1))
                        cols = uniqcols(len(np.unique(self.fulldat["id"])))
                        for i,id in enumerate(np.unique(self.fulldat["id"]) if self.tresh_type.startswith("bw") else [self.tresh_type]):
                            idinds = [i for i,j in enumerate(self.fulldat["id"]) if j == id]
                            trajdat = [self.fulldat["icom"][k] for k in idinds if self.fulldat["frame"][k] in tframes]
                            trajcol = namedcols(self.tresh_type) if not self.tresh_type.startswith("bw") else cols[i] if self.config.vis.idcol else eval(self.config.vis.traj_col)
                            draw_traj(self.img_draw, trajdat, trajcol,
                                      self.config.vis.traj_minthick,
                                      self.config.vis.traj_maxthick,
                                      self.config.vis.traj_opacity)

                        # hide trajectories behind contours
                        if self.config.vis.trajs_below:
                            self.img_draw[self.img_tresh == 255] = self.img[self.img_tresh == 255]

                        # Draw all contours
                        cv2.drawContours(self.img_draw, self.allcons, -1, namedcols("red"), 1)

                        # Subset conlist to ID'ed contours
                        cl = self.conlist
                        for key in cl.keys():
                            cl[key] = [cl[key][i] for i in inds]

                        # Draw contours and centroids within size range
                        col = 128 if not self.tresh_type.startswith("bw") else eval(self.config.vis.contour_col) if not cl["consmerged"] else 128
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
                            if cl["icom"][i]==cl["icom"][i]:
                                idcol = namedcols(self.tresh_type) if not self.tresh_type.startswith("bw") else eval(self.config.vis.centre_col)
                                cv2.circle(self.img_draw, cl["icom"][i], 0, idcol, self.config.vis.centre_lwidth)
                                if self.tresh_type.startswith("bw"):
                                    draw_text(self.img_draw, str(j), (cl["icom"][i][0]-4, cl["icom"][i][1]-4), 0.3, "black", 0, 1)

                # Linking of bw and color treshtypes
                if self.treshcolors:
                    currframe_inds = [i for i,f in enumerate(self.fulldat["frame"]) if f==self.frame_nr]
                    ids = [id for i,id in enumerate(self.fulldat["id"]) if i in currframe_inds]
                    if  len(ids)>0 and (len(ids) == 2*len([i for i in ids if type(i)==str])):
                        coms = [com for i,com in enumerate(self.fulldat["com"]) if i in currframe_inds]
                        colids,colcoms = zip(*[(ids[i],com) for i,com in enumerate(coms) if type(ids[i])==str])
                        bwids,bwcoms = zip(*[(ids[i],com) for i,com in enumerate(coms) if type(ids[i])!=str])
                        dismat = cdist(colcoms, bwcoms)
                        _, bw_inds = linear_sum_assignment(dismat)
                        for i,id in enumerate(colids):
                            angle = int(points_to_angle(colcoms[i], bwcoms[bw_inds[i]], flip=True))
                            self.fulldat["angle"][currframe_inds[ids.index(id)]] = angle
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
                    self.vidout.write(self.img_draw)

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
                if self.config.track.create_vid:
                    shutil.move(self.trackedfile, os.path.join(self.dirs["tracked"], self.filename+self.suffix+"_TR.mp4"))
                #movedir = self.dirs["todo"] if self._check_tomove() else self.dirs["originals"]
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
