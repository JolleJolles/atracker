#! /usr/bin/env python

from __future__ import print_function

import os
import cv2
import yaml
import time
import shutil
import warnings
import numpy as np
import pandas as pd

from box import Box
from random import sample
from ast import literal_eval
from localconfig import LocalConfig
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

from pirecorder.convert import Convert
from pythutils.fileutils import listfiles
from pythutils.sysutils import lineprint
from pythutils.drawutils import namedcols, uniqcols
from pythutils.mediautils import get_vid_params, check_media, crop
from pythutils.datutils import to_query
from pythutils.mathutils import ptsToDist

from atracker.__version__ import __version__
from atracker.bg_extract import bg_extract
from atracker.ivideo import ivideo
from atracker.tracker import Tracker
from atracker.processor import Processor
from atracker.utils import *

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

class ATracker:

    """
    Tracker class for setting up automated video tracking

    Parameters
    ----------
    filedir : str; default = "."
        The directory containing the video files

    Returns
    -------
    ATracker : class; the ATracker class
    """

    def __init__(AT, filedir = "."):

        lineprint("ATracker "+__version__+" started!")
        lineprint("="*50, False)

        if not os.path.exists(filedir):
            raise OSError("Directory does not exist..")
        AT.dir = filedir.rstrip(os.path.sep)

        dirs = ["originals","todo","temp","tracked","processed"]
        dpaths = [os.path.join(AT.dir, str(i)+d) for i,d in enumerate(dirs)]
        AT.dirs = dict(zip(dirs, dpaths))
        if os.path.exists(AT.dirs["todo"]):
            lineprint("Tracking folder loaded |", newline=False)
        else:
            for i in AT.dirs:
                os.makedirs(AT.dirs[i])
            lineprint("Set up tracking folder |", newline=False)

        for file in listfiles(AT.dir, (".h264",".mp4",".MP4",".mov",".m4v")):
            shutil.move(os.path.join(AT.dir, file), AT.dirs["originals"])

        cfiles = ["overview.xlsx","config.conf","treshinfo.yml"]
        fpaths = [os.path.join(AT.dir, f"{os.path.basename(AT.dir)}_{f}") for f in cfiles]
        AT.cfiles = dict(zip([os.path.splitext(f)[0] for f in cfiles], fpaths))

        if os.path.exists(AT.cfiles["overview"]):
            AT.reload()
            print("Overview file loaded", end=" | ")
        else:
            cols = ["video","fps","fcount","resolution","frame_start",
                    "frame_stop","roi","conv","exp","date","trial","session",
                    "setup","ID","bgimg","maskimg","tresh_types",
                    "wallimg","zoneimg","skip","objects","exclude"]
            AT.overview = pd.DataFrame(columns=cols)
            AT.save()
            print("", sep="", end=" | ")

        AT.config = LocalConfig(AT.cfiles["config"], compact_form=True)
        if not os.path.exists(AT.cfiles["config"]):
            print("Configfile not found, new file created", end=" | ")
            for section in ["exp","track","bgextract","orient","vis"]:
                if section not in list(AT.config):
                    AT.config.add_section(section)
            AT.set_config(fps=25, real_dims=None, startframe=1,
                          stopframe=99999, keep_frames=10, bg_frames=25,
                          orient_get=False, orient_delwindow=1,
                          orient_minvel=4, orient_minconvex=0.25,
                          orient_minextdistratio=1.1, show_tracking=True,
                          vid_displaysize=1, frame_disstep=100, userwait=False,
                          idcol=True, simple=True, orientfrombw=False,
                          contour_col="blue", centre_col="white", front_col="black",
                          orient_col="black", traj_col="yellow", centre_lwidth=13,
                          orient_lwidth=2, orient_tip=0.15, orient_length=15,
                          traj_length=4, traj_minthick=6.4, traj_maxthick=9,
                          traj_opacity=0.5, mask_opacity=0.15, box_opacity=0.7,
                          draw_contournrs=False, trajs_below=False, strict=False,
                          create_vid=True, create_dat=True, overwrite=True,
                          internal="", linkdistreshold=100)
            print("Config settings stored", end=" | ")
        else:
            print("Config settings loaded", end=" | ")

        if os.path.exists(AT.cfiles["treshinfo"]):
            with open(AT.cfiles["treshinfo"]) as file:
                AT.treshinfo = yaml.load(file, Loader=yaml.FullLoader)
            print("Treshinfo file loaded")
        else:
            AT.treshinfo = {}
            with open(AT.cfiles["treshinfo"], "w") as file:
                yaml.dump(AT.treshinfo, file, default_flow_style=False)
            print("Treshinfo file created")

        os.chdir(AT.dir)
        odir = os.path.join(AT.dirs["originals"], "")
        AT.vids = [os.path.join(odir, str(v) + ".mp4") for v in AT.overview.video]


    def _name_and_index(AT, vid):

        name,ext = os.path.splitext(vid)
        name = os.path.basename(name)
        ind = AT.overview.index[AT.overview.video == name].tolist()
        if len(ind) == 0:
            maxind = AT.overview.index.max()
            ind = (maxind+1) if (len(AT.overview) > 0) else 0
        else:
            ind = ind[0]

        return name, ind


    def _get_all_inds(AT, query, cats, ind):

        if cats is None:
            qry = query
        else:
            cats = [cats] if type(cats) is not list else cats
            vals = AT.overview.loc[ind, cats].values.tolist()
            nqry = to_query(cats, vals)
            qry = query+' and '+nqry if query is not None else nqry
        inds = AT.overview.query(qry).index.tolist()

        return inds


    def save(AT):
        # saves the overview file to disk
        AT.overview.to_excel(AT.cfiles["overview"], index=False)
        lineprint("Overview stored..")


    def reload(AT):
        # loads the overview file from disk
        AT.conv = {col: str for col in [0]+list(range(8,17))}
        AT.overview = pd.read_excel(AT.cfiles["overview"], converters=AT.conv, engine='openpyxl')


    def showinfo(AT, files=[], zones=False):
            files = [files] if type(files) == str else files
            if zones:
                inds = [list(AT.overview[(AT.overview.video==f[:-3])&(AT.overview.region==int(f[-1:]))].index.values)[0] for f in files]
            else:
                inds = AT.overview.loc[AT.overview["video"].isin(files)]
            return inds


    def get_inds(self, namelist):

        if isinstance(namelist, str):  # If a single string is passed, convert to list
            namelist = [namelist]

        # Get indices for all videos in namelist
        indices = []
        for name in namelist:
            matches = self.overview.index[self.overview["video"] == name].tolist()
            if not matches:  # If a name is not found, return an empty list
                return []
            indices.extend(matches)

        return indices


    def get_files(AT, cdir="originals", inds=None, query=None, cats=None, filetype=".mp4", existonly=False):

        overview = AT.overview

        # Exclude rows where the "exclude" column is 1
        if "exclude" in overview.columns:
            overview = overview[overview["exclude"] != 1]

        if inds is not None:
            inds = [i for i in inds if i <= (len(overview)-1)]
            pass
        else:
            if query is not None:
                overview = overview.query(query)
            if cats is not None:
                cats = [cats] if type(cats) is not list else cats
                for cat in cats:
                    if cat not in AT.overview:
                        raise ValueError(cat+" variable does not exist")
                overview = overview.drop_duplicates(subset=cats)
            inds = overview.index.tolist()
        files = [os.path.join(AT.dirs[cdir], str(v) + filetype) for v in AT.overview.loc[inds].video]

        if existonly:
            output = list(zip(*[(i,f) for i,f in enumerate(files) if os.path.exists(f)]))
            inds, files = ([],[]) if len(output)==0 else map(list,output)

        return inds, files


    def set_regions(AT, inds, nr):

        for ind in inds:
            AT.overview = duplicate_row(AT.overview, ind, nr)
        AT.overview = AT.overview.sort_values(by=["video", "region"])
        AT.save()
        lineprint("Region information added..")


    def set_objects(AT, inds, objects):
        for ind, obj in zip(inds, objects):
            AT.overview.loc[ind, "objects"] = obj
        AT.save()
        lineprint("Objects information added..")

    
    def set_config(AT, **kwargs):

        """
        Dynamically sets the configuration file

        Parameters
        ----------
        regions: bool, default = False
            If regions should be considered
        fps: int, default = 25
            The fps of the recorded videos
        real_dims: tuple, default = None
            The real dimensions of the arena in mm
        startframe: int, default = 1
            The start frame to be used for tracking
        stopframe: int, default = 99999
            The stop frame to be used for tracking
        keep_frames: int, default = 10
            The number of frames to keep for getting previous contour
        strict : bool, default = False
            If contour distance exclusion should be strict or not. Strict
            is for example required when tracking a large object among
            many smaller objects such as a pike among shiners.
        overwrite : bool, default = True
            If tracking data should be overwritten or not
        create_vid : bool, default = True
            If a tracking video should be created
        create_dat : bool, default = True
            If data should be written to file
        simple : bool, default = True
            If only simple contour data or complex contour data (skeleton,
            head, tail, orientation, curvature etc) should be extracted
        orientfrombw : bool, default = False
            If orientation data should be acquired from the difference in 
            centroid and other contour (such as color or barcode) 
        bg_frames : int, default = 25
            The number of frames that should be used to create a background
            image between the start and stopframe
        orient_delwindow : int, default = 4
            The timewindow in seconds in which the orientation angle should
            be checked >>> ! Currently not implemented
        orient_minvel : int, default = 4
            Minimal velocity to update orientation with heading >>> ! Currently not implemented
        orient_minextdistratio : float, default = 1.1
            Minimum ratio between distances from centroid to extreme points to
            use for checking orientation >>> ! Currently not implemented
        orient_minconvex : float, default = 0.25
            Minimal convexity to update orientation >>> ! Currently not implemented
        mergedmindist : int, default = None
            Minimal distance that previous contours should be to a potential
            merged contour as condition for being a merged contour
        linkdistreshold : int, default = 100
            Maximum distance in converted pixels per frame to be used to link
            two IDs during tracking
        show_tracking : boolean, default = True
            If tracking should be shown live
        vid_displaysize : int, default = 1
            Size of the video display window relative to the video size. The
            default of 1 is thus the same as the video dimensions.
        frame_disstep : int, default = 100
            Nr of timesteps at which framenumber should be displayed inline to
            keep up-to-date with the status of tracking
        userwait : boolean, default = False
            If tracking display should wait for user key
        idcol : boolean, default = True
            If ids should have a unique color
        contour_col : str, default = "blue"
            Colour of object contours
        centre_col : str, default = "red"
            Colour of centre points of objects
        front_col : str, default = "black"
            Colour of front points of objects
        orient_col : str, default = "black"
            Colour of orientation arrow
        traj_col : str, default = "yellow"
            Colour of trajectories
        centre_lwidth : int, default = 13
            Line thickness of the object centre points
        orient_lwidth : int, default = 2
            Line thickness of the orientation arrows
        orient_tip : float, default = 0.3
            Width of the tip of the orientation arrows
        orient_length : int, default = 30
            Length of the orientation arrows
        traj_length : float, default = 4
            Delay with which the object trajectories should be
            displayed, in seconds
        traj_minthick : float, default = 6.4
            Minimum thickness of the trajectory
        traj_maxthick : float, default = 9
            Maximum thickness of the trajectory
        traj_opacity : float, default = 0.5
            Opacity of the trajectory
        mask_opacity : float, default = 0.15
            Opacity of the mask layer
        box_opacity : float, default = 0.7
            Opacity of the box displaying tracking information
        draw_contournrs : boolena, default = False
            If the blob contour numbers should be drawn
        """

        if "regions" in kwargs:
            AT.config.exp.regions = kwargs["regions"]
            if kwargs["regions"] and not "region" in AT.overview:
                AT.overview.insert(1, "region", 1)
                AT.save()
            if not kwargs["regions"] and "region" in AT.overview:
                AT.overview.drop("region", axis=1, inplace=True)
                AT.save()

        if "fps" in kwargs:
            AT.config.exp.fps = int(kwargs["fps"])
        if "real_dims" in kwargs:
            AT.config.exp.realdims = kwargs["real_dims"]

        if "startframe" in kwargs:
            AT.config.track.startframe = kwargs["startframe"]
        if "stopframe" in kwargs:
            AT.config.track.stopframe = kwargs["stopframe"]
        if "keep_frames" in kwargs:
            AT.config.track.keep_frames = kwargs["keep_frames"]
        if "strict" in kwargs:
            AT.config.track.strict = kwargs["strict"]
        if "overwrite" in kwargs:
            AT.config.track.overwrite = kwargs["overwrite"]
        if "create_vid" in kwargs:
            AT.config.track.create_vid = kwargs["create_vid"]
        if "create_dat" in kwargs:
            AT.config.track.create_dat = kwargs["create_dat"]
        if "simple" in kwargs:
            AT.config.track.simple = kwargs["simple"]
        if "orientfrombw" in kwargs:
            AT.config.track.orientfrombw = kwargs["orientfrombw"]
        if "mergedmindist" in kwargs:
            AT.config.track.mergedmindist = kwargs["mergedmindist"]
        if "linkdistreshold" in kwargs:
            AT.config.track.linkdistreshold = kwargs["linkdistreshold"]

        if "bg_frames" in kwargs:
            AT.config.bgextract.bg_frames = kwargs["bg_frames"]

        if "orient_get" in kwargs:
            AT.config.orient.get = kwargs["orient_get"]
        if "orient_delwindow" in kwargs:
            AT.config.orient.delwindow = kwargs["orient_delwindow"]
        if "orient_minvel" in kwargs:
            AT.config.orient.minvel = kwargs["orient_minvel"]
        if "minextdistratio" in kwargs:
            AT.config.orient.minextdistratio = kwargs["minextdistratio"]
        if "minconvex" in kwargs:
            AT.config.orient.minconvex = kwargs["minconvex"]

        if "show_tracking" in kwargs:
            AT.config.vis.show_tracking = kwargs["show_tracking"]
        if "vid_displaysize" in kwargs:
            AT.config.vis.vid_displaysize = kwargs["vid_displaysize"]
        if "frame_disstep" in kwargs:
            AT.config.vis.frame_disstep = kwargs["frame_disstep"]
        if "userwait" in kwargs:
            AT.config.vis.waitkey = 0 if kwargs["userwait"] is True else 1
        if "idcol" in kwargs:
            AT.config.vis.idcol = namedcols(kwargs["idcol"])
        if "contour_col" in kwargs:
            AT.config.vis.contour_col = namedcols(kwargs["contour_col"])
        if "centre_col" in kwargs:
            AT.config.vis.centre_col = namedcols(kwargs["centre_col"])
        if "centre_lwidth" in kwargs:
            AT.config.vis.centre_lwidth = kwargs["centre_lwidth"]
        if "orient_col" in kwargs:
            AT.config.vis.orient_col = namedcols(kwargs["orient_col"])
        if "orient_lwidth" in kwargs:
            AT.config.vis.orient_lwidth = kwargs["orient_lwidth"]
        if "orient_tip" in kwargs:
            AT.config.vis.orient_tip = kwargs["orient_tip"]
        if "orient_length" in kwargs:
            AT.config.vis.orient_length = kwargs["orient_length"]
        if "traj_col" in kwargs:
            AT.config.vis.traj_col = namedcols(kwargs["traj_col"])
        if "traj_length" in kwargs:
            AT.config.vis.traj_length = kwargs["traj_length"]
        if "traj_minthick" in kwargs:
            AT.config.vis.traj_minthick = kwargs["traj_minthick"]
        if "traj_maxthick" in kwargs:
            AT.config.vis.traj_maxthick = kwargs["traj_maxthick"]
        if "traj_opacity" in kwargs:
            AT.config.vis.traj_opacity = kwargs["traj_opacity"]
        if "mask_opacity" in kwargs:
            AT.config.vis.mask_opacity = kwargs["mask_opacity"]
        if "box_opacity" in kwargs:
            AT.config.vis.box_opacity = kwargs["box_opacity"]
        if "draw_contournrs" in kwargs:
            AT.config.vis.contnrs = kwargs["draw_contournrs"]
        if "trajs_below" in kwargs:
            AT.config.vis.trajs_below = kwargs["trajs_below"]

        if len(kwargs) > 0:
            AT.config.save()

        if "internal" not in kwargs:
            print("Config settings stored and loaded..")


    def setup_files(AT, fname_extract=True, fname_vars=("date","exp","trial", "session","setup","ID"), fname_sep="-", skip=False, autoconvert=True):

        """Extracts file and video information for tracking"""

        lineprint("Preparing video files for tracking..", end=" ")

        # First check if there are any files that need converting
        convlist = listfiles(AT.dirs["originals"], type=".h264", keepdir=False, keepext=False)
        convedlistcap = listfiles(AT.dirs["originals"], type=".MP4", keepdir=True, keepext=True)
        convedlistcap = convedlistcap+listfiles(AT.dirs["originals"], type=".m4v", keepdir=True, keepext=True)
        if len(convedlistcap)>0:
            for file in convedlistcap:
                os.rename(file, os.path.splitext(file)[0]+".mp4")
        convedlist = listfiles(AT.dirs["originals"], type=".mp4", keepdir=False, keepext=False)
        convlist = [f for f in convlist if f not in convedlist]
        if autoconvert:
            if len(convlist)==0:
                lineprint("No files to convert..", end=" ")
            else:
                Convert(indir=AT.dirs["originals"], outdir=AT.dirs["originals"],
                        fps=AT.config.exp.fps, overwrite=False)

        # Now check if there are any files that need processing
        todovids = listfiles(AT.dirs["originals"], type=".mp4", keepdir=True)
        if skip:
            checklist = listfiles(AT.dirs["originals"], type=".mp4", keepdir=False, keepext=False)
            checklist = [f[0] for f in enumerate(checklist) if f[1] not in list(AT.overview["video"])]
            todovids = [todovids[i] for i in checklist]
        if len(todovids)==0:
            print("No files to prepare..", end=" ")
        else:
            for i,vid in enumerate(todovids):
                name, ind = AT._name_and_index(vid)
                lineprint("Video "+str(i+1)+"|"+str(len(todovids))+" "+name, True, False)
                AT.overview.loc[ind,"video"] = name

                if fname_extract:
                    namevals = name.split(fname_sep)
                    if len(namevals)!=len(fname_vars):
                        raise ValueError("Check fname_vars input or set fname_extract to False..")
                    for j,nameval in enumerate(namevals):
                        AT.overview.loc[ind,fname_vars[j]] = nameval
                    print("Filename vars extracted", end=" ")

                flag = check_media(vid)
                if not flag:
                    continue

                fps, width, height, fcount = get_vid_params(vid)
                AT.overview.loc[ind,"fps"] = fps
                AT.overview.loc[ind,"resolution"] = str((width,height))
                AT.overview.loc[ind,"roi"] = str(((0,0),(width,height)))
                max_pyframe = find_max_working_pyframe(cv2.VideoCapture(vid))
                AT.overview.loc[ind,"fcount"] = max_pyframe+1
                print("Video info extracted")

            AT.save()

  
    def get_bgfiles(AT, inds=[], starts=[], stops=[]):

        """Extracts background file from video"""

        lineprint("Extracting background files..")

        #! Need to improve this code..
        if len(inds)==0:
            checklist = listfiles(AT.dirs["originals"], type=".mp4", keepdir=False, keepext=False)
            vidstocheck = [i for i in checklist if i in list(AT.overview.video[AT.overview["skip"]!=1])]
            bglisttocheck = [list(AT.overview[AT.overview["video"]==v]["bgimg"]) for v in vidstocheck]
            bgidstodo = [len([j for j in i if type(j) is not str]) for i in bglisttocheck]
            todolist = [os.path.join(AT.dirs["originals"], f"{j[1]}.mp4") for j in enumerate(vidstocheck) if bgidstodo[j[0]] >= 1]
        else:
            todolist = [os.path.join(AT.dirs["originals"], f"{AT.overview.video[j]}.mp4") for j in inds]

        if len(todolist)==0:
            print("All files done..", end=" ")
        else:
            for i,vid in enumerate(todolist):
                name, ind = AT._name_and_index(vid)
                lineprint("Video "+str(i+1)+"|"+str(len(todolist))+" "+name, True, False)
                bgname = name+"_bg.jpg"
                start, stop = AT.overview.loc[ind,["frame_start","frame_stop"]]
                start = int(start) if not np.isnan(start) else None
                stop = int(stop) if not np.isnan(stop) else None
                if len(starts)>0:
                    start = starts[0] if len(starts)==1 else starts[i]
                if len(stops)>0:
                    stop = stops[0] if len(stops)==1 else stops[i]
                framenr = AT.config.bgextract.bg_frames
                img_bg = bg_extract(vid, start, stop, framenr)
                cv2.imwrite(os.path.join(AT.dirs["originals"], bgname), img_bg)
                AT.overview.loc[ind, "bgimg"] = bgname

            AT.save()


    def set_interactive(AT, inds=None, framelimits=None, roi=None, mask=None, maskzone=None, zones=None, walls=None, conv=None, getpts=None, conv_mm=None, treshtypes=None, query=None, cats=None, ptcolnames=[], treshfile=None):

        inputs = [framelimits,roi,mask,walls,conv,treshtypes]
        if len(inputs) - inputs.count(None)>1:
            raise ValueError("Choose one input setting..")
        if treshtypes is not None:
            if type(treshtypes) is not list:
                raise valueError("treshtypes should be list..")
        showHelperlines = False
        drawLine = False
        drawRect = False
        drawPoly = False
        drawPt = False
        if framelimits:
            lineprint("Interactive video mode for setting framelimits..")
        elif roi:
            lineprint("Interactive video mode for setting region of interest..")
            showHelperlines = True
            drawRect = True
        elif mask:
            lineprint("Interactive video mode for setting internal mask..")
            showHelperlines = True
            drawPoly = True
        elif maskzone:
            lineprint("Interactive video mode for creating mask for zone")
            showHelperlines = True
            drawPoly = True
        elif zones:
            lineprint("Interactive video mode for creating zones")
            showHelperlines = True
            drawPoly = True
        elif getpts:
            lineprint("Interactive video mode for setting pt coordinates..")
            showHelperlines = True
            drawPt = True
        elif walls:
            lineprint("Interactive video mode for setting walls mask..")
            showHelperlines = True
        elif conv:
            if conv_mm == None:
                if len(AT.config.exp.realdims)==0:
                    print("No real measure provided..")
                    return
                else:
                    conv_mm = literal_eval(AT.config.exp.realdims)
            lineprint("Interactive video mode for setting conversion..")
            drawLine = True
        elif treshtypes is not None:
            lineprint("Interactive video mode for setting tresholds..")
        else:
            lineprint("Nothing to set..")
            return

        inds, vids = AT.get_files("originals", inds, query, cats)

        for i,vid in enumerate(vids):
            name,_ = AT._name_and_index(vid)
            ind = inds[i]

            printext = "Video "+str(i+1)+"|"+str(len(vids))+" "+name
            if "region" in AT.overview:
                printext = printext+" region "+str(AT.overview.loc[ind].region)
                name = name+"_R"+str(AT.overview.loc[ind].region)
            lineprint(printext, True, False)

            if not os.path.isfile(vid):
                print("Does not exist, continuing..")
                continue

            if conv or treshtypes not in [None,[None]]:
                roival = literal_eval(AT.overview.loc[ind]["roi"])
            else:
                roival = ((0,0),literal_eval(AT.overview.loc[ind,"resolution"]))

            treshtypes = [None] if treshtypes is None else treshtypes
            for treshtype in treshtypes:
                if treshtype not in [None, "bw"]:
                    print(treshtype, end=" ")
                    if not treshtype.startswith("bw_"):
                        if type(namedcols(treshtype))!=tuple:
                            return
                firstframe = AT.overview.loc[ind,"frame_start"]
                firstframe = 1 if np.isnan(firstframe) else int(firstframe)
                lastframe = AT.overview.loc[ind,"frame_stop"]
                lastframe = None if np.isnan(lastframe) else int(lastframe)
                fcount = AT.overview.loc[ind,"fcount"]
                lastframe = None if np.isnan(fcount) else int(fcount)
                
                ivid = ivideo(vid, treshtype=treshtype, treshinfo=AT.treshinfo,
                              firstframe=firstframe, lastframe=lastframe,
                              fullrange=framelimits, displaysize=AT.config.vis.vid_displaysize,
                              orientation=AT.config.orient.get,
                              orient_minconvex=AT.config.orient.minconvex,
                              simple=AT.config.track.simple, roi=roival,
                              showHelperlines=showHelperlines, drawLine=drawLine,
                              drawRect=drawRect,drawPoly=drawPoly,drawPt=drawPt)
                ivid.pt1, ivid.pt2 = roival
                if treshtype is not None:
                    bgimg = AT.overview.loc[ind]["bgimg"]
                    if type(bgimg) is float:
                        lineprint("No background image exists, exiting..")
                        ivid.key = 27
                        continue
                    else:
                        bgimg = os.path.join(AT.dirs["originals"], bgimg)
                    ivid.img_bg = cv2.imread(bgimg)
                    ivid.img_bg = crop(ivid.img_bg, ivid.pt1, ivid.pt2)
                    ivid.img_mask = None
                    if isinstance(AT.overview.loc[ind]["maskimg"], str):
                        img_mask = cv2.imread(os.path.join(AT.dirs["originals"], AT.overview.loc[ind]["maskimg"]), 0)
                        kernel = np.ones((5,5),np.uint8)
                        img_mask = cv2.erode(img_mask, kernel)
                        ivid.img_mask = cv2.dilate(img_mask, kernel)
                        ivid.img_mask = crop(ivid.img_mask, ivid.pt1, ivid.pt2)

                ivid.show()

                if chr(ivid.key) == "s":
                    if any([query, cats]):
                        allinds = AT._get_all_inds(query, cats, ind)
                    else:
                        allinds = ind
                    if framelimits:
                        AT.overview.loc[allinds,"frame_start"] = ivid.start_frame
                        AT.overview.loc[allinds,"frame_stop"] = ivid.stop_frame
                    if roi:
                        roi=fix_roi(ivid.m.twoPoint,literal_eval(AT.overview.loc[ind,"resolution"]))
                        AT.overview.loc[allinds,"roi"] = str(roi)
                    if mask:
                        maskname = name+"_mask.jpg"
                        cv2.imwrite(os.path.join(AT.dirs["originals"], maskname), ivid.mask)
                        AT.overview.loc[allinds,"maskimg"] = maskname
                    if maskzone:
                        zonename = name+"_zones.jpg"
                        cv2.imwrite(os.path.join(AT.dirs["originals"], zonename), ivid.mask)
                        AT.overview.loc[allinds,"zoneimg"] = zonename
                    if zones:
                        if len(ivid.coords)>0:
                            AT.zoneimg = np.zeros((ivid.vidh,ivid.vidw,3), np.uint8)+255
                            cols = uniqcols(len(ivid.coords))
                            for i,j in enumerate(ivid.coords):
                                cv2.fillPoly(AT.zoneimg, np.int32([ivid.coords[j]]), cols[i])
                            zonename = name+"_zones.jpg"
                            cv2.imwrite(os.path.join(AT.dirs["originals"], zonename), AT.zoneimg)
                            AT.overview.loc[allinds,"zoneimg"] = zonename
                            lineprint("Zone image stored..", end=" ")
                        else:
                            lineprint("No zone coordinates provided..", end=" ")
                    if walls:
                        maskname = name+"_walls.jpg"
                        cv2.imwrite(os.path.join(AT.dirs["originals"], maskname), ivid.mask)
                        AT.overview.loc[allinds,"wallimg"] = maskname
                    if getpts:
                        if len(ivid.m.pts)>0:
                            if len(ptcolnames) != len(ivid.m.pts):
                                ptcolnames = ["pt"+str(i+1) for i,j in enumerate(ivid.m.pts)]
                            AT.overview.loc[allinds,ptcolnames] = [str(i) for i in ivid.m.pts]
                            lineprint("Point coordinates for "+str(ptcolnames)+" stored.." , end=" ")
                        else:
                            lineprint("No point coordinates stored.." , end=" ")
                    if conv:
                        pixdis = ptsToDist(ivid.m.twoPoint[0],ivid.m.twoPoint[1])
                        if type(conv_mm) != tuple:
                            convdat = str(round(conv_mm/pixdis,4))
                        else:
                            convdat = []
                            for i in range(len(conv_mm)):
                                convdat.append(round(conv_mm[i]/pixdis,4))
                                if i < len(conv_mm)-1:
                                    ivid = ivideo(vid, treshtype=treshtype, treshinfo=AT.treshinfo,
                                                  firstframe=firstframe, lastframe=lastframe,
                                                  fullrange=framelimits, roi=roival,
                                                  drawLine=drawLine)
                                    ivid.show()
                                    pixdis = ptsToDist(ivid.m.twoPoint[0],ivid.m.twoPoint[1])
                            convdat = str(sum(convdat)/len(convdat))
                        lineprint("With and height conversion: "+convdat, end=" ")
                        AT.overview.loc[allinds,"conv"] = convdat
                    if treshtype is not None:
                        AT.treshinfo[treshtype] = ivid.treshinfo
                        treshfile = treshfile if treshfile is not None else AT.cfiles["treshinfo"]
                        treshfile = treshfile+".yml" if ".yml" not in treshfile else treshfile
                        with open(treshfile, 'w') as f:
                            yaml.safe_dump(AT.treshinfo, f, default_flow_style=False)

                if ivid.key == 27:
                    break

            if ivid.key == 27:
                break

        AT.save()


    def drymode(AT, rand_filenr=10, rand_seqnr=5, rand_seqlen=100, suffix="dry", rerun=False):

        a = AT.config.track.overwrite
        b = AT.config.vis.show_tracking
        c = AT.config.vis.frame_disstep
        d = AT.config.vis.trajs_below
        e = AT.config.track.create_vid
        f = AT.config.track.create_dat

        AT.set_config(overwrite=True, show_tracking=False, frame_disstep=99999,
            trajs_below=False, create_vid=True, create_dat=False)

        if not rerun or not hasattr(AT, 'dryinds'):
            fullinds = list(AT.overview.index[AT.overview["skip"]!=1])
            AT.dryinds = sample(fullinds, min(rand_filenr,len(fullinds)-1))

        for ind in AT.dryinds:
            sub = AT.overview.loc[ind]
            if not rerun or not hasattr(AT, 'dryframes'):
                minf = int(sub["frame_start"] if sub["frame_start"]==sub["frame_start"] else 1)
                maxf = int(sub["frame_stop"] if sub["frame_stop"]==sub["frame_stop"] else sub["fcount"])
                startframes = sample(list(range(minf,maxf-rand_seqlen-1)),rand_seqnr)
                AT.dryframes = list(zip(startframes,[i+rand_seqlen for i in startframes]))
            for i,seq in enumerate(AT.dryframes):
                suffixi = suffix+str(i+1).zfill(len(str(len(AT.dryframes))))
                AT.track(folder="originals", inds=[ind], start=seq[0], stop=seq[1], suffix=suffixi)

        AT.set_config(overwrite=a, show_tracking=b, frame_disstep=c,
            trajs_below=d, create_vid=e, create_dat=f)


    def track(AT, inds=None, names=None, query=None, cats=None, pools=1, folder="todo", start=None,
        stop=None, custreshtypes=None, cusobjects=None, checkconschange=False, suffix="", treshfile=None):
       
        if treshfile is not None:
            try:
                with open(treshfile, "r") as f:
                    AT.treshinfo = yaml.load(f, Loader=yaml.FullLoader)
                    print(f"Loading custom treshfile '{treshfile}'")
            except FileNotFoundError:
                raise FileNotFoundError(f"Treshfile '{treshfile}' not found.")
            except Exception as e:
                raise RuntimeError(f"Error loading treshfile '{treshfile}': {e}")
        else:
            with open(AT.cfiles["treshinfo"], 'r') as f:
                AT.treshinfo = yaml.load(f, Loader=yaml.FullLoader)

        if names is not None:
            inds = AT.get_inds(names)
        
        if inds is not None:
            if "exclude" in AT.overview.columns:
                inds = [i for i in inds if AT.overview.loc[i, "exclude"] != 1]
            trackfiles = [os.path.join(AT.dirs[folder], f"{video}.mp4") 
                          for video in AT.overview.loc[inds, "video"]]
        else:
            inds, trackfiles = AT.get_files(folder, inds, query, cats, existonly=False)

        # Fix Localconfig messing up the class variables
        cbak = AT.config
        del AT.config
        AT.config = Box({s: {k:v for (k,v) in cbak.items(s)} for s in cbak})

        # Set up Tracker instance
        T = Tracker(pools, inds, trackfiles, AT.dirs, AT.overview,
                    AT.config, AT.treshinfo, start, stop, custreshtypes, cusobjects, checkconschange, suffix)
        lineprint("Tracking started of "+str(len(trackfiles))+" files..")

        if pools<2:
            counter = -1
            try:
                while len(T.inds)>0:
                    counter += 1
                    ind = T.inds[0]
                    trackfile = os.path.join(AT.dirs[folder], AT.overview.loc[ind]["video"] + ".mp4")
                    T.setuptracking(ind, trackfile)
                lineprint("Tracking finished..")
            except KeyboardInterrupt:
                lineprint("\nUser terminated tracking..")
        else:
            AT.config.vis.show_tracking = False
            AT.config.vis.waitkey = 1
            def callback_function(output): T.inds = output
            if not notebook():
                pool = multiprocessing.Pool(min(pools, len(trackfiles)))
                counter = -1
                try:
                    while len(T.inds)>0:
                        counter += 1
                        ind = T.inds[0]
                        trackfile = os.path.join(AT.dirs[folder], AT.overview.loc[ind]["video"] + ".mp4")
                        tempool = [pool.apply_async(T.setuptracking,
                                                    (ind,trackfile),
                                                    callback=callback_function)]
                        time.sleep(0.2)  #rather than sleep try the tempool.wait() function
                    [i.get() for i in tempool]
                    pool.close()
                except KeyboardInterrupt:
                    lineprint("\nUser terminated tracking pool..")
                    pool.terminate()
                except Exception as e:
                    lineprint("Got exception: %r, terminating pool" % (e,))
                    pool.terminate()
                    lineprint("pool is terminated")
                finally:
                    pool.join()
                    print("Tracking completed..")
            else:
                lineprint("Pooled tracking can only be run from the terminal, exiting..")
        AT.config = cbak


    def _pworker(AT, trackedfile, config_dict):
        """Each worker creates its own Processor instance and processes the file."""
        thread_id = threading.get_ident()
        P = Processor(**config_dict)  # No pickling issues with threads
        P.setup(trackedfile, thread_id)  
    
    def process(AT, pools=1, names=None, overwrite=False, fulldata=True, convert=True, 
                removeoutliers=True, alonewindow=5, manfix=False, man_vidresizeval=0.5, man_types=["c"], 
                man_customstep=100, manonly=True, man_frameloc=0, man_timewindow=0, man_treshold_speed=0,
                smoothwin=10, changefps=None, addIDs=True, nearmaskdis=20, trajgap = 50, edgedis=10, 
                filllentresh_com=500, inmaskdis=10, mintrajlength=10, filllentresh_headtail=40, 
                headtoorientspeedtresh=1, fillmissingorientdifftresh=100, 
                centertype = None, centralise=False, delcontdata=False, powermate=False):
        
        """
        Runs data processing

        Custom Parameters
        ----------
        pools : int; default = 1
            The number of pools to create for parallel processing.
        names : list; default = None
            A list of files to process. Will lookin the "3tracked" folder.
        overwrite : bool; default = False
            If the output file should be overwritten if it already exists.
        fulldata : bool; default = True
            If missing tracking data should be extended for the full time series 
            from the start to stop frame.
        convert : bool; default = True
            If the data should be converted from pixels to mm, using the conv setting.
        removeoutliers : bool; default = True
            If automatically spatial outliers should be removed.
        alonewindow : int; default = 5
            The time window in frames used for considering data to be outliers or not 
            for the "removeoutliers" function.

        manfix : bool; default = False
            If the interactive manual tracking and fixing function should be run or not.
        man_vidresizeval : float; default = 0.5
            A value to show a resized version of the original video, 1 being 100%
        man_types : list of ["c","h","t"]; default = ["c"]
            One or multiple point locations that need to be tracked, can be "c",
            coordinate in the centre of the oject; "f", tip coordinate on the front of 
            the oject; "b", bottom coordinate on the back of the object.
        man_customstep : int; default = None
            Custom (forward) step size in frames, for moving through the video with keypress.
        manonly : bool; default = True
            If manfix, if no additional processing should be run beyond manual fixing.
        man_frameloc : int; default = 0
            The pyframe that should be shown when starting mantrack.
        man_timewindow : int; default = 0
            The default timewindow for which tracking data should be shown when starting mantrack.
        man_treshold_speed : int; default = 0
            The detaulf speed treshold in mm beyond which data will be displayed differently.
        
        smoothwin : int; default = 10
            The window in frames to use for smoothing the data.
        changefps : int, default = None
            To subset the framerate of the video to a lower value, e.g. to help reduce the filesize.
        addIDs : bool; default = None
            If ID information should be added from the overview file to the datafile.
        nearmaskdis : int, default = 20
            The distance from the mask in pixels. Parameter used for checking if object is under/near
            or away from the mask, used for interpolation functions.
        trajgap : int, default = 50
            Distance in frames at which point missing coordinate data should be split in separate 
            trajectories.
        edgedis : int; default = 10
            The distance from the roi in pixels at which the object is considered to be on the edge. 
            Parameter for removing data outside of the region of interest.
        filllentresh_com : int; default = 500
            Maximum distance in frames between coordinate data outside of any potential mask that
            should be filled-in by differentiating.
        inmaskdis : int; default = 10
            Distance in pixels from mask that should be considered to be under the mask and
            therefore removed. Different from nearmaskdis, which is only used for determining
            the frames used for interpolating.
        mintrajlength : int; default = 10
            Minimum length of frames for a trajectory. Trajectories shorted than this are deleted.  
        filllentresh_headtail : int, default = 40
            Maximum difference in frames for missing head and tail coordinate data outside of any 
            potential mask that shoudl be filled-in y differentiating.
        headtoorientspeedtresh : float; default = 1 
            The speed value in converted units above which heading should be used to set orientation.
        fillmissingorientdifftresh : int; default = 100
            Maximum difference in frames for missing orientation vector data outside of any potential
            mask that should be filled-in by differentiating.

        centertype : str in [None, "walls","roi","pt"]; default = "walls"
            What type of data should be used to compute the arena center
        centralise : bool; default = False
            If the data should be centralised or not, based on the centertype
        delcontdata : bool; default = False
            If contour data of bw contours should be removed or not
        """

        # Get the list of files to process and normalise for windows compatibility
        trackedfiles = listfiles(AT.dirs["tracked"], type=".csv", keepdir=True)
        trackedfiles = [os.path.normpath(file) for file in trackedfiles]

        # If no names are provided, use base names from trackedfiles
        if names is None:
            # Extract base names without directory or extension from trackedfiles
            base_names = [os.path.splitext(os.path.basename(file))[0] for file in trackedfiles]
        else:
            # If custom names are provided, create a list of base names (without directory and without extension)
            if isinstance(names, str):
                names = [names]
            base_names = [os.path.splitext(os.path.basename(name))[0] for name in names]

        # Create a set to store selected files and avoid duplicates
        selected_files = set()

        for base_name in base_names:
            # Check if _E.csv version exists for the current base name
            e_file = os.path.normpath(os.path.join(AT.dirs["tracked"], base_name + "_E.csv"))
            original_file = os.path.normpath(os.path.join(AT.dirs["tracked"], base_name + ".csv"))
            
            # Prefer the _E.csv file if it exists, otherwise use the original .csv file
            if e_file in trackedfiles:
                selected_files.add(e_file)  # Add _E file
            elif original_file in trackedfiles:
                selected_files.add(original_file)  # Add original file if no _E file

        # Convert set to a list for final output
        trackedfiles = sorted(list(selected_files))
        if len(trackedfiles) == 0:
            lineprint("No files to fix..")
            return

        config_dict = {  # No need for Manager.dict()
            "dirs": AT.dirs,
            "config": AT.config,
            "overview": AT.overview,
            "trackedfiles": trackedfiles,
            "orientfrombw": AT.config.track.orientfrombw,
            "overwrite": overwrite,
            "addIDs": addIDs,
            "fulldata": fulldata,
            "removeoutliers": removeoutliers,
            "alonewindow": alonewindow,
            "manfix": manfix,
            "changefps": changefps,
            "nearmaskdis": nearmaskdis,
            "inmaskdis": inmaskdis,
            "trajgap": trajgap,
            "edgedis": edgedis,
            "convert": convert,
            "smoothwin": smoothwin,
            "centertype": centertype,
            "centralise": centralise,
            "filllentresh_com": filllentresh_com,
            "filllentresh_headtail": filllentresh_headtail,
            "man_vidresizeval": man_vidresizeval,
            "man_types": man_types,
            "man_customstep": man_customstep,
            "manonly": manonly,
            "mintrajlength": mintrajlength,
            "fillmissingorientdifftresh": fillmissingorientdifftresh,
            "headtoorientspeedtresh": headtoorientspeedtresh,
            "man_frameloc": man_frameloc,
            "man_timewindow": man_timewindow,
            "man_treshold_speed": man_treshold_speed,
            "delcontdata": delcontdata,
            "powermate": powermate
        }

        lineprint("Processing started of " + str(len(trackedfiles)) + " files..")

        if pools<2:
            P = Processor(**config_dict)
            for trackedfile in trackedfiles:
                P.setup(trackedfile, None)
            print("Processing completed..")
        else:
            if not notebook():
                with ThreadPoolExecutor(max_workers=pools) as executor:
                    executor.map(lambda f: AT._pworker(f, config_dict), trackedfiles)
            else:
                lineprint("Pooled processing can only be run from the terminal, exiting..")  