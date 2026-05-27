#! /usr/bin/env python

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
import subprocess
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

from pythutils.fileutils import listfiles
from pythutils.sysutils import lineprint
from pythutils.drawutils import namedcols
from pythutils.mediautils import get_vid_params, check_media
from pythutils.datutils import to_query

from atracker.__version__ import __version__
from atracker.visual_editor import annotation_gui
from atracker.tracker import Tracker
from atracker.post_processor import Processor
from atracker.media import convert_h264_to_mp4
from atracker.utils import *

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')


def _git_version_info():
    """Return short git hash and commit date, or None if not in a git repo."""
    try:
        pkg_dir = os.path.dirname(__file__)
        result = subprocess.run(
            ["git", "log", "-1", "--format=%h %cd", "--date=short"],
            capture_output=True, text=True, timeout=2, cwd=pkg_dir
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split()
            return f"{parts[1]} ({parts[0]})"
    except Exception:
        pass
    return None


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

        git_info = _git_version_info()
        version_str = f"ATracker {__version__}" + (f" — {git_info}" if git_info else "")
        lineprint(version_str + " started!")
        lineprint("="*50, False)

        # Use the working directory of the notebook if filedir is "."
        if filedir == ".":
            filedir = os.getcwd()

        # Normalize the path for any OS and input format
        filedir = os.path.abspath(os.path.normpath(filedir.strip('"')))

        if not os.path.exists(filedir):
            raise OSError("Directory does not exist..")
    
        AT.dir = filedir.rstrip(os.path.sep)

        # Set up subdirectories
        dirs = ["originals","todo","temp","tracked","processed"]
        dpaths = [os.path.join(AT.dir, str(i)+d) for i,d in enumerate(dirs)]
        AT.dirs = dict(zip(dirs, dpaths))
        
        if os.path.exists(AT.dirs["todo"]):
            lineprint("Tracking folder loaded |", newline=False)
        else:
            for i in AT.dirs:
                os.makedirs(AT.dirs[i])
            lineprint("Set up tracking folder |", newline=False)

        # Move video files
        for file in listfiles(AT.dir, (".h264",".mp4",".MP4",".mov",".m4v")):
            shutil.move(os.path.join(AT.dir, file), AT.dirs["originals"])

        # Set up config file paths
        basename = os.path.basename(AT.dir.rstrip(os.path.sep)) or "tracking"
        cfiles = ["overview.xlsx", "config.conf", "threshinfo.yml"]
        fpaths = [os.path.join(AT.dir, f"{basename}_{f}") for f in cfiles]
        AT.cfiles = dict(zip([os.path.splitext(f)[0] for f in cfiles], fpaths))

        if os.path.exists(AT.cfiles["overview"]):
            AT.reload()
            print("Overview file loaded", end=" | ")
        else:
            cols = ["video","fps","fcount","resolution","frame_start",
                    "frame_stop","roi","conv","exp","date","trial","session",
                    "setup","ID","bgimg","maskimg","thresh_types",
                    "wallimg","zoneimg","objects","exclude"]
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
                          show_tracking=True, vid_displaysize=1, frame_disstep=100,
                          userwait=False, idcol=True, simple=True, orientfrombw=False,
                          contour_col="blue", centre_col="white", front_col="black",
                          orient_col="black", traj_col="yellow", centre_lwidth=13,
                          orient_lwidth=2, orient_tip=0.15, orient_length=15,
                          traj_length=4, traj_minthick=6.4, traj_maxthick=9,
                          traj_opacity=0.5, mask_opacity=0.15, box_opacity=0.7,
                          draw_contournrs=False, trajs_below=False, strict=False,
                          create_vid=True, create_dat=True, overwrite=True,
                          shape_area_tol=0.25, shape_history_len=500, linkdisthreshold=100, internal="")
            print("Config settings stored", end=" | ")
        else:
            print("Config settings loaded", end=" | ")

        if os.path.exists(AT.cfiles["threshinfo"]):
            with open(AT.cfiles["threshinfo"]) as file:
                AT.threshinfo = yaml.load(file, Loader=yaml.FullLoader)
            print("Threshinfo file loaded")
        else:
            AT.threshinfo = {}
            with open(AT.cfiles["threshinfo"], "w") as file:
                yaml.dump(AT.threshinfo, file, default_flow_style=False)
            print("Threshinfo file created")

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
        AT.overview.to_excel(AT.cfiles["overview"], index=False)
        lineprint("Overview stored..")

    def reload(AT):
        AT.conv = {col: str for col in [0]+list(range(8,17))}
        AT.overview = pd.read_excel(AT.cfiles["overview"], converters=AT.conv, engine='openpyxl')
        AT.overview["date"] = AT.overview["date"].astype(str).str[:10]
        AT.config = LocalConfig(AT.cfiles["config"], compact_form=True)

        # Normalise column names: merge legacy "Exclude" and "skip" into "exclude"
        ov = AT.overview
        if "exclude" not in ov.columns:
            ov["exclude"] = np.nan
        if "Exclude" in ov.columns:
            ov.loc[ov["Exclude"] == 1, "exclude"] = 1
            ov.drop(columns=["Exclude"], inplace=True)
        if "skip" in ov.columns:
            ov.loc[ov["skip"] == 1, "exclude"] = 1
            ov.drop(columns=["skip"], inplace=True)
        AT.overview = ov

    def showinfo(AT, files=None, inds=None, return_inds=False):
        ov = AT.overview
        if inds is not None:
            out = ov.loc[inds]
        elif files is not None:
            if isinstance(files, str):
                files = [files]
            names = [os.path.splitext(os.path.basename(f))[0] for f in files]
            vids = ov["video"].astype(str).apply(lambda v: os.path.splitext(os.path.basename(v))[0])
            out = ov[vids.isin(names)]
        else:
            raise ValueError("Provide either files or inds.")
        
        inds_list = list(out.index)
        if return_inds:
            return out, inds_list
        else:
            return out

    def get_inds(self, namelist):
        if isinstance(namelist, str):
            namelist = [namelist]

        indices = []

        for name in namelist:
            region = None

            # --- Detect region notation video_R2 ---
            if "_R" in name and name.split("_R")[-1].isdigit():
                base, reg = name.rsplit("_R", 1)
                region = int(reg)
                video_name = base
            else:
                video_name = name

            # --- Select rows for this video ---
            df = self.overview[self.overview["video"] == video_name]

            if df.empty:
                return []

            if region is None:
                # Default = region 1 if it exists
                if "region" in df.columns:
                    reg_index = df.index[df["region"] == 1]
                    if len(reg_index) == 0:
                        raise ValueError(f"No region 1 for {video_name}")
                    indices.extend(reg_index.tolist())
                else:
                    indices.extend(df.index.tolist())

            else:
                # Select explicit region
                if "region" not in df.columns:
                    raise ValueError(f"{video_name} has no region column")
                reg_index = df.index[df["region"] == region]
                if len(reg_index) == 0:
                    raise ValueError(f"No region {region} for {video_name}")
                indices.extend(reg_index.tolist())

        return indices

    def get_files(
        AT, 
        cdir="originals", 
        inds=None, 
        query=None, 
        cats=None, 
        filetype=".mp4", 
        existonly=False, 
        full=True, 
        show_extension=True
    ):
        overview = AT.overview

        # # Exclude rows where the "exclude" column is 1
        # if "exclude" in overview.columns:
        #     overview = overview[overview["exclude"] != 1]

        # Select indices
        if inds is not None:
            inds = [i for i in inds if i <= (len(overview) - 1)]
        else:
            if query is not None:
                overview = overview.query(query)
            if cats is not None:
                cats = [cats] if not isinstance(cats, list) else cats
                for cat in cats:
                    if cat not in AT.overview:
                        raise ValueError(cat + " variable does not exist")
                overview = overview.drop_duplicates(subset=cats)
            inds = overview.index.tolist()
        
        # Get filenames (with or without extension)
        videos = AT.overview.loc[inds].video
        if show_extension:
            filenames = [str(v) + filetype for v in videos]
        else:
            filenames = [str(v) for v in videos]

        if full:
            files = [os.path.join(AT.dirs[cdir], f) for f in filenames]
        else:
            files = filenames

        if existonly:
            output = list(zip(*[(i, f) for i, f in zip(inds, files) if os.path.exists(f)]))
            inds, files = ([], []) if len(output) == 0 else map(list, output)

        return inds, files

    def update_overview(AT, column, value, video=None, inds=None, save=True):
        """
        Inline replacement for overview DataFrame, by video name(s) or indices.
        """
        if (video is None) and (inds is None):
            raise ValueError("Provide at least one of 'video' or 'inds'")
        
        if video is not None:
            if isinstance(video, str):
                row_selector = AT.overview["video"] == video
            elif isinstance(video, (list, tuple)):
                row_selector = AT.overview["video"].isin(video)
            else:
                raise TypeError("'video' must be str or list/tuple of str")
        elif inds is not None:
            row_selector = AT.overview.index.isin(inds)
        else:
            raise ValueError("No valid selection method.")
        
        AT.overview.loc[row_selector, column] = value
        if save:
            AT.save()
            
    def set_regions(AT, inds, nr):

        for ind in inds:
            AT.overview = duplicate_row(AT.overview, ind, nr)
        AT.overview = AT.overview.sort_values(by=["video", "region"])
        AT.save()
        lineprint("Region information added..")

    def set_objects(AT, inds=None, objects=None):
        if objects is None:
            raise ValueError("The 'objects' parameter cannot be None.")
        if inds is None:
            AT.overview["objects"] = objects
        else:
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
        contour_mode : str, default = "static"
            Contour consistency mode. "static" uses only the global area/aspect
            thresholds set in the threshold file (current behaviour). "dynamic"
            additionally applies a per-ID rolling area check: once an ID has 10
            accepted frames of history, detections whose area falls below 25% or
            above 400% of that ID's rolling median are rejected as noise.
        orientfrombw : bool, default = False
            If orientation data should be acquired from the difference in 
            centroid and other contour (such as color or barcode) 
        bg_frames : int, default = 25
            The number of frames that should be used to create a background
            image between the start and stopframe
        mergedmindist : int, default = None
            Minimal distance that previous contours should be to a potential
            merged contour as condition for being a merged contour
        linkdisthreshold : int, default = 100
            Maximum distance in converted pixels per frame to be used to link
            two IDs during tracking
        shape_area_tol : float, default = 0.25
            Ratio threshold for the dynamic per-ID area consistency filter (contour_mode="dynamic").
            A detection is accepted if its area is within [tol × median, (1/tol) × median].
            E.g. 0.25 accepts areas between 25% and 400% of the rolling median.
        shape_history_len : int, default = 500
            Number of accepted frames used to compute the rolling median area per ID.
            500 frames at 25fps = 20 seconds of history, giving a stable baseline.
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
        if "contour_mode" in kwargs:
            AT.config.track.contour_mode = kwargs["contour_mode"]
        if "orientfrombw" in kwargs:
            AT.config.track.orientfrombw = kwargs["orientfrombw"]
        if "mergedmindist" in kwargs:
            AT.config.track.mergedmindist = kwargs["mergedmindist"]
        if "linkdisthreshold" in kwargs:
            AT.config.track.linkdisthreshold = kwargs["linkdisthreshold"]
        if "shape_area_tol" in kwargs:
            AT.config.track.shape_area_tol = kwargs["shape_area_tol"]
        if "shape_history_len" in kwargs:
            AT.config.track.shape_history_len = kwargs["shape_history_len"]

        if "bg_frames" in kwargs:
            AT.config.bgextract.bg_frames = kwargs["bg_frames"]

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

    def setup_files(AT, fname_extract=True, fname_vars=("date", "exp", "trial", "session", "setup", "ID"),
                    fname_sep="-", skip=False, autoconvert=True, fps=None):
        """
        Prepares video files for tracking by converting, extracting metadata, 
        and updating the overview.
        """
        lineprint("Preparing video files for tracking..", end=" ")

        originals_dir = AT.dirs["originals"]

        # Detect which .h264 files still need to be converted
        convlist = []
        for file in listfiles(originals_dir, type=".h264", keepdir=False, keepext=True):
            base = os.path.splitext(file)[0]
            if not os.path.exists(os.path.join(originals_dir, base + ".mp4")):
                convlist.append(file)

        if autoconvert:
            if convlist:
                lineprint(f"Converting {len(convlist)} files...", newline=False)
                conversion_fps = fps if fps is not None else AT.config.exp.fps
                convert_h264_to_mp4(originals_dir, fps=conversion_fps)
            else:
                lineprint("No files to convert..")

        # Build list of all video files, preferring .mp4 over .h264 if both exist
        mp4_files = {os.path.splitext(os.path.basename(f))[0]: f
                    for f in listfiles(originals_dir, type=".mp4", keepdir=True)}
        h264_files = {os.path.splitext(os.path.basename(f))[0]: f
                    for f in listfiles(originals_dir, type=".h264", keepdir=True)}
        # Merge: .mp4 preferred
        all_vids = dict(h264_files)
        all_vids.update(mp4_files)  # mp4 overwrites h264 with same base name
        todovids = list(all_vids.values())
        totalvids = len(todovids)

        if skip:
            existing = set(os.path.splitext(v)[0] for v in AT.overview["video"].dropna())
            todovids = [v for v in todovids if os.path.splitext(os.path.basename(v))[0] not in existing]

        if not todovids:
            print("No files to prepare..", end=" ")
            return

        expected_n = len(fname_vars)

        for i, vid in enumerate(todovids):
            name, ind = AT._name_and_index(vid)
            lineprint(f"Video {i+1}|{totalvids} {name}", True, False)
            AT.overview.loc[ind, "video"] = name

            if fname_extract:
                namevals = name.split(fname_sep)
                version_val = ""
                # Check if last part is a version string (_vXX)
                if len(namevals) > expected_n and re.match(r"v\d{2}$", namevals[-1]):
                    version_val = namevals.pop()
                if len(namevals) != expected_n:
                    print(f"Filename: {name} splits into {len(namevals)} parts, expected {expected_n}")
                    raise ValueError("Check fname_vars input or set fname_extract to False..")
                for j, nameval in enumerate(namevals):
                    AT.overview.loc[ind, fname_vars[j]] = nameval
                AT.overview.loc[ind, "vidseq"] = version_val  # Always adds this column
                print("Filename vars extracted", end=" ")

            # Check and extract video info
            if not check_media(vid):
                continue

            cap = cv2.VideoCapture(vid)
            fps, width, height, fcount = get_vid_params(cap)
            AT.overview.loc[ind, "fps"] = fps
            AT.overview.loc[ind, "resolution"] = str((width, height))
            AT.overview.loc[ind, "roi"] = str(((0, 0), (width, height)))
            max_pyframe = find_max_working_pyframe(cap)
            AT.overview.loc[ind, "fcount"] = max_pyframe + 1
            print("Video info extracted")

        AT.save()

    def get_bgfiles(AT, inds=[], starts=[], stops=[], overwrite=False):

        lineprint("Extracting background files..")

        # 1) Determine which rows to process
        if len(inds) == 0:
            # All rows not excluded
            df = AT.overview[AT.overview.get("exclude", pd.Series(dtype=object)) != 1]
            rows = df.index.tolist()
        else:
            rows = inds

        # 2) Build a todo-list per row (i.e. per region)
        todolist = []
        for idx in rows:
            video = AT.overview.loc[idx, "video"]
            region = AT.overview.loc[idx].get("region", None)

            # unique bg filename
            if region is None:
                bgname = f"{video}_bg.jpg"
            else:
                bgname = f"{video}_R{region}_bg.jpg"

            bgpath = os.path.join(AT.dirs["originals"], bgname)

            # we will extract per row
            if overwrite or not os.path.isfile(bgpath):
                todolist.append((idx, video, region, bgname, bgpath))
        
        # 3) Extract backgrounds
        if len(todolist) == 0:
            print("All files done..", end=" ")
        else:
            for k, (idx, video, region, bgname, bgpath) in enumerate(todolist):

                vidpath = os.path.join(AT.dirs["originals"], f"{video}.mp4")
                name, _ = AT._name_and_index(vidpath)
                lineprint(f"Video {k+1}|{len(todolist)} {name}", True, False)

                # --- Get region-specific start/stop ---
                start, stop = AT.overview.loc[idx, ["frame_start", "frame_stop"]]
                start = int(start) if not np.isnan(start) else None
                stop = int(stop) if not np.isnan(stop) else None

                # Override start/stop if provided manually
                if starts:
                    start = starts[0] if len(starts) == 1 else starts[k]
                if stops:
                    stop = stops[0] if len(stops) == 1 else stops[k]

                framenr = AT.config.bgextract.bg_frames

                img_bg = bg_extract(vidpath, start, stop, framenr)
                cv2.imwrite(bgpath, img_bg)

                # Update ONLY this row
                AT.overview.loc[idx, "bgimg"] = bgname

            AT.save()


    def set_interactive(AT, inds=None, framelimits=None, roi=None, mask=None, maskzone=None, zones=None,
                        walls=None, conv=None, getpts=None, conv_mm=None, threshtypes=None,
                        query=None, cats=None, ptcolnames=None, threshfile=None, events=False):
        """
        Interactive mode for various tasks, including event annotation.

        Args:
            events (bool): If True, enables event annotation mode.
        """
        # Validate input
        input_flags = [framelimits, roi, mask, maskzone, zones, walls, conv, getpts, threshtypes, events]
        active_modes = sum(x is not None and x is not False for x in input_flags)

        if active_modes == 0:
            mode = "default"
        elif active_modes == 1:
            if conv: mode, true_mode = "measure", "measure"
            elif framelimits: mode, true_mode = "framelimits", "framelimits"
            elif roi: mode, true_mode = "roi", "roi"
            elif mask: mode, true_mode = "mask", "mask"
            elif walls: mode, true_mode = "mask", "walls"
            elif maskzone: mode, true_mode = "mask", "maskzone"
            elif zones: mode, true_mode = "zones", "zones"
            elif getpts: mode, true_mode = "points", "points"
            elif events: mode, true_mode = "events", "events"
        else:
            raise ValueError("Please specify exactly one interactive mode (e.g., framelimits=True).")

        inds, vids = AT.get_files("originals", inds, query, cats)
        fileaction = "overwrite"
        datafile = None
        overview_dirty = False

        for i, ind in enumerate(inds):
            allinds = AT._get_all_inds(query, cats, ind) if query or cats else ind
            vid = os.path.join(AT.dirs["originals"], f"{AT.overview.loc[ind, 'video']}.mp4")
            name, _ = AT._name_and_index(vid)

            if "region" in AT.overview.columns and not pd.isna(AT.overview.loc[ind, "region"]):
                region = AT.overview.loc[ind, "region"]
                name += f"_R{int(region)}"

            lineprint(f"\nVideo {i + 1} of {len(vids)} — {name}", True, False)

            if not os.path.isfile(vid):
                print("Video file does not exist, skipping..")
                continue

            # Read ROI if present, otherwise fallback later where needed
            if "roi" in AT.overview.columns and isinstance(AT.overview.loc[ind].get("roi", None), str):
                roival = literal_eval(AT.overview.loc[ind]["roi"])
            else:
                # if resolution present, use it as full-frame ROI
                try:
                    res = literal_eval(AT.overview.loc[ind, "resolution"])
                    roival = ((0, 0), res)
                except Exception:
                    roival = None

            firstframe = AT.overview.loc[ind, "frame_start"]
            firstframe = 1 if pd.isna(firstframe)or str(firstframe).strip() == "" else int(firstframe)
            lastframe = AT.overview.loc[ind, "frame_stop"]
            lastframe = None if pd.isna(lastframe) or str(lastframe).strip() == "" else int(lastframe)

            bgimg = AT.overview.loc[ind].get("bgimg", None)
            bgpath = os.path.join(AT.dirs["originals"], bgimg) if isinstance(bgimg, str) else None

            maskpath = None
            mask_column = None
            if mask: mask_column = "maskimg"
            elif maskzone: mask_column = "zoneimg"
            elif walls: mask_column = "wallimg"
            elif zones: mask_column = "zoneimg"
            elif threshtypes: mask_column = "maskimg"
            if mask_column:
                maskfile_entry = AT.overview.loc[ind].get(mask_column, None)
                if isinstance(maskfile_entry, str):
                    maskpath = os.path.join(AT.dirs["originals"], maskfile_entry)
                    if os.path.isfile(maskpath):
                        lineprint(f"Loaded {mask_column} file: {maskfile_entry}")
                    else:
                        print(f"{mask_column} file listed but not found: {maskpath}")

            # ===== Handle multi-threshold interaction mode =====
            if threshtypes:
                print(f"Video {i + 1} of {len(vids)} — {name}", end="")

                for j, ttype in enumerate(threshtypes):
                    mode = "thresholding" if ttype.lower().startswith("bw") else "thresholding color"
                    print(f" | Thresholding: {ttype}", end="")
                    result = annotation_gui(
                        media_file=vid,
                        background_file=bgpath,
                        mask_file=maskpath,
                        mode=mode,
                        threshold_dict=AT.threshinfo.get(ttype, {}),
                        firstframe=firstframe,
                        lastframe=lastframe,
                        fileaction=fileaction,
                        data_file=datafile
                    )
                    if result is None or result == "exit":
                        print(" — exited")
                        break
                    if isinstance(result[1], dict) and len(result[1]) > 0:
                        AT.threshinfo[ttype] = result[1]
                        print(f" stored...", end=" ")
                        # Save threshold info to file
                        finalfile = threshfile
                        if not finalfile:
                            finalfile = AT.cfiles.get("threshinfo", "threshinfo.yml")
                        if not finalfile.endswith(".yml"):
                            finalfile += ".yml"
                        with open(finalfile, "w") as f:
                            yaml.safe_dump(AT.threshinfo, f, default_flow_style=False)
                    else:
                        print(f" → no values")
                continue  # Skip all other interactive modes

            # ======= Event Annotation Mode =======
            if mode == "events":
                events_list = []  # new value will overwrite existing
                print(f"Video {i + 1} of {len(vids)} — {name} Event annotation mode")
                print("Press event key (K) to record frames, L to remove last, S to finish, Esc/Q to abort.")
                while True:
                    result = annotation_gui(
                        media_file=vid,
                        background_file=bgpath,
                        mask_file=maskpath,
                        mode="timepoints",
                        firstframe=firstframe,
                        lastframe=lastframe,
                        fileaction=None,
                        data_file=None
                    )

                    # user requested quit from GUI
                    if result == "exit":
                        # Save overview if changed
                        if overview_dirty:
                            AT.save()
                            print("Overview stored..")
                        print("Exiting event annotation mode.")
                        return

                    # remove any accidental CSV the GUI may have written
                    try:
                        csv_path = os.path.splitext(vid)[0] + ".csv"
                        if os.path.isfile(csv_path):
                            os.remove(csv_path)
                    except Exception:
                        pass

                    # Accept only explicit ("events", [frames]) from the GUI.
                    if isinstance(result, tuple) and len(result) == 2 and result[0] == "events":
                        frames = result[1] or []
                        frames = sorted(set(int(f) for f in frames))
                        if not frames:
                            # no frames recorded -> re-open GUI
                            continue

                        # Pair frames into tuples in order: (f0,f1),(f2,f3),... last singleton if odd
                        grouped = []
                        for j in range(0, len(frames), 2):
                            if j + 1 < len(frames):
                                grouped.append((int(frames[j]), int(frames[j + 1])))
                            else:
                                grouped.append((int(frames[j]),))

                        # Overwrite events column for these rows
                        AT.overview.loc[allinds, "events"] = str(grouped)
                        overview_dirty = True
                        # single-line confirmation
                        print("Event frames recorded: " + " ".join(str(f) for f in frames))
                        break  # done with this video

                    # otherwise ignore and re-open GUI
                    continue

                # next video
                continue

            # ======= Handle all other modes =======
            result = annotation_gui(
                media_file=vid,
                background_file=bgpath,
                mask_file=maskpath,
                mode=mode,
                firstframe=firstframe,
                lastframe=lastframe,
                fileaction=fileaction,
                data_file=datafile
            )

            if result is None:
                continue
            if result == "exit":
                break

            # ----- Measure / conversion -----
            if mode == "measure":
                # Helper to extract pixel length from the annotation GUI result
                def _result_pxlen(res):
                    # Expecting either: ("line"/"polygon", px_len) or ("polygon", points, px_len, ...)
                    if res is None:
                        return None
                    if isinstance(res, tuple):
                        # common older format: ("line", px_len)
                        if len(res) > 1 and isinstance(res[1], (int, float)):
                            return float(res[1])
                        # newer format seen: ("polygon", points, px_len, ...)
                        if len(res) > 2 and isinstance(res[2], (int, float)):
                            return float(res[2])
                        # fallback: if second element is a sequence of two points, compute euclidean distance
                        if len(res) > 1 and isinstance(res[1], (list, tuple)) and len(res[1]) >= 2:
                            p0 = res[1][0]
                            p1 = res[1][1]
                            try:
                                dx = float(p0[0]) - float(p1[0])
                                dy = float(p0[1]) - float(p1[1])
                                return float((dx*dx + dy*dy) ** 0.5)
                            except Exception:
                                return None
                    # unknown shape
                    return None

                px_len = _result_pxlen(result)
                if px_len is None:
                    print("Could not determine pixel length from GUI result:", result)
                    continue

                if isinstance(conv_mm, (int, float)):
                    # Single value, single interaction
                    AT.overview.loc[allinds, "conv"] = round(conv_mm / px_len, 4)
                    lineprint(f"Conversion set to {AT.overview.loc[ind, 'conv']} mm/pixel", end=" ")

                elif isinstance(conv_mm, (list, tuple)):
                    conv_vals = []

                    for j, mm in enumerate(conv_mm):
                        print(f"Draw conversion line {j+1} of {len(conv_mm)} ({mm} mm)")
                        if j > 0:
                            result = annotation_gui(
                                media_file=vid,
                                background_file=bgpath,
                                mask_file=maskpath,
                                mode=mode,
                                firstframe=firstframe,
                                lastframe=lastframe,
                                fileaction=fileaction,
                                data_file=datafile
                            )
                            if result is None or result == "exit":
                                break
                        px_len = _result_pxlen(result)
                        if px_len is None:
                            print("Could not determine pixel length from GUI result:", result)
                            continue

                        conv_val = mm / px_len
                        conv_vals.append(conv_val)
                        lineprint(f"Line {j+1}: {mm} mm over {round(px_len, 2)} px = {round(conv_val, 4)} mm/px", end=" ")

                    if len(conv_vals) > 0:
                        avg = round(sum(conv_vals) / len(conv_vals), 4)
                        AT.overview.loc[allinds, "conv"] = avg
                        print(f"Average conversion set to {avg} mm/pixel")

            elif mode == "framelimits":
                AT.overview.loc[allinds, "frame_start"] = result[1][0]
                AT.overview.loc[allinds, "frame_stop"] = result[1][1]
                print(f"Stored framelimits: start = {result[1][0]}, stop = {result[1][1]}")

            elif mode == "roi":
                AT.overview.loc[allinds, "roi"] = str(result[1])
                print(f"Stored ROI: {result[1]}")

            elif mode in ["mask", "maskzone", "zones", "walls"]:
                if isinstance(result[1], np.ndarray):
                    outname = f"{name}_{true_mode}.jpg"
                    outpath = os.path.join(AT.dirs["originals"], outname)
                    cv2.imwrite(outpath, result[1])

                    # Determine column name
                    if mask:
                        colname = "maskimg"
                    else:
                        singular_map = {"zones": "zone", "walls": "wall", "maskzone": "zone"}
                        base = singular_map.get(true_mode, true_mode)
                        colname = f"{base}img"

                    # Ensure the column exists
                    if colname not in AT.overview.columns:
                        AT.overview[colname] = pd.Series(dtype=object)

                    # Assign output name to rows
                    AT.overview.loc[allinds, colname] = outname
                    print(f"Stored {colname} image: {outname}")

                else:
                    print("No changes made.")

            elif mode == "points":
                points = result[1]  # List of drawn QPoint or tuple
                if len(points) == 0:
                    print("No points drawn.")
                    continue
                if ptcolnames:
                    if len(points) < len(ptcolnames):
                        print(f"Only {len(points)} of {len(ptcolnames)} required points drawn. Please draw all and try again.")
                        continue  # Skip this item and allow retry or safe exit
                    if len(points) > len(ptcolnames):
                        print(f"{len(points)} points drawn but only {len(ptcolnames)} labels provided. Extra points ignored.")
                        points = points[:len(ptcolnames)]
                    for j, col in enumerate(ptcolnames):
                        AT.overview.loc[allinds, col] = str(points[j])
                    print(f"Stored {len(points)} named points: {dict(zip(ptcolnames, points))}")
                else:
                    # fallback if no column names
                    for j, pt in enumerate(points):
                        AT.overview.loc[allinds, f"pt{j+1}"] = str(pt)
                    print(f"Stored unnamed points: {[str(p) for p in points]}")

            elif isinstance(result, tuple) and result[0] == "timepoints":
                # result is ("timepoints", df)
                df = result[1]
                if df is None or df.shape[0] == 0:
                    print("No valid frames obtained from GUI.")
                    continue
                # Save timepoints CSV adjacent to video
                csv_out = os.path.splitext(vid)[0] + ".csv"
                df.to_csv(csv_out, index=False)
                print(f"Saved timepoints to: {csv_out}")

            # final exit check for GUI result
            if result == "exit":
                break

        # after processing all videos, save overview if there were changes
        if overview_dirty:
            AT.save()
            print("Overview stored..")
        else:
            # ensure we still persist other changes (existing behavior)
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
            fullinds = list(AT.overview.index[AT.overview.get("exclude", pd.Series(dtype=object))!=1])
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
                AT.track(folder="originals", inds=[ind], frame_start=seq[0], frame_stop=seq[1], suffix=suffixi)

        AT.set_config(overwrite=a, show_tracking=b, frame_disstep=c,
            trajs_below=d, create_vid=e, create_dat=f)

    def track(AT, inds=None, names=None, query=None, cats=None, pools=1, folder="todo", frame_start=None,
        frame_stop=None, threshtype=None, objects=None, checkconschange=False, suffix="", threshfile=None,
        max_framedist=200, overwrite=None, check_flicker=False, skip_frames=0):
       
        if threshfile is not None:
            try:
                with open(threshfile, "r") as f:
                    AT.threshinfo = yaml.load(f, Loader=yaml.FullLoader)
                    print(f"Loading custom threshfile '{threshfile}'")
            except FileNotFoundError:
                raise FileNotFoundError(f"Threshfile '{threshfile}' not found.")
            except Exception as e:
                raise RuntimeError(f"Error loading threshfile '{threshfile}': {e}")
        else:
            with open(AT.cfiles["threshinfo"], 'r') as f:
                AT.threshinfo = yaml.load(f, Loader=yaml.FullLoader)

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

        existing = [os.path.exists(f) for f in trackfiles]
        existing_inds = [i for i, e in zip(inds, existing) if e]
        existing_trackfiles = [f for f, e in zip(trackfiles, existing) if e]
        missing_count = len(trackfiles) - len(existing_trackfiles)
        if missing_count > 0:
            missed = f"Skipped {missing_count} missing video files. "
        else:
            missed = ""
        inds = existing_inds
        trackfiles = existing_trackfiles

        # Now create the Tracker with only existing files
        T = Tracker(pools, inds, trackfiles, AT.dirs, AT.overview, 
                    AT.config, AT.threshinfo, frame_start, frame_stop, threshtype,
                    objects, checkconschange, suffix,
                    max_framedist=max_framedist, overwrite=overwrite, 
                    check_flicker=check_flicker, skip_frames=skip_frames)
        
        # Filter to only untracked files before starting pool
        if not overwrite:
            untracked_inds = []
            for ind in T.inds:
                filename = os.path.splitext(os.path.basename(trackfiles[T.inds.index(ind)]))[0]
                tracked_path = os.path.join(AT.dirs["tracked"], filename + suffix + "_TR.mp4")
                if not os.path.exists(tracked_path):
                    untracked_inds.append(ind)
            T.inds = untracked_inds
            lineprint(f"Tracking started of {len(T.inds)} files (skipping {len(trackfiles) - len(T.inds)} already tracked)..")
        else:
            lineprint(f"Tracking started of {len(T.inds)} files..")

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
            except Exception as e:
                video = AT.overview.loc[ind]["video"] if ind is not None else "unknown"
                lineprint(f"Error on row {ind} ({video}): {type(e).__name__}: {e}")
                raise
        else:
            AT.config.vis.show_tracking = False
            AT.config.vis.waitkey = 1
            def callback_function(output): T.inds = output
            if not notebook():
                pool = multiprocessing.Pool(min(pools, len(trackfiles)))
                counter = -1
                last_ind, last_video = None, "unknown"
                try:
                    while len(T.inds)>0:
                        counter += 1
                        ind = T.inds[0]
                        T.inds = T.inds[1:]
                        last_ind = ind
                        last_video = AT.overview.loc[ind]["video"]
                        trackfile = os.path.join(AT.dirs[folder], last_video + ".mp4")
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
                    lineprint(f"Error on row {last_ind} ({last_video}): {type(e).__name__}: {e}, terminating pool")
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
    
    def check_interactive(AT, folder="tracked", inds=None, names=None, query=None, cats=None, fileaction="overwrite"):
        if names is not None:
            inds = AT.get_inds(names)
        inds, vids = AT.get_files(folder, inds, query, cats)
        for i, ind in enumerate(inds):
            video_name = AT.overview.loc[ind, "video"]
            region = AT.overview.loc[ind].get("region", None)
            if region is not None:
                basename = f"{video_name}_R{region}"
            else:
                basename = video_name
            vid = os.path.join(AT.dirs["originals"], f"{video_name}.mp4")
            datafile = os.path.join(AT.dirs["tracked"], f"{basename}.csv")

            bgimg = AT.overview.loc[ind].get("bgimg", None)
            bgpath = os.path.join(AT.dirs["originals"], bgimg) if isinstance(bgimg, str) else None
            maskimg = AT.overview.loc[ind].get("maskimg", None)
            maskpath = os.path.join(AT.dirs["originals"], maskimg) if isinstance(maskimg, str) else None

            # --- Read and parse ROI ---
            if "roi" in AT.overview.columns and isinstance(AT.overview.loc[ind]["roi"], str):
                roival = literal_eval(AT.overview.loc[ind]["roi"])
            else:
                res = literal_eval(AT.overview.loc[ind, "resolution"])
                roival = ((0, 0), res)

            firstframe = AT.overview.loc[ind, "frame_start"]
            firstframe = 1 if pd.isna(firstframe) else int(firstframe)
            lastframe = AT.overview.loc[ind, "frame_stop"]
            lastframe = None if pd.isna(lastframe) else int(lastframe)

            # Pass roi to your annotation GUI if supported
            result = annotation_gui(
                media_file=vid,
                background_file=bgpath,
                mask_file=maskpath,
                mode="timepoints",
                firstframe=firstframe,
                lastframe=lastframe,
                fileaction=fileaction,
                data_file=datafile,
                roi=roival
            )
            if result == "exit":
                break

    def process(AT, pools=1, names=None, overwrite=False, fulldata=True, convert=True, 
                removeoutliers=True, alonewindow=5, smoothwin=10, changefps=None, 
                addIDs=True, nearmaskdis=20, trajgap = 50, edgedis=10, 
                filllenthresh_com=500, inmaskdis=10, mintrajlength=10, filllenthresh_headtail=40, 
                headtoorientspeedthresh=1, fillmissingorientdiffthresh=100, 
                centertype = None, centralise=False, delcontdata=False, powermate=False, force_single_traj=False):
        
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
        
        smoothwin : int; default = 10
            The window in frames to use for smoothing the data.
        changefps : int, default = None
            To subset the framerate of the video to a lower value, e.g. to help reduce the filesize.
        addIDs : bool, default = None
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
        filllenthresh_com : int; default = 500
            Maximum distance in frames between coordinate data outside of any potential mask that
            should be filled-in by differentiating.
        inmaskdis : int; default = 10
            Distance in pixels from mask that should be considered to be under the mask and
            therefore removed. Different from nearmaskdis, which is only used for determining
            the frames used for interpolating.
        mintrajlength : int; default = 10
            Minimum length of frames for a trajectory. Trajectories shorted than this are deleted.  
        filllenthresh_headtail : int, default = 40
            Maximum difference in frames for missing head and tail coordinate data outside of any 
            potential mask that shoudl be filled-in y differentiating.
        headtoorientspeedthresh : float; default = 1 
            The speed value in converted units above which heading should be used to set orientation.
        fillmissingorientdiffthresh : int; default = 100
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
            "changefps": changefps,
            "nearmaskdis": nearmaskdis,
            "inmaskdis": inmaskdis,
            "trajgap": trajgap,
            "edgedis": edgedis,
            "convert": convert,
            "smoothwin": smoothwin,
            "centertype": centertype,
            "centralise": centralise,
            "filllenthresh_com": filllenthresh_com,
            "filllenthresh_headtail": filllenthresh_headtail,
            "mintrajlength": mintrajlength,
            "fillmissingorientdiffthresh": fillmissingorientdiffthresh,
            "headtoorientspeedthresh": headtoorientspeedthresh,
            "delcontdata": delcontdata,
            "powermate": powermate,
            "force_single_traj": force_single_traj
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