#! /usr/bin/env python

import os
import re
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

    def __init__(self, filedir = "."):

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
    
        self.dir = filedir.rstrip(os.path.sep)

        # Set up subdirectories
        dirs = ["originals","todo","temp","tracked","processed"]
        dpaths = [os.path.join(self.dir, str(i)+d) for i,d in enumerate(dirs)]
        self.dirs = dict(zip(dirs, dpaths))
        
        if os.path.exists(self.dirs["todo"]):
            lineprint("Tracking folder loaded |", newline=False)
        else:
            for i in self.dirs:
                os.makedirs(self.dirs[i])
            lineprint("Set up tracking folder |", newline=False)

        # Move video files
        for file in listfiles(self.dir, (".h264",".mp4",".MP4",".mov",".m4v")):
            shutil.move(os.path.join(self.dir, file), self.dirs["originals"])

        # Set up config file paths
        basename = os.path.basename(self.dir.rstrip(os.path.sep)) or "tracking"
        cfiles = ["overview.xlsx", "config.conf", "threshinfo.yml"]
        fpaths = [os.path.join(self.dir, f"{basename}_{f}") for f in cfiles]
        self.cfiles = dict(zip([os.path.splitext(f)[0] for f in cfiles], fpaths))

        if os.path.exists(self.cfiles["overview"]):
            self.reload()
            print("Overview file loaded", end=" | ")
        else:
            cols = ["video","fps","fcount","resolution","frame_start",
                    "frame_stop","roi","conv","exp","date","trial","session",
                    "setup","ID","bgimg","maskimg","thresh_types",
                    "wallimg","zoneimg","objects","exclude"]
            self.overview = pd.DataFrame(columns=cols)
            self.save()
            print("", sep="", end=" | ")

        self.config = LocalConfig(self.cfiles["config"], compact_form=True)
        if not os.path.exists(self.cfiles["config"]):
            print("Configfile not found, new file created", end=" | ")
            for section in ["exp","track","bgextract","orient","vis"]:
                if section not in list(self.config):
                    self.config.add_section(section)
            self.set_config(fps=25, real_dims=None, startframe=1,
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

        if os.path.exists(self.cfiles["threshinfo"]):
            with open(self.cfiles["threshinfo"]) as file:
                self.threshinfo = yaml.load(file, Loader=yaml.FullLoader)
            print("Threshinfo file loaded")
        else:
            self.threshinfo = {}
            with open(self.cfiles["threshinfo"], "w") as file:
                yaml.dump(self.threshinfo, file, default_flow_style=False)
            print("Threshinfo file created")

        os.chdir(self.dir)

    def _name_and_index(self, vid):

        name,ext = os.path.splitext(vid)
        name = os.path.basename(name)
        ind = self.overview.index[self.overview.video == name].tolist()
        if len(ind) == 0:
            maxind = self.overview.index.max()
            ind = (maxind+1) if (len(self.overview) > 0) else 0
        else:
            ind = ind[0]

        return name, ind

    def _get_all_inds(self, query, cats, ind):

        if cats is None:
            qry = query
        else:
            cats = [cats] if type(cats) is not list else cats
            vals = self.overview.loc[ind, cats].values.tolist()
            nqry = to_query(cats, vals)
            qry = query+' and '+nqry if query is not None else nqry
        inds = self.overview.query(qry).index.tolist()

        return inds

    def save(self, silent=False):
        self.overview.to_excel(self.cfiles["overview"], index=False)
        if not silent:
            lineprint("Overview stored..")

    def reload(self):
        _conv = {col: str for col in [0]+list(range(8,17))}
        self.overview = pd.read_excel(self.cfiles["overview"], converters=_conv, engine='openpyxl')
        self.overview["date"] = self.overview["date"].astype(str).str[:10]
        self.config = LocalConfig(self.cfiles["config"], compact_form=True)

        # Normalise column names: merge legacy "Exclude" and "skip" into "exclude"
        ov = self.overview
        if "exclude" not in ov.columns:
            ov["exclude"] = np.nan
        if "Exclude" in ov.columns:
            ov.loc[ov["Exclude"] == 1, "exclude"] = 1
            ov.drop(columns=["Exclude"], inplace=True)
        if "skip" in ov.columns:
            ov.loc[ov["skip"] == 1, "exclude"] = 1
            ov.drop(columns=["skip"], inplace=True)
        self.overview = ov

    def showinfo(self, files=None, inds=None, return_inds=False):
        ov = self.overview
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
        self, 
        cdir="originals", 
        inds=None, 
        query=None, 
        cats=None, 
        filetype=".mp4", 
        existonly=False, 
        full=True, 
        show_extension=True
    ):
        overview = self.overview

        if "exclude" in overview.columns:
            overview = overview[overview["exclude"] != 1]

        # Select indices
        if inds is not None:
            inds = [i for i in inds if i in self.overview.index]
        else:
            if query is not None:
                overview = overview.query(query)
            if cats is not None:
                cats = [cats] if not isinstance(cats, list) else cats
                for cat in cats:
                    if cat not in self.overview:
                        raise ValueError(cat + " variable does not exist")
                overview = overview.drop_duplicates(subset=cats)
            inds = overview.index.tolist()
        
        # Get filenames (with or without extension)
        videos = self.overview.loc[inds].video
        if show_extension:
            filenames = [str(v) + filetype for v in videos]
        else:
            filenames = [str(v) for v in videos]

        if full:
            files = [os.path.join(self.dirs[cdir], f) for f in filenames]
        else:
            files = filenames

        if existonly:
            output = list(zip(*[(i, f) for i, f in zip(inds, files) if os.path.exists(f)]))
            inds, files = ([], []) if len(output) == 0 else map(list, output)

        return inds, files

    def update_overview(self, column, value, video=None, inds=None, save=True):
        """
        Inline replacement for overview DataFrame, by video name(s) or indices.
        """
        if (video is None) and (inds is None):
            raise ValueError("Provide at least one of 'video' or 'inds'")
        
        if video is not None:
            if isinstance(video, str):
                row_selector = self.overview["video"] == video
            elif isinstance(video, (list, tuple)):
                row_selector = self.overview["video"].isin(video)
            else:
                raise TypeError("'video' must be str or list/tuple of str")
        elif inds is not None:
            row_selector = self.overview.index.isin(inds)
        else:
            raise ValueError("No valid selection method.")
        
        self.overview.loc[row_selector, column] = value
        if save:
            self.save()
            
    def set_regions(self, inds, nr):

        for ind in inds:
            self.overview = duplicate_row(self.overview, ind, nr)
        self.overview = self.overview.sort_values(by=["video", "region"])
        self.save(silent=True)
        lineprint("Region information added..")

    def set_objects(self, inds=None, objects=None):
        if objects is None:
            raise ValueError("The 'objects' parameter cannot be None.")
        if inds is None:
            self.overview["objects"] = objects
        else:
            for ind, obj in zip(inds, objects):
                self.overview.loc[ind, "objects"] = obj
        self.save(silent=True)
        lineprint("Objects information added..")

    def set_config(self, **kwargs):

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
        frame_start: int, default = 1
            The start frame to be used for tracking
        frame_stop: int, default = 99999
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
        min_aspect_ratio : float, default = 1.4
            Minimum contour aspect ratio accepted as a valid animal detection.
        max_aspect_ratio : float, default = 10
            Maximum contour aspect ratio accepted as a valid animal detection.
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
        draw_contournrs : bool, default = False
            If the blob contour numbers should be drawn
        """

        # Special: regions modifies overview structure
        if "regions" in kwargs:
            self.config.exp.regions = kwargs["regions"]
            if kwargs["regions"] and "region" not in self.overview:
                self.overview.insert(1, "region", 1)
                self.save()
            if not kwargs["regions"] and "region" in self.overview:
                self.overview.drop("region", axis=1, inplace=True)
                self.save()

        # Public frame_start/frame_stop map to internal startframe/stopframe
        if "frame_start" in kwargs:
            kwargs["startframe"] = kwargs.pop("frame_start")
        if "frame_stop" in kwargs:
            kwargs["stopframe"] = kwargs.pop("frame_stop")

        # exp section
        if "fps" in kwargs:
            self.config.exp.fps = int(kwargs["fps"])
        if "real_dims" in kwargs:
            self.config.exp.realdims = kwargs["real_dims"]

        # track section
        _track_map = {
            "startframe": "startframe",
            "stopframe": "stopframe",
            "keep_frames": "keep_frames",
            "strict": "strict",
            "overwrite": "overwrite",
            "create_vid": "create_vid",
            "create_dat": "create_dat",
            "simple": "simple",
            "contour_mode": "contour_mode",
            "orientfrombw": "orientfrombw",
            "mergedmindist": "mergedmindist",
            "linkdisthreshold": "linkdisthreshold",
            "shape_area_tol": "shape_area_tol",
            "shape_history_len": "shape_history_len",
            "min_aspect_ratio": "min_aspect_ratio",
            "max_aspect_ratio": "max_aspect_ratio",
        }
        for k, attr in _track_map.items():
            if k in kwargs:
                setattr(self.config.track, attr, kwargs[k])

        # bgextract section
        if "bg_frames" in kwargs:
            self.config.bgextract.bg_frames = kwargs["bg_frames"]

        # vis section — plain scalar assignments
        _vis_map = {
            "show_tracking": "show_tracking",
            "vid_displaysize": "vid_displaysize",
            "frame_disstep": "frame_disstep",
            "centre_lwidth": "centre_lwidth",
            "orient_lwidth": "orient_lwidth",
            "orient_tip": "orient_tip",
            "orient_length": "orient_length",
            "traj_length": "traj_length",
            "traj_minthick": "traj_minthick",
            "traj_maxthick": "traj_maxthick",
            "traj_opacity": "traj_opacity",
            "mask_opacity": "mask_opacity",
            "box_opacity": "box_opacity",
            "draw_contournrs": "contnrs",
            "trajs_below": "trajs_below",
        }
        for k, attr in _vis_map.items():
            if k in kwargs:
                setattr(self.config.vis, attr, kwargs[k])

        # vis section — colour params (need namedcols conversion)
        _vis_col_map = {
            "idcol": "idcol",
            "contour_col": "contour_col",
            "centre_col": "centre_col",
            "front_col": "front_col",
            "orient_col": "orient_col",
            "traj_col": "traj_col",
        }
        for k, attr in _vis_col_map.items():
            if k in kwargs:
                setattr(self.config.vis, attr, namedcols(kwargs[k]))

        # userwait translates to waitkey int
        if "userwait" in kwargs:
            self.config.vis.waitkey = 0 if kwargs["userwait"] is True else 1

        if len(kwargs) > 0:
            self.config.save()

        if "internal" not in kwargs:
            print("Config settings stored and loaded..")

    def setup_files(self, fname_extract=True, fname_vars=("date", "exp", "trial", "session", "setup", "ID"),
                    fname_sep="-", skip=False, autoconvert=True, fps=None):
        """
        Prepares video files for tracking by converting, extracting metadata, 
        and updating the overview.
        """
        lineprint("Preparing video files for tracking..", end=" ")

        originals_dir = self.dirs["originals"]

        # Detect which .h264 files still need to be converted
        convlist = []
        for file in listfiles(originals_dir, type=".h264", keepdir=False, keepext=True):
            base = os.path.splitext(file)[0]
            if not os.path.exists(os.path.join(originals_dir, base + ".mp4")):
                convlist.append(file)

        if autoconvert:
            if convlist:
                lineprint(f"Converting {len(convlist)} files...", newline=False)
                conversion_fps = fps if fps is not None else self.config.exp.fps
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
            existing = set(os.path.splitext(v)[0] for v in self.overview["video"].dropna())
            todovids = [v for v in todovids if os.path.splitext(os.path.basename(v))[0] not in existing]

        if not todovids:
            print("No files to prepare..", end=" ")
            return

        expected_n = len(fname_vars)

        for i, vid in enumerate(todovids):
            name, ind = self._name_and_index(vid)
            lineprint(f"Video {i+1}|{totalvids} {name}", True, False)
            self.overview.loc[ind, "video"] = name

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
                    self.overview.loc[ind, fname_vars[j]] = nameval
                self.overview.loc[ind, "vidseq"] = version_val  # Always adds this column
                print("Filename vars extracted", end=" ")

            # Check and extract video info
            if not check_media(vid):
                continue

            cap = cv2.VideoCapture(vid)
            fps, width, height, fcount = get_vid_params(cap)
            self.overview.loc[ind, "fps"] = fps
            self.overview.loc[ind, "resolution"] = str((width, height))
            self.overview.loc[ind, "roi"] = str(((0, 0), (width, height)))
            max_pyframe = find_max_working_pyframe(cap)
            self.overview.loc[ind, "fcount"] = max_pyframe + 1
            print("Video info extracted")

        self.save()

    def get_bgfiles(self, inds=[], starts=[], stops=[], overwrite=False):

        lineprint("Extracting background files..")

        # 1) Determine which rows to process
        if len(inds) == 0:
            # All rows not excluded
            df = self.overview[self.overview.get("exclude", pd.Series(dtype=object)) != 1]
            rows = df.index.tolist()
        else:
            rows = inds

        # 2) Build a todo-list per row (i.e. per region)
        todolist = []
        for idx in rows:
            video = self.overview.loc[idx, "video"]
            region = self.overview.loc[idx].get("region", None)

            # unique bg filename
            if region is None:
                bgname = f"{video}_bg.jpg"
            else:
                bgname = f"{video}_R{region}_bg.jpg"

            bgpath = os.path.join(self.dirs["originals"], bgname)

            # we will extract per row
            if overwrite or not os.path.isfile(bgpath):
                todolist.append((idx, video, region, bgname, bgpath))
        
        # 3) Extract backgrounds
        if len(todolist) == 0:
            print("All files done..", end=" ")
        else:
            for k, (idx, video, region, bgname, bgpath) in enumerate(todolist):

                vidpath = os.path.join(self.dirs["originals"], f"{video}.mp4")
                name, _ = self._name_and_index(vidpath)
                lineprint(f"Video {k+1}|{len(todolist)} {name}", True, False)

                # --- Get region-specific start/stop ---
                start, stop = self.overview.loc[idx, ["frame_start", "frame_stop"]]
                start = int(start) if not np.isnan(start) else None
                stop = int(stop) if not np.isnan(stop) else None

                # Override start/stop if provided manually
                if starts:
                    start = starts[0] if len(starts) == 1 else starts[k]
                if stops:
                    stop = stops[0] if len(stops) == 1 else stops[k]

                framenr = self.config.bgextract.bg_frames

                img_bg = bg_extract(vidpath, start, stop, framenr)
                cv2.imwrite(bgpath, img_bg)

                # Update ONLY this row
                self.overview.loc[idx, "bgimg"] = bgname

            self.save(silent=True)

    def set_interactive(self, inds=None, framelimits=None, roi=None, mask=None, maskzone=None, zones=None,
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

        inds, vids = self.get_files("originals", inds, query, cats)
        fileaction = "overwrite"
        datafile = None
        overview_dirty = False

        def _result_pxlen(res):
            if res is None:
                return None
            if isinstance(res, tuple):
                if len(res) > 1 and isinstance(res[1], (int, float)):
                    return float(res[1])
                if len(res) > 2 and isinstance(res[2], (int, float)):
                    return float(res[2])
                if len(res) > 1 and isinstance(res[1], (list, tuple)) and len(res[1]) >= 2:
                    try:
                        p0, p1 = res[1][0], res[1][1]
                        return float(((float(p0[0])-float(p1[0]))**2 + (float(p0[1])-float(p1[1]))**2)**0.5)
                    except Exception:
                        return None
            return None

        for i, ind in enumerate(inds):
            allinds = self._get_all_inds(query, cats, ind) if query or cats else ind
            vid = os.path.join(self.dirs["originals"], f"{self.overview.loc[ind, 'video']}.mp4")
            name, _ = self._name_and_index(vid)

            if "region" in self.overview.columns and not pd.isna(self.overview.loc[ind, "region"]):
                region = self.overview.loc[ind, "region"]
                name += f"_R{int(region)}"

            lineprint(f"\nVideo {i + 1} of {len(vids)} — {name}", True, False)

            if not os.path.isfile(vid):
                print("Video file does not exist, skipping..")
                continue

            # Read ROI if present, otherwise fallback later where needed
            if "roi" in self.overview.columns and isinstance(self.overview.loc[ind].get("roi", None), str):
                roival = literal_eval(self.overview.loc[ind]["roi"])
            else:
                # if resolution present, use it as full-frame ROI
                try:
                    res = literal_eval(self.overview.loc[ind, "resolution"])
                    roival = ((0, 0), res)
                except Exception:
                    roival = None

            firstframe = self.overview.loc[ind, "frame_start"]
            firstframe = 1 if pd.isna(firstframe)or str(firstframe).strip() == "" else int(firstframe)
            lastframe = self.overview.loc[ind, "frame_stop"]
            lastframe = None if pd.isna(lastframe) or str(lastframe).strip() == "" else int(lastframe)

            bgimg = self.overview.loc[ind].get("bgimg", None)
            bgpath = os.path.join(self.dirs["originals"], bgimg) if isinstance(bgimg, str) else None

            maskpath = None
            mask_column = None
            if mask: mask_column = "maskimg"
            elif maskzone: mask_column = "zoneimg"
            elif walls: mask_column = "wallimg"
            elif zones: mask_column = "zoneimg"
            elif threshtypes: mask_column = "maskimg"
            if mask_column:
                maskfile_entry = self.overview.loc[ind].get(mask_column, None)
                if isinstance(maskfile_entry, str):
                    maskpath = os.path.join(self.dirs["originals"], maskfile_entry)
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
                        threshold_dict=self.threshinfo.get(ttype, {}),
                        firstframe=firstframe,
                        lastframe=lastframe,
                        fileaction=fileaction,
                        data_file=datafile
                    )
                    if result is None or result == "exit":
                        print(" — exited")
                        break
                    if isinstance(result[1], dict) and len(result[1]) > 0:
                        self.threshinfo[ttype] = result[1]
                        print(f" stored...", end=" ")
                        # Save threshold info to file
                        finalfile = threshfile
                        if not finalfile:
                            finalfile = self.cfiles.get("threshinfo", "threshinfo.yml")
                        if not finalfile.endswith(".yml"):
                            finalfile += ".yml"
                        with open(finalfile, "w") as f:
                            yaml.safe_dump(self.threshinfo, f, default_flow_style=False)
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
                            self.save()
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
                        self.overview.loc[allinds, "events"] = str(grouped)
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
                px_len = _result_pxlen(result)
                if px_len is None:
                    print("Could not determine pixel length from GUI result:", result)
                    continue

                if isinstance(conv_mm, (int, float)):
                    # Single value, single interaction
                    self.overview.loc[allinds, "conv"] = round(conv_mm / px_len, 4)
                    overview_dirty = True
                    lineprint(f"Conversion set to {self.overview.loc[ind, 'conv']} mm/pixel", end=" ")

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
                        self.overview.loc[allinds, "conv"] = avg
                        overview_dirty = True
                        print(f"Average conversion set to {avg} mm/pixel")

            elif mode == "framelimits":
                self.overview.loc[allinds, "frame_start"] = result[1][0]
                self.overview.loc[allinds, "frame_stop"] = result[1][1]
                overview_dirty = True
                print(f"Stored framelimits: start = {result[1][0]}, stop = {result[1][1]}")

            elif mode == "roi":
                self.overview.loc[allinds, "roi"] = str(result[1])
                overview_dirty = True
                print(f"Stored ROI: {result[1]}")

            elif mode in ["mask", "maskzone", "zones", "walls"]:
                if isinstance(result[1], np.ndarray):
                    outname = f"{name}_{true_mode}.jpg"
                    outpath = os.path.join(self.dirs["originals"], outname)
                    cv2.imwrite(outpath, result[1])

                    # Determine column name
                    if mask:
                        colname = "maskimg"
                    else:
                        singular_map = {"zones": "zone", "walls": "wall", "maskzone": "zone"}
                        base = singular_map.get(true_mode, true_mode)
                        colname = f"{base}img"

                    # Ensure the column exists
                    if colname not in self.overview.columns:
                        self.overview[colname] = pd.Series(dtype=object)

                    self.overview.loc[allinds, colname] = outname
                    overview_dirty = True
                    print(f"Stored {colname} image: {outname}")

                else:
                    print("No changes made.")

            elif mode == "points":
                points = result[1]
                if len(points) == 0:
                    print("No points drawn.")
                    continue
                if ptcolnames:
                    if len(points) < len(ptcolnames):
                        print(f"Only {len(points)} of {len(ptcolnames)} required points drawn. Please draw all and try again.")
                        continue
                    if len(points) > len(ptcolnames):
                        print(f"{len(points)} points drawn but only {len(ptcolnames)} labels provided. Extra points ignored.")
                        points = points[:len(ptcolnames)]
                    for j, col in enumerate(ptcolnames):
                        self.overview.loc[allinds, col] = str(points[j])
                    overview_dirty = True
                    print(f"Stored {len(points)} named points: {dict(zip(ptcolnames, points))}")
                else:
                    for j, pt in enumerate(points):
                        self.overview.loc[allinds, f"pt{j+1}"] = str(pt)
                    overview_dirty = True
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

        if overview_dirty:
            self.save()

    def drymode(self, rand_filenr=10, rand_seqnr=5, rand_seqlen=100, suffix="dry", rerun=False):

        a = self.config.track.overwrite
        b = self.config.vis.show_tracking
        c = self.config.vis.frame_disstep
        d = self.config.vis.trajs_below
        e = self.config.track.create_vid
        f = self.config.track.create_dat

        self.set_config(overwrite=True, show_tracking=False, frame_disstep=99999,
            trajs_below=False, create_vid=True, create_dat=False)

        if not rerun or not hasattr(self, 'dryinds'):
            fullinds = list(self.overview.index[self.overview.get("exclude", pd.Series(dtype=object))!=1])
            self.dryinds = sample(fullinds, min(rand_filenr,len(fullinds)-1))

        for ind in self.dryinds:
            sub = self.overview.loc[ind]
            minf = int(sub["frame_start"] if sub["frame_start"]==sub["frame_start"] else 1)
            maxf = int(sub["frame_stop"] if sub["frame_stop"]==sub["frame_stop"] else sub["fcount"])
            if not rerun or not hasattr(self, '_dryframes_cache'):
                self._dryframes_cache = {}
            if ind not in self._dryframes_cache:
                startframes = sample(list(range(minf, maxf - rand_seqlen - 1)), rand_seqnr)
                self._dryframes_cache[ind] = list(zip(startframes, [s + rand_seqlen for s in startframes]))
            seqs = self._dryframes_cache[ind]
            for i, seq in enumerate(seqs):
                suffixi = suffix + str(i+1).zfill(len(str(len(seqs))))
                self.track(folder="originals", inds=[ind], frame_start=seq[0], frame_stop=seq[1], suffix=suffixi)

        self.set_config(overwrite=a, show_tracking=b, frame_disstep=c,
            trajs_below=d, create_vid=e, create_dat=f)

    def track(self, inds=None, names=None, query=None, cats=None, pools=1, folder="todo", frame_start=None,
        frame_stop=None, threshtype=None, objects=None, checkconschange=False, suffix="", threshfile=None,
        max_framedist=200, overwrite=None, check_flicker=False, skip_frames=0, watch=False, watch_interval=60):

        if watch:
            lineprint(f"Watch mode enabled — checking every {watch_interval}s (Ctrl+C to stop)..")

        _stop = False
        while not _stop:

            # In watch mode reload overview and threshinfo from disk each pass
            if watch:
                self.reload()

            if threshfile is not None:
                try:
                    with open(threshfile, "r") as f:
                        self.threshinfo = yaml.load(f, Loader=yaml.FullLoader)
                        if not watch:
                            print(f"Loading custom threshfile '{threshfile}'")
                except FileNotFoundError:
                    raise FileNotFoundError(f"Threshfile '{threshfile}' not found.")
                except Exception as e:
                    raise RuntimeError(f"Error loading threshfile '{threshfile}': {e}")
            else:
                with open(self.cfiles["threshinfo"], 'r') as f:
                    self.threshinfo = yaml.load(f, Loader=yaml.FullLoader)

            _inds = inds
            if names is not None:
                _inds = self.get_inds(names)

            if _inds is not None:
                if "exclude" in self.overview.columns:
                    _inds = [i for i in _inds if self.overview.loc[i, "exclude"] != 1]
                trackfiles = [os.path.join(self.dirs[folder], f"{video}.mp4")
                              for video in self.overview.loc[_inds, "video"]]
            else:
                _inds, trackfiles = self.get_files(folder, _inds, query, cats, existonly=False)

            # Fix Localconfig messing up the class variables
            cbak = self.config
            del self.config
            self.config = Box({s: {k:v for (k,v) in cbak.items(s)} for s in cbak})

            existing = [os.path.exists(f) for f in trackfiles]
            existing_inds = [i for i, e in zip(_inds, existing) if e]
            existing_trackfiles = [f for f, e in zip(trackfiles, existing) if e]
            missing_count = len(trackfiles) - len(existing_trackfiles)
            if missing_count > 0:
                lineprint(f"Skipped {missing_count} missing video file(s).")
            _inds = existing_inds
            trackfiles = existing_trackfiles

            # Now create the Tracker with only existing files
            T = Tracker(pools, _inds, trackfiles, self.dirs, self.overview,
                        self.config, self.threshinfo, frame_start, frame_stop, threshtype,
                        objects, checkconschange, suffix,
                        max_framedist=max_framedist, overwrite=overwrite,
                        check_flicker=check_flicker, skip_frames=skip_frames)

            lineprint(f"Tracking started of {len(T.inds)} files..")

            if len(T.inds) == 0:
                self.config = cbak
                if not watch:
                    _stop = True
                else:
                    lineprint(f"Watch: nothing to track, sleeping {watch_interval}s..")
                    try:
                        time.sleep(watch_interval)
                    except KeyboardInterrupt:
                        lineprint("Watch mode stopped.")
                        _stop = True
                continue

            if pools < 2:
                stop = False
                while len(T.inds) > 0 and not stop:
                    ind = T.inds[0]
                    trackfile = os.path.join(self.dirs[folder], self.overview.loc[ind]["video"] + ".mp4")
                    try:
                        T.setuptracking(ind, trackfile)
                    except KeyboardInterrupt:
                        lineprint("\nUser terminated tracking..")
                        stop = True
                        _stop = True
                    except Exception as e:
                        video = self.overview.loc[ind]["video"]
                        lineprint(f"Error on row {ind} ({video}): {type(e).__name__}: {e} — skipping")
                        if ind in T.inds:
                            T.inds.remove(ind)
                if not _stop:
                    lineprint("Tracking finished..")
            else:
                self.config.vis.show_tracking = False
                self.config.vis.waitkey = 1
                def callback_function(output): T.inds = output
                if not notebook():
                    pool = multiprocessing.Pool(min(pools, len(trackfiles)))
                    last_ind, last_video = None, "unknown"
                    try:
                        all_async = []
                        while len(T.inds) > 0:
                            ind = T.inds[0]
                            T.inds = T.inds[1:]
                            last_ind = ind
                            last_video = self.overview.loc[ind]["video"]
                            trackfile = os.path.join(self.dirs[folder], last_video + ".mp4")
                            all_async.append(pool.apply_async(T.setuptracking,
                                                              (ind, trackfile),
                                                              callback=callback_function))
                            time.sleep(0.2)
                        for r in all_async:
                            r.get()
                        pool.close()
                    except KeyboardInterrupt:
                        lineprint("\nUser terminated tracking pool..")
                        pool.terminate()
                        _stop = True
                    except Exception as e:
                        lineprint(f"Error on row {last_ind} ({last_video}): {type(e).__name__}: {e}, terminating pool")
                        pool.terminate()
                        lineprint("pool is terminated")
                    finally:
                        pool.join()
                        if not _stop:
                            lineprint("Tracking completed..")
                else:
                    lineprint("Pooled tracking can only be run from the terminal, exiting..")

            self.config = cbak

            if not watch or _stop:
                _stop = True
            else:
                lineprint(f"Watch: pass complete, sleeping {watch_interval}s..")
                try:
                    time.sleep(watch_interval)
                except KeyboardInterrupt:
                    lineprint("Watch mode stopped.")
                    _stop = True

    def _pworker(self, trackedfile, config_dict):
        """Each worker creates its own Processor instance and processes the file."""
        thread_id = threading.get_ident()
        P = Processor(**config_dict)  # No pickling issues with threads
        P.setup(trackedfile, thread_id)  
    
    def check_interactive(self, folder="tracked", inds=None, names=None, query=None, cats=None, fileaction="overwrite"):
        if names is not None:
            inds = self.get_inds(names)
        inds, vids = self.get_files(folder, inds, query, cats)
        for i, ind in enumerate(inds):
            video_name = self.overview.loc[ind, "video"]
            region = self.overview.loc[ind].get("region", None)
            if region is not None:
                basename = f"{video_name}_R{region}"
            else:
                basename = video_name
            vid = os.path.join(self.dirs["originals"], f"{video_name}.mp4")
            datafile = os.path.join(self.dirs["tracked"], f"{basename}.csv")

            bgimg = self.overview.loc[ind].get("bgimg", None)
            bgpath = os.path.join(self.dirs["originals"], bgimg) if isinstance(bgimg, str) else None
            maskimg = self.overview.loc[ind].get("maskimg", None)
            maskpath = os.path.join(self.dirs["originals"], maskimg) if isinstance(maskimg, str) else None

            # --- Read and parse ROI ---
            if "roi" in self.overview.columns and isinstance(self.overview.loc[ind]["roi"], str):
                roival = literal_eval(self.overview.loc[ind]["roi"])
            else:
                res = literal_eval(self.overview.loc[ind, "resolution"])
                roival = ((0, 0), res)

            firstframe = self.overview.loc[ind, "frame_start"]
            firstframe = 1 if pd.isna(firstframe) else int(firstframe)
            lastframe = self.overview.loc[ind, "frame_stop"]
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

    def process(self, pools=1, names=None, overwrite=False, fulldata=True, convert=True, 
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
        trackedfiles = listfiles(self.dirs["tracked"], type=".csv", keepdir=True)
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
            e_file = os.path.normpath(os.path.join(self.dirs["tracked"], base_name + "_E.csv"))
            original_file = os.path.normpath(os.path.join(self.dirs["tracked"], base_name + ".csv"))
            
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
            "dirs": self.dirs,
            "config": self.config,
            "overview": self.overview,
            "trackedfiles": trackedfiles,
            "orientfrombw": self.config.track.orientfrombw,
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
                    executor.map(lambda f: self._pworker(f, config_dict), trackedfiles)
            else:
                lineprint("Pooled processing can only be run from the terminal, exiting..")