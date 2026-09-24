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

from pythutils.fileutils import listfiles
from pythutils.sysutils import lineprint
from pythutils.drawutils import namedcols
from pythutils.mediautils import get_vid_params, check_media
from pythutils.datutils import to_query

from atracker.__version__ import __version__
from atracker.editor import annotation_gui
from atracker.track import Tracker
from atracker.process import Processor
from atracker.helpers.media import convert_h264_to_mp4, bg_extract, find_max_working_pyframe
from atracker.helpers.data import duplicate_row, notebook
from atracker.helpers.contours import coordsfrommask, coordsfromzones
from atracker.helpers.pool import run_pool
from atracker.visualise import Visualiser as _Visualiser

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
            for section in ["track","vis"]:
                if section not in list(self.config):
                    self.config.add_section(section)
            self.set_config(
                overwrite=True, create_vid=True, create_dat=True,
                advanced=False, link_dist=100, merge_dist=20,
                size_filter=False, size_filter_tol=0.25, size_filter_memory=500,
                min_aspect_ratio=1.4, max_aspect_ratio=10,
                check_flicker=False, skip_frames=0, max_framedist=200, track_merges=False,
                show_tracking=True, vid_displaysize=1, frame_disstep=100,
                userwait=False, idcol=True,
                contour_col="blue", centre_col="white", front_col="black",
                orient_col="black", traj_col="yellow", centre_lwidth=13,
                orient_lwidth=2, orient_tip=0.15, orient_length=15,
                traj_length=4, traj_minthick=6.4, traj_maxthick=9,
                traj_opacity=0.5, mask_opacity=0.15, box_opacity=0.7,
                draw_contournrs=False, trajs_below=False, internal="")
            print("Config settings stored", end=" | ")
        else:
            self._migrate_config()
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

    def _migrate_config(self):
        """Detect a legacy config format and migrate it to the current one."""
        sections = list(self.config)
        legacy_sections = [s for s in ["exp", "bgextract", "orient"] if s in sections]
        try:
            track_keys = set(k for k, _ in self.config.items("track"))
        except Exception:
            track_keys = set()
        legacy_params = {"simple", "contour_mode", "startframe", "stopframe",
                         "keep_frames", "strict"} & track_keys

        if not legacy_sections and not legacy_params:
            return

        # Save backup of old config
        backup_path = self.cfiles["config"].replace(".conf", "_legacy_backup.conf")
        shutil.copy2(self.cfiles["config"], backup_path)

        # Read old track values to preserve them
        def _gt(key, default):
            try:
                val = dict(self.config.items("track")).get(key, default)
                return default if val is None else val
            except Exception:
                return default

        overwrite       = _gt("overwrite", True)
        create_vid      = _gt("create_vid", True)
        create_dat      = _gt("create_dat", True)
        advanced        = not bool(_gt("simple", True))
        link_dist       = int(float(_gt("linkdisthreshold", 100)))
        merge_dist      = int(float(_gt("mergedmindist", 20)))
        size_filter     = _gt("contour_mode", "static") == "dynamic"
        size_filter_tol = float(_gt("shape_area_tol", 0.25))
        size_filter_mem = int(float(_gt("shape_history_len", 500)))
        min_ar          = float(_gt("min_aspect_ratio", 1.4))
        max_ar          = float(_gt("max_aspect_ratio", 10))

        # Recreate config from scratch with new format
        os.remove(self.cfiles["config"])
        self.config = LocalConfig(self.cfiles["config"], compact_form=True)
        for section in ["track", "vis"]:
            self.config.add_section(section)

        self.set_config(
            overwrite=overwrite, create_vid=create_vid, create_dat=create_dat,
            advanced=advanced, link_dist=link_dist, merge_dist=merge_dist,
            size_filter=size_filter, size_filter_tol=size_filter_tol,
            size_filter_memory=size_filter_mem,
            min_aspect_ratio=min_ar, max_aspect_ratio=max_ar,
            check_flicker=False, skip_frames=0, max_framedist=200, track_merges=False,
            show_tracking=True, vid_displaysize=1, frame_disstep=100,
            userwait=False, idcol=True,
            contour_col="blue", centre_col="white", front_col="black",
            orient_col="black", traj_col="yellow", centre_lwidth=13,
            orient_lwidth=2, orient_tip=0.15, orient_length=15,
            traj_length=4, traj_minthick=6.4, traj_maxthick=9,
            traj_opacity=0.5, mask_opacity=0.15, box_opacity=0.7,
            draw_contournrs=False, trajs_below=False, internal="")

        print(f"\n{'='*60}")
        print("CONFIG MIGRATION: Legacy config format detected!")
        print(f"  Backup saved as: {os.path.basename(backup_path)}")
        print("  Track settings migrated:")
        print(f"    simple={not advanced!s:<5} -> advanced={advanced}")
        print(f"    linkdisthreshold   -> link_dist     = {link_dist}")
        print(f"    mergedmindist      -> merge_dist    = {merge_dist}")
        print(f"    contour_mode       -> size_filter   = {size_filter}")
        print(f"    shape_area_tol     -> size_filter_tol    = {size_filter_tol}")
        print(f"    shape_history_len  -> size_filter_memory = {size_filter_mem}")
        if legacy_sections:
            print(f"  Removed sections: {legacy_sections}")
        print("  Visualisation settings reset to defaults.")
        print("  Review your new config file and update if needed.")
        print(f"{'='*60}\n")

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
        Dynamically sets the configuration file.

        Parameters
        ----------
        regions : bool, default False
            If regions should be considered.
        overwrite : bool, default True
            If tracking data should be overwritten or not.
        create_vid : bool, default True
            If a tracking video should be created.
        create_dat : bool, default True
            If data should be written to file.
        advanced : bool, default False
            Enable advanced contour extraction: skeleton, head, tail,
            orientation, curvature. False = centroid and area only.
        size_filter : bool, default False
            Enable per-ID rolling area consistency check on top of global
            min/max thresholds. Rejects detections that deviate strongly
            from each ID's recent area history.
        size_filter_tol : float, default 0.25
            Tolerance for size_filter. Accepts area within
            [tol × median, (1/tol) × median].
        size_filter_memory : int, default 500
            Number of accepted frames used to compute the rolling median area.
        link_dist : int, default 100
            Maximum distance in pixels per frame to link two IDs during tracking.
        merge_dist : int, default 20
            Minimum distance previous contours must be from a potential merged
            contour for it to be flagged as a merge.
        min_aspect_ratio : float, default 1.4
            Minimum contour aspect ratio accepted as a valid detection.
        max_aspect_ratio : float, default 10
            Maximum contour aspect ratio accepted as a valid detection.
        check_flicker : bool, default False
            If brightness flicker detection should be used.
        skip_frames : int, default 0
            Number of frames to skip between tracked frames (0 = track all).
        max_framedist : int, default 200
            Maximum pixel distance per frame before a detection is rejected as a jump.
        track_merges : bool, default False
            If merge/split detection should be attempted.
        video_codec : str, default "libx264"
            FFmpeg video encoder used for tracking videos.
        video_preset : str, default "ultrafast"
            FFmpeg encoder preset. Faster presets use more storage.
        video_crf : int, default 23
            Constant-rate-factor quality setting for FFmpeg encoders.
        video_resize : float, default 1.0
            Output-only scale factor for tracking videos. Values below 1 reduce
            video size and encoding cost while tracking remains full resolution.
        show_tracking : bool, default True
            If tracking should be shown live.
        vid_displaysize : float, default 1
            Size of the display window relative to video size.
        frame_disstep : int, default 100
            Frame interval at which frame number is printed inline.
        userwait : bool, default False
            If tracking display should wait for user keypress.
        idcol : bool, default True
            If IDs should have unique colours.
        contour_col : str, default "blue"
            Colour of object contours.
        centre_col : str, default "white"
            Colour of centre points.
        front_col : str, default "black"
            Colour of front points.
        orient_col : str, default "black"
            Colour of orientation arrows.
        traj_col : str, default "yellow"
            Colour of trajectories.
        centre_lwidth : int, default 13
            Line thickness of the centre point marker.
        orient_lwidth : int, default 2
            Line thickness of orientation arrows.
        orient_tip : float, default 0.15
            Width of orientation arrow tip.
        orient_length : int, default 15
            Length of orientation arrows.
        traj_length : float, default 4
            Trajectory display length in seconds.
        traj_minthick : float, default 6.4
            Minimum thickness of trajectory line.
        traj_maxthick : float, default 9
            Maximum thickness of trajectory line.
        traj_opacity : float, default 0.5
            Opacity of trajectories.
        mask_opacity : float, default 0.15
            Opacity of the mask overlay.
        box_opacity : float, default 0.7
            Opacity of the info box.
        draw_contournrs : bool, default False
            If blob contour numbers should be drawn.
        trajs_below : bool, default False
            If trajectories should be drawn below contours.
        """

        # Special: regions modifies overview structure
        if "regions" in kwargs:
            self.config.track.regions = kwargs["regions"]
            if kwargs["regions"] and "region" not in self.overview:
                self.overview.insert(1, "region", 1)
                self.save()
            if not kwargs["regions"] and "region" in self.overview:
                self.overview.drop("region", axis=1, inplace=True)
                self.save()

        # track section
        _track_map = {
            "overwrite": "overwrite",
            "create_vid": "create_vid",
            "create_dat": "create_dat",
            "advanced": "advanced",
            "size_filter": "size_filter",
            "size_filter_tol": "size_filter_tol",
            "size_filter_memory": "size_filter_memory",
            "link_dist": "link_dist",
            "merge_dist": "merge_dist",
            "min_aspect_ratio": "min_aspect_ratio",
            "max_aspect_ratio": "max_aspect_ratio",
            "check_flicker": "check_flicker",
            "skip_frames": "skip_frames",
            "max_framedist": "max_framedist",
            "track_merges": "track_merges",
            "video_codec": "video_codec",
            "video_preset": "video_preset",
            "video_crf": "video_crf",
            "video_resize": "video_resize",
        }
        for k, attr in _track_map.items():
            if k in kwargs:
                setattr(self.config.track, attr, kwargs[k])

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
                conversion_fps = fps if fps is not None else 25
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

                framenr = 25

                img_bg = bg_extract(vidpath, start, stop, framenr)
                cv2.imwrite(bgpath, img_bg)

                # Update ONLY this row
                self.overview.loc[idx, "bgimg"] = bgname

            self.save(silent=True)

    def set_interactive(self, inds=None, framelimits=None, roi=None, mask=None, maskzone=None, zones=None,
                        walls=None, conv=None, getpts=None, conv_mm=None, threshtypes=None,
                        query=None, cats=None, ptcolnames=None, threshfile=None, events=False):
        """
        Interactive mode for various tasks. Deprecated — use AT.editor() for the new multi-file editor.

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

        gui_state = {"was_fullscreen": False}

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
            elif maskzone: mask_column = "maskzoneimg"
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
                        data_file=datafile,
                        start_fullscreen=gui_state["was_fullscreen"],
                        _state=gui_state
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
                        data_file=None,
                        start_fullscreen=gui_state["was_fullscreen"],
                        _state=gui_state
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
                data_file=datafile,
                start_fullscreen=gui_state["was_fullscreen"],
                _state=gui_state
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
                                data_file=datafile,
                                start_fullscreen=gui_state["was_fullscreen"],
                                _state=gui_state
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
                    elif maskzone:
                        colname = "maskzoneimg"
                    else:
                        singular_map = {"zones": "zone", "walls": "wall"}
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

    def editor(self, names=None, query=None, cats=None, inds=None, purpose="mask"):
        """
        Open the interactive editor for one or more files.

        Parameters
        ----------
        names : list[str] | str | None
            Video names to select. Same behaviour as track().
        query : str | None
            Pandas query string applied to the overview.
        cats : str | list[str] | None
            Category columns; rows with identical values are grouped.
        inds : list[int] | None
            Explicit overview row indices.
        purpose : str
            Initial editing purpose: "mask", "roi", "zones", "framelimits",
            "timepoints", "measure", "thresholding". Default "mask".
        """
        from atracker.editor import editor_gui

        # Resolve indices
        _inds = inds
        if names is not None:
            _inds = self.get_inds(names)
        resolved_inds, _ = self.get_files("originals", _inds, query, cats)

        file_infos = []
        for ind in resolved_inds:
            row = self.overview.loc[ind]
            vid_name = str(row.get("video", ""))
            vid_path = os.path.join(self.dirs["originals"], vid_name + ".mp4")

            bgimg = row.get("bgimg")
            bgpath = os.path.join(self.dirs["originals"], bgimg) if isinstance(bgimg, str) else None

            maskimg = row.get("maskimg")
            maskpath = os.path.join(self.dirs["originals"], maskimg) if isinstance(maskimg, str) else None

            zoneimg = row.get("zoneimg")
            zonespath = os.path.join(self.dirs["originals"], zoneimg) if isinstance(zoneimg, str) else None

            roi_val = row.get("roi")
            roi = None
            if isinstance(roi_val, str):
                try:
                    roi = literal_eval(roi_val)
                except Exception:
                    pass

            frame_start = row.get("frame_start")
            frame_start = None if pd.isna(frame_start) or str(frame_start).strip() == "" else int(frame_start)
            frame_stop = row.get("frame_stop")
            frame_stop = None if pd.isna(frame_stop) or str(frame_stop).strip() == "" else int(frame_stop)

            tracked_csv = os.path.join(self.dirs["tracked"], vid_name + ".csv")
            if not os.path.isfile(tracked_csv):
                tracked_csv = None

            thresh_types = row.get("thresh_types")
            thresh_dict = {}
            if isinstance(thresh_types, str):
                for tt in thresh_types.split(","):
                    tt = tt.strip()
                    if tt in self.threshinfo:
                        thresh_dict = self.threshinfo[tt]
                        break

            file_infos.append({
                "ind": ind,
                "video_name": vid_name,
                "video_path": vid_path if os.path.isfile(vid_path) else None,
                "background_path": bgpath if (bgpath and os.path.isfile(bgpath)) else None,
                "mask_path": maskpath if (maskpath and os.path.isfile(maskpath)) else None,
                "zones_path": zonespath if (zonespath and os.path.isfile(zonespath)) else None,
                "roi": roi,
                "frame_start": frame_start,
                "frame_stop": frame_stop,
                "tracked_csv": tracked_csv,
                "threshold_dict": thresh_dict,
                "dirs": self.dirs,
            })

        if not file_infos:
            lineprint("No files found for editor.")
            return

        overview_dirty = False

        def save_callback(file_idx, ind, purpose_key, data):
            nonlocal overview_dirty
            fi = file_infos[file_idx]
            vid_name = fi["video_name"]

            if purpose_key == "mask":
                if isinstance(data, np.ndarray):
                    outname = f"{vid_name}_mask.jpg"
                    outpath = os.path.join(self.dirs["originals"], outname)
                    cv2.imwrite(outpath, data)
                    self.overview.loc[ind, "maskimg"] = outname
                    overview_dirty = True
                    lineprint(f"Stored mask: {outname}")

            elif purpose_key == "roi":
                if data:
                    self.overview.loc[ind, "roi"] = str(data)
                    overview_dirty = True
                    lineprint(f"Stored ROI: {data}")

            elif purpose_key == "zones":
                if isinstance(data, np.ndarray):
                    outname = f"{vid_name}_zone.jpg"
                    outpath = os.path.join(self.dirs["originals"], outname)
                    cv2.imwrite(outpath, data)
                    if "zoneimg" not in self.overview.columns:
                        self.overview["zoneimg"] = pd.Series(dtype=object)
                    self.overview.loc[ind, "zoneimg"] = outname
                    overview_dirty = True
                    lineprint(f"Stored zones: {outname}")

            elif purpose_key == "framelimits":
                if data:
                    self.overview.loc[ind, "frame_start"] = data[0]
                    self.overview.loc[ind, "frame_stop"] = data[1]
                    overview_dirty = True
                    lineprint(f"Stored frame limits: start={data[0]}, stop={data[1]}")

            elif purpose_key == "timepoints":
                if data is not None and hasattr(data, "to_csv"):
                    csv_path = os.path.join(self.dirs["tracked"], vid_name + ".csv")
                    data.to_csv(csv_path, index=False)
                    lineprint(f"Stored coordinate data: {os.path.basename(csv_path)}")

            elif purpose_key == "thresholding":
                if isinstance(data, dict):
                    thresh_types = self.overview.loc[ind].get("thresh_types")
                    if isinstance(thresh_types, str):
                        tt = thresh_types.split(",")[0].strip()
                        self.threshinfo[tt] = data
                        with open(self.cfiles["threshinfo"], "w") as f:
                            yaml.safe_dump(self.threshinfo, f, default_flow_style=False)
                        lineprint(f"Stored thresholding for {tt}")

        editor_gui(file_infos, purpose=purpose, save_callback=save_callback)

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
        max_framedist=None, overwrite=None, check_flicker=None, skip_frames=None, watch=False, watch_interval=60):

        check_flicker = check_flicker if check_flicker is not None else bool(getattr(self.config.track, 'check_flicker', False))
        skip_frames = skip_frames if skip_frames is not None else int(getattr(self.config.track, 'skip_frames', 0))
        max_framedist = max_framedist if max_framedist is not None else int(getattr(self.config.track, 'max_framedist', 200))

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

            # Pre-filter already-tracked files when overwrite is off
            effective_overwrite = overwrite if overwrite is not None else self.config.track.overwrite
            if not effective_overwrite:
                suffix_str = f"_{suffix}" if suffix else ""
                todo_inds, todo_trackfiles = [], []
                skip_count = 0
                for ind, tf in zip(_inds, trackfiles):
                    video = self.overview.loc[ind, "video"]
                    region = self.overview.loc[ind, "region"] if "region" in self.overview.columns else None
                    name = f"{video}_R{int(region)}" if (region is not None and pd.notnull(region)) else video
                    csv_path = os.path.join(self.dirs["tracked"], name + suffix_str + ".csv")
                    if os.path.exists(csv_path):
                        skip_count += 1
                    else:
                        todo_inds.append(ind)
                        todo_trackfiles.append(tf)
                if skip_count > 0:
                    lineprint(f"Skipped {skip_count} already-tracked file(s).")
                _inds = todo_inds
                trackfiles = todo_trackfiles

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
                track_items = [(ind, os.path.join(self.dirs[folder], self.overview.loc[ind]["video"] + ".mp4"))
                               for ind in T.inds]
                if not run_pool(T.setuptracking, track_items, pools=pools, mode="process", label="tracking"):
                    _stop = True

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

    def process(self, pools=1, names=None, overwrite=False,
                fulldata=True, convert=True, changefps=None,
                mask_margin=15, max_traj_gap=50, min_traj_len=10,
                roi_edge_margin=10,
                interp_gap_com=500, interp_gap_orient=100,
                smoothwin=10, orient_min_speed=1,
                interpolate=True, compute_movement=True, compute_distances=True):
        """
        Post-process tracked CSV files into analysis-ready data.

        Parameters
        ----------
        pools : int, default 1
            Number of parallel workers.
        names : list or None
            Specific file names to process; processes all tracked files if None.
        overwrite : bool, default False
            Overwrite existing processed files.
        fulldata : bool, default True
            Extend output to every tracked frame in the video window
            (untracked frames → NaN).
        convert : bool, default True
            Convert pixels to real-world units using the conv factor in the overview.
        changefps : int or None
            Resample output to a lower frame rate (must be ≤ video fps).
        mask_margin : int, default 15
            Pixel distance from the mask: detections within this distance are
            removed; gaps within this distance are not interpolated.
        max_traj_gap : int, default 50
            Frame gap above which a break splits into a new trajectory.
        min_traj_len : int, default 10
            Minimum trajectory length; shorter trajectories and isolated bursts
            of the same length are discarded.
        roi_edge_margin : int, default 10
            Pixels from the ROI edge: head/tail blanked within this distance;
            ROI exits within this distance are not interpolated.
        interp_gap_com : int, default 500
            Maximum frame gap to interpolate centroid data over.
        interp_gap_orient : int, default 100
            Maximum frame gap to interpolate head/tail and orientation over.
        smoothwin : int, default 10
            Savitzky–Golay smoothing window in frames (1 = no smoothing).
        orient_min_speed : float, default 1
            Minimum speed above which heading is used as fallback for orientation.
        interpolate : bool, default True
            Interpolate gaps in centroid, head/tail, and orientation data.
            Set to False to keep only raw detections with no gap-filling.
        compute_movement : bool, default True
            Compute movement variables: displacement, speed, acceleration,
            heading, orientation, and turn rates.
        compute_distances : bool, default True
            Compute distance measures: ROI edge, mask, walls, zones, custom points.
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

        config_dict = {
            "dirs": self.dirs,
            "config": self.config,
            "overview": self.overview,
            "trackedfiles": trackedfiles,
            "orientfrombw": bool(getattr(self.config.track, 'orientfrombw', False)),
            "overwrite": overwrite,
            "fulldata": fulldata,
            "convert": convert,
            "changefps": changefps,
            "mask_margin": mask_margin,
            "max_traj_gap": max_traj_gap,
            "min_traj_len": min_traj_len,
            "roi_edge_margin": roi_edge_margin,
            "interp_gap_com": interp_gap_com,
            "interp_gap_orient": interp_gap_orient,
            "smoothwin": smoothwin,
            "orient_min_speed": orient_min_speed,
            "interpolate": interpolate,
            "compute_movement": compute_movement,
            "compute_distances": compute_distances,
        }

        lineprint("Processing started of " + str(len(trackedfiles)) + " files..")

        def _proc_worker(f):
            Processor(**config_dict).setup(f, threading.get_ident() if pools > 1 else None)

        run_pool(_proc_worker, trackedfiles, pools=pools, mode="thread", label="processing")


    def visualise(self, folder="processed", names=None, overwrite=False, pools=1, **kwargs):
        """
        Create visualisation videos for tracked or processed CSV files.

        Parameters
        ----------
        folder : str, default "processed"
            Source folder containing CSVs: "processed" or "tracked".
        names : list or str or None
            Specific base names (without extension) to visualise; all files
            in the folder if None.
        overwrite : bool, default False
            Overwrite existing visualisation videos.
        pools : int, default 1
            Number of parallel workers. 1 = sequential.
        **kwargs
            Additional arguments forwarded to the Visualiser,
            e.g. resize, trajlength, writevideo, showvideo, drawptonmask, etc.
        """
        csvfiles = listfiles(self.dirs[folder], type=".csv", keepdir=True)
        csvfiles = [os.path.normpath(f) for f in csvfiles]

        if names is not None:
            if isinstance(names, str):
                names = [names]
            names_set = {os.path.splitext(os.path.basename(n))[0] for n in names}
            csvfiles = [f for f in csvfiles if os.path.splitext(os.path.basename(f))[0] in names_set]

        if not csvfiles:
            lineprint("No CSV files found to visualise..")
            return

        lineprint(f"Visualising {len(csvfiles)} file(s) from '{folder}'..")

        def _img_path(fname):
            if not isinstance(fname, str):
                return None
            p = os.path.join(self.dirs["originals"], fname)
            return p if os.path.isfile(p) else None

        vis_items = []
        for i, csvfile in enumerate(csvfiles):
            base = os.path.splitext(os.path.basename(csvfile))[0]
            tracking_base = base[:-2] if (folder == "processed" and base.endswith("_F")) else base

            region_match = re.search(r'_R(\d+)$', tracking_base)
            if region_match:
                region = int(region_match.group(1))
                video_name = tracking_base[:region_match.start()]
            else:
                region = None
                video_name = tracking_base

            rows = self.overview[self.overview["video"] == video_name]
            if "region" in rows.columns and region is not None:
                rows = rows[rows["region"] == region]
            if len(rows) == 0:
                lineprint(f"Video {i+1}|{len(csvfiles)} {base}: no overview row found, skipping")
                continue
            row = self.overview.loc[rows.index[0]]

            outfile = os.path.join(self.dirs[folder], base + "_V.mp4")
            if os.path.exists(outfile) and not overwrite:
                lineprint(f"Video {i+1}|{len(csvfiles)} {base}: already exists, skipping")
                continue

            orig_video = os.path.join(self.dirs["originals"], f"{video_name}.mp4")
            if not os.path.isfile(orig_video):
                lineprint(f"Video {i+1}|{len(csvfiles)} {base}: original video not found, skipping")
                continue

            try:
                fps_val = float(row["fps"]) if not pd.isna(row.get("fps", np.nan)) else 25.0
            except (TypeError, ValueError):
                fps_val = 25.0

            roi = None
            if "roi" in self.overview.columns and isinstance(row.get("roi"), str):
                try:
                    roi = literal_eval(row["roi"])
                except Exception:
                    pass

            vis_items.append({
                "csvfile": csvfile,
                "outfile": outfile,
                "orig_video": orig_video,
                "fps": fps_val,
                "roi": roi,
                "bgimg_path": _img_path(row.get("bgimg")),
                "maskimg_path": _img_path(row.get("maskimg")),
                "wallimg_path": _img_path(row.get("wallimg")),
                "zoneimg_path": _img_path(row.get("zoneimg")),
                "config": self.config,
                "label": f"Video {i+1}|{len(csvfiles)} {base}",
                "kwargs": kwargs,
            })

        def _vis_worker(info):
            img_bg = cv2.imread(info["bgimg_path"]) if info["bgimg_path"] else None
            img_mask = cv2.imread(info["maskimg_path"]) if info["maskimg_path"] else None
            wallconts = None
            if info["wallimg_path"]:
                wall_arr = cv2.imread(info["wallimg_path"])
                if wall_arr is not None:
                    wallconts, _ = coordsfrommask(wall_arr)
            zone_coords = None
            if info["zoneimg_path"]:
                zone_arr = cv2.imread(info["zoneimg_path"])
                if zone_arr is not None:
                    _zc = coordsfromzones(zone_arr, all_parts=True)
                    zone_coords = _zc if _zc else None
            lineprint(info["label"], True, False)
            data = pd.read_csv(info["csvfile"])
            vis = _Visualiser(wall_contours=wallconts, zone_coords=zone_coords, config=info["config"])
            call_kwargs = dict(fps=info["fps"])
            call_kwargs.update(info["kwargs"])
            vis.render(data=data, videofile=info["orig_video"], img_bg=img_bg, img_mask=img_mask,
                       roi=info["roi"], outfile=info["outfile"], **call_kwargs)

        run_pool(_vis_worker, vis_items, pools=pools, mode="thread", label="visualising")

    def centralise(self, centertype="roi", names=None):
        """
        Subtract the arena centre from converted coordinates in processed files.

        Modifies cx_c, cy_c (and fx_c, fy_c if present) in-place in each
        processed CSV so that (0, 0) is the arena centre.

        Parameters
        ----------
        centertype : str, default "roi"
            How to compute the centre: "roi" uses the ROI midpoint, "walls" uses
            the mean of wall contour coordinates, "pt" uses the pt column in the
            overview.
        names : list or None
            Specific base names to centralise; all processed files if None.
        """
        raise NotImplementedError(
            "centralise() is not yet implemented. "
            "Apply arena-centre subtraction manually on the processed CSV."
        )