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
from .helpers.logging import lineprint
from pythutils.drawutils import namedcols
from pythutils.mediautils import get_vid_params, check_media

from atracker.__version__ import __version__
from atracker.track import Tracker, recover_tracking_sources
from atracker.process import Processor
from atracker.helpers.media import convert_h264_to_mp4, bg_extract, find_max_working_pyframe
from atracker.helpers.data import duplicate_row
from atracker.helpers.contours import coordsfrommask, coordsfromzones
from atracker.helpers.pool import run_pool, BatchInterrupted
from atracker.helpers.naming import output_basename
from atracker.visualise import Visualiser as _Visualiser

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')


def _git_version_info():
    """Describe the last commit and local package edits, when Git is available."""
    try:
        pkg_dir = os.path.dirname(__file__)
        result = subprocess.run(
            ["git", "log", "-1", "--format=%h %cd", "--date=short"],
            capture_output=True, text=True, timeout=2, cwd=pkg_dir
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split()
            status = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=normal", "--", "."],
                capture_output=True, text=True, timeout=2, cwd=pkg_dir)
            local = " + local changes" if status.returncode == 0 and status.stdout.strip() else ""
            return f"last commit {parts[0]} ({parts[1]}){local}"
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
        lineprint(version_str)
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
                body_mode="centroid", head_rule="near_widest", skeleton_points=10,
                save_body=False, link_dist=100, merge_dist=20,
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
                draw_contournrs=False, trajs_below=False, draw_zones=False, internal="")
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
            if "advanced" in track_keys:
                old = dict(self.config.items("track"))
                self.config.track.body_mode = old.get("body_mode", "skeleton" if old["advanced"] else "centroid")
                self.config.save()
                path = self.cfiles["config"]
                with open(path) as stream:
                    contents = stream.read()
                with open(path, "w") as stream:
                    stream.write(re.sub(r"(?m)^[ \t]*advanced[ \t]*=.*(?:\n|$)", "", contents))
                self.config = LocalConfig(path, compact_form=True)
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
            body_mode="skeleton" if advanced else "centroid", link_dist=link_dist, merge_dist=merge_dist,
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
            draw_contournrs=False, trajs_below=False, draw_zones=False, internal="")

        print(f"\n{'='*60}")
        print("CONFIG MIGRATION: Legacy config format detected!")
        print(f"  Backup saved as: {os.path.basename(backup_path)}")
        print("  Track settings migrated:")
        print(f"    body_mode={'skeleton' if advanced else 'centroid'}")
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
        body_mode : str, default "centroid"
            "centroid" uses the blob centroid; "skeleton" uses the centreline midpoint and tips.
        head_rule : str, default "near_widest"
            "near_widest", "near_centroid", "far_centroid" or "unassigned".
        skeleton_points : int, default 10
            Equally spaced centreline points per object in optional geometry output.
        save_body : bool, default False
            Save per-frame centreline/contour geometry to a compressed JSONL sidecar.
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
            Project default for the minimum rotated-box long/short side ratio.
            A threshold configuration's explicit value overrides this default.
        max_aspect_ratio : float, default 10
            Project default for the maximum ratio; threshold-specific values override it.
        check_flicker : bool, default False
            Reject B/W detection passes with unusually strong ROI-wide darkening.
            The cutoff is 20 times the median mean difference from up to 100 sampled
            frames, using the same gamma as detection. Not applied to HSV tracking.
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
        draw_zones : bool, default False
            Draw zone fills and labels in the live tracking display and saved
            tracking/visualisation videos. Does not affect zone processing.
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

        if "advanced" in kwargs or "simple" in kwargs:
            raise ValueError('Use body_mode="centroid" or body_mode="skeleton"')
        from atracker.helpers.body import body_settings
        body_settings(kwargs.get("body_mode", (getattr(self.config.track, "body_mode", None) or "centroid")),
                      kwargs.get("head_rule", (getattr(self.config.track, "head_rule", None) or "near_widest")),
                      kwargs.get("skeleton_points", (getattr(self.config.track, "skeleton_points", None) or 10)))

        # track section
        _track_map = {
            "overwrite": "overwrite",
            "create_vid": "create_vid",
            "create_dat": "create_dat",
            "body_mode": "body_mode",
            "head_rule": "head_rule",
            "skeleton_points": "skeleton_points",
            "save_body": "save_body",
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
            "draw_zones": "draw_zones",
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
            bgname = output_basename(self.overview, idx) + "_bg.jpg"

            bgpath = os.path.join(self.dirs["originals"], bgname)

            # we will extract per row
            if overwrite or not os.path.isfile(bgpath):
                todolist.append((idx, video, bgname, bgpath))
        
        # 3) Extract backgrounds
        if len(todolist) == 0:
            print("All files done..", end=" ")
        else:
            for k, (idx, video, bgname, bgpath) in enumerate(todolist):

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

    def export_geometry(self, names=None, inds=None, query=None, cats=None,
                        output=None, zone_names=None, epsilon=0, overwrite=False):
        """Export portable full-frame polygons, calibration and file references.

        Saved as <video>[_R<region>].geometry.json in originals by default.
        zone_names maps numeric palette IDs to names; existing exports are kept
        unless overwrite=True. Tracking/processing still use the source images.
        """
        from atracker.geometry import export_geometry
        return export_geometry(self, names=names, inds=inds, query=query, cats=cats,
            output=output, zone_names=zone_names, epsilon=epsilon, overwrite=overwrite)

    def event_checker(self, csv_file, frame_base=1, modifier=None, unreviewed_only=False):
        """Review CSV event windows against project originals.

        modifier adds a named checkbox, for example modifier="complete", saved
        as true/false in its own CSV column. unreviewed_only selects blank reviews.
        """
        from atracker.editor import event_checker
        return event_checker(csv_file, self.dirs["originals"], frame_base=frame_base, modifier=modifier,
                             unreviewed_only=unreviewed_only)

    def editor(self, names=None, query=None, cats=None, inds=None, purpose=None, threshtypes=None,
               candidates=None, frame_base=1, modifier=None, unreviewed_only=False):
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
            "timepoints", "measure", "thresholding", "thresholding color", "event_checker".
            Defaults to threshold editing when threshtypes is supplied, otherwise "mask".
        candidates : str | None
            MoveR candidate CSV for purpose="event_checker"; selects its own videos.
        modifier : str | None
            Optional event-checkbox name, saved as its own CSV column.
        frame_base : int
            Candidate CSV frame numbering (1 by default; 0 accepted explicitly).
        threshtypes : list[str] | str | None
            Threshold configurations to load or create instead of using the overview
            assignments. Save current writes the selected entry to the project's
            threshold YAML file; overview assignments are unchanged.
        """
        if purpose == "event_checker":
            if candidates is None:
                raise ValueError("Supply candidates=path_to_csv for event_checker")
            if any(value is not None for value in (names, query, cats, inds, threshtypes)):
                raise ValueError("Event checker selects videos from its candidate CSV; omit other selectors")
            return self.event_checker(candidates, frame_base=frame_base, modifier=modifier,
                                      unreviewed_only=unreviewed_only)
        if candidates is not None:
            raise ValueError("candidates is only supported with purpose='event_checker'")
        from atracker.editor import editor_gui
        from atracker.helpers.detection_settings import aspect_limits

        default_min_ar, default_max_ar = aspect_limits({}, self.config.track)

        if threshtypes is not None:
            threshtypes = [threshtypes] if isinstance(threshtypes, str) else list(threshtypes)
            if not threshtypes or any(not isinstance(t, str) or not t.strip() for t in threshtypes):
                raise ValueError("threshtypes must contain at least one nonempty configuration name")
            threshtypes = list(dict.fromkeys(t.strip() for t in threshtypes))
        if purpose is None:
            purpose = ("thresholding" if threshtypes[0].lower().startswith("bw") else "thresholding color") if threshtypes else "mask"

        # Resolve indices
        _inds = inds
        if names is not None:
            _inds = self.get_inds(names)
        resolved_inds, _ = self.get_files("originals", _inds, query, cats)

        from atracker.editor._project import editor_files, save_editor_data
        file_infos = editor_files(self, resolved_inds, threshtypes,
                                  (default_min_ar, default_max_ar))

        if not file_infos:
            lineprint("No files found for editor.")
            return

        def save_callback(file_idx, ind, purpose_key, data):
            save_editor_data(self, file_infos, file_idx, ind, purpose_key, data)

        editor_gui(file_infos, purpose=purpose, save_callback=save_callback)

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

    def watch_folder(self, folder, template_ind, poll_interval=30, settle_seconds=60,
                     extract_background=True):
        """Discover new MP4s and track them using a configured overview row.

        Reuses ROI, calibration, masks, walls, zones, objects and threshold names;
        reads FPS/dimensions/frame count and extracts a new background per video.
        Frame limits cover the full new recording. Stops with Ctrl+C.
        """
        from .helpers.watch import watch_folder
        return watch_folder(self, folder, template_ind, poll_interval, settle_seconds,
                            extract_background)

    def track(self, inds=None, names=None, query=None, cats=None, pools=1, folder="todo", frame_start=None,
        frame_stop=None, threshtype=None, objects=None, checkconschange=False, suffix="", threshfile=None,
        max_framedist=None, overwrite=None, check_flicker=None, skip_frames=None, watch=False, watch_interval=60,
        profile_interval=0):

        if not isinstance(profile_interval, int) or profile_interval < 0:
            raise ValueError("profile_interval must be a non-negative integer")

        check_flicker = check_flicker if check_flicker is not None else bool(getattr(self.config.track, 'check_flicker', False))
        skip_frames = skip_frames if skip_frames is not None else int(getattr(self.config.track, 'skip_frames', 0))
        max_framedist = max_framedist if max_framedist is not None else int(getattr(self.config.track, 'max_framedist', 200))

        if watch:
            lineprint(f"Watch mode enabled — checking every {watch_interval}s (Ctrl+C to stop)..")

        _stop = False
        success = True
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

            # Give workers a serialisable snapshot without changing the project's configuration.
            cbak = self.config
            tracking_config = Box({s: {k:v for (k,v) in cbak.items(s)} for s in cbak})

            existing = [os.path.exists(f) for f in trackfiles]
            existing_inds = [i for i, e in zip(_inds, existing) if e]
            existing_trackfiles = [f for f, e in zip(trackfiles, existing) if e]
            missing_count = len(trackfiles) - len(existing_trackfiles)
            if missing_count > 0:
                lineprint(f"Skipped {missing_count} missing video file(s).")
                for path, exists in zip(trackfiles, existing):
                    if not exists:
                        lineprint(f"Missing source: {path}")
            _inds = existing_inds
            trackfiles = existing_trackfiles

            # Pre-filter already-tracked files when overwrite is off
            effective_overwrite = overwrite if overwrite is not None else tracking_config.track.overwrite
            if not effective_overwrite:
                suffix_str = f"_{suffix}" if suffix else ""
                todo_inds, todo_trackfiles = [], []
                skip_count = 0
                for ind, tf in zip(_inds, trackfiles):
                    name = output_basename(self.overview, ind)
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
                        tracking_config, self.threshinfo, frame_start, frame_stop, threshtype,
                        objects, checkconschange, suffix,
                        max_framedist=max_framedist, overwrite=overwrite,
                        check_flicker=check_flicker, skip_frames=skip_frames,
                        profile_interval=profile_interval)

            count = len(T.inds)
            workers = min(max(1, pools), count)
            lineprint(f"Tracking {count} file{'s' if count != 1 else ''} with "
                      f"{workers} worker{'s' if workers != 1 else ''}.")

            if len(T.inds) == 0:
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
                failed_rows = []
                while len(T.inds) > 0 and not stop:
                    ind = T.inds[0]
                    trackfile = os.path.join(self.dirs[folder], self.overview.loc[ind]["video"] + ".mp4")
                    try:
                        T.setuptracking(ind, trackfile)
                    except (KeyboardInterrupt, BatchInterrupted):
                        lineprint("\nUser terminated tracking..")
                        stop = True
                        success = False
                        _stop = True
                    except Exception as e:
                        video = self.overview.loc[ind]["video"]
                        lineprint(f"Error on row {ind} ({video}): {type(e).__name__}: {e} — skipping")
                        failed_rows.append(ind)
                        if ind in T.inds:
                            T.inds.remove(ind)
                if not _stop:
                    lineprint(f"Tracking finished; failed rows: {failed_rows or 'none'}.")
                if failed_rows:
                    success = False
                    _stop = True
            else:
                tracking_config.vis.show_tracking = False
                tracking_config.vis.waitkey = 1
                track_items = [(ind, os.path.join(self.dirs[folder], self.overview.loc[ind]["video"] + ".mp4"))
                               for ind in T.inds]
                groups = {}
                for item in track_items:
                    groups.setdefault(item[1], []).append(item)
                try:
                    if not run_pool(T.track_group, list(groups.values()), pools=pools, mode="process", label="tracking"):
                        success = False
                        _stop = True
                finally:
                    recover_tracking_sources(track_items, self.dirs)


            if not watch or _stop:
                _stop = True
            else:
                lineprint(f"Watch: pass complete, sleeping {watch_interval}s..")
                try:
                    time.sleep(watch_interval)
                except KeyboardInterrupt:
                    lineprint("Watch mode stopped.")
                    _stop = True

        return success

    def process(self, pools=1, names=None, overwrite=False,
                fulldata=True, convert=True, changefps=None,
                mask_margin=15, max_traj_gap=50, min_traj_len=10,
                roi_edge_margin=10,
                interp_gap_com=500, interp_gap_orient=100,
                smoothwin=10, orient_min_speed=1,
                interpolate=True, compute_movement=True, compute_distances=True,
                orientation_fallback="none", correct_head_tail=True):
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
        orientation_fallback : str, default "none"
            Set "heading" to explicitly allow movement-derived orientation.
        correct_head_tail : bool, default True
            Correct isolated head/tail transpositions before interpolation.
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
        missing_names = []

        for base_name in base_names:
            # Check if _E.csv version exists for the current base name
            e_file = os.path.normpath(os.path.join(self.dirs["tracked"], base_name + "_E.csv"))
            original_file = os.path.normpath(os.path.join(self.dirs["tracked"], base_name + ".csv"))
            
            # Prefer the _E.csv file if it exists, otherwise use the original .csv file
            if e_file in trackedfiles:
                selected_files.add(e_file)  # Add _E file
            elif original_file in trackedfiles:
                selected_files.add(original_file)  # Add original file if no _E file
            else:
                missing_names.append(base_name)
        if missing_names:
            lineprint(f"Missing tracked CSVs: {', '.join(missing_names)}")

        # Convert set to a list for final output
        trackedfiles = sorted(list(selected_files))
        if len(trackedfiles) == 0:
            lineprint("No tracked CSV files found to process.")
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
            "orientation_fallback": orientation_fallback,
            "correct_head_tail": correct_head_tail,
            "interpolate": interpolate,
            "compute_movement": compute_movement,
            "compute_distances": compute_distances,
        }

        lineprint("Processing started of " + str(len(trackedfiles)) + " files..")

        def _proc_worker(f):
            Processor(**config_dict).setup(f, threading.get_ident() if pools > 1 else None)

        return run_pool(_proc_worker, trackedfiles, pools=pools, mode="thread", label="processing")


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

        return run_pool(_vis_worker, vis_items, pools=pools, mode="thread", label="visualising")

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
