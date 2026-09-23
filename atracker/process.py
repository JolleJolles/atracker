#! /usr/bin/env python

import re
import os
import numpy as np
import pandas as pd
import multiprocessing
from ast import literal_eval
from math import gcd
from functools import reduce

from pythutils.mathutils import points_to_angle, angle_to_vec, ptsToDist

from .helpers.geometry import (calc_borderdistdf, convert, dist_to_point, dist_to_poly,
                               dist_to_rect, dist_to_zone, is_axis_aligned_rectangle,
                               series_to_point_tuple, valid_img_path, get_anglediff)
from .helpers.trajectory import (calcudiff, differentiate, fillmissing, getalones,
                                  process_trajectories, smooth)
from .helpers.contours import coordsfrommask, coordsfromzones
from scipy.spatial import KDTree
from .helpers.data import ensure_columns, lit_converter


class Processor:

    def __init__(self, dirs, config, overview, trackedfiles, orientfrombw, overwrite,
                 fulldata=True, convert=True, datflip=True,
                 changefps=None,
                 mask_margin=15, max_traj_gap=50, min_traj_len=10,
                 roi_edge_margin=10,
                 interp_gap_com=500, interp_gap_orient=100,
                 smoothwin=10, orient_min_speed=1,
                 interpolate=True, compute_movement=True, compute_distances=True):
        """
        Post-processing pipeline for tracked coordinate data.

        Parameters
        ----------
        dirs : dict
            Directory paths used by the tracker.
        config : Box
            Project configuration.
        overview : pd.DataFrame
            Overview file with per-video metadata.
        trackedfiles : list of str
            Paths to the raw tracked CSV files.
        orientfrombw : bool
            If True, use the bw tracking angle column as orientation instead of
            head/tail coordinates.
        overwrite : bool
            Overwrite existing processed files.
        fulldata : bool
            Extend output to cover every tracked frame from frame_start to
            frame_stop; untracked frames are filled with NaN (default True).
        convert : bool
            Convert pixel coordinates to real-world units using the conv factor
            from the overview file (default True).
        datflip : bool
            Flip the y-axis on conversion so coordinates match video orientation
            (default True).
        changefps : int or None
            Resample output to a lower frame rate (must be ≤ video fps).
        mask_margin : int
            Pixel distance from the mask boundary used for two purposes: detections
            within this distance are removed as mask-covered; gaps within this
            distance are not interpolated (default 15).
        max_traj_gap : int
            Frame gap above which a break in detections starts a new trajectory
            (default 50).
        min_traj_len : int
            Minimum trajectory length in frames; shorter trajectories — and
            isolated detection bursts of the same length — are discarded (default 10).
        roi_edge_margin : int
            Pixels from the ROI edge; detections closer than this have their
            head/tail coordinates blanked, and ROI exits are not interpolated
            (default 10).
        interp_gap_com : int
            Maximum gap (frames) to interpolate centroid data over (default 500).
        interp_gap_orient : int
            Maximum gap (frames) to interpolate head/tail and orientation over
            (default 100).
        smoothwin : int
            Savitzky–Golay smoothing window in frames (default 10; 1 = no smoothing).
        orient_min_speed : float
            Minimum speed (converted units/frame) above which heading is used as a
            fallback for missing orientation (default 1).
        interpolate : bool
            Interpolate gaps in centroid, head/tail, and orientation data (default True).
            Set to False to keep only raw detections with no gap-filling.
        compute_movement : bool
            Compute movement variables: displacement, speed, acceleration, heading,
            orientation, turn rates (default True).
        compute_distances : bool
            Compute distance measures: ROI edge, mask, walls, zones, custom points
            (default True).
        """
        self.dirs = dirs
        self.config = config
        self.overview = overview
        self.overview.roi = [literal_eval(i) if isinstance(i, str) else i
                             for i in self.overview.roi]
        self.trackedfiles = trackedfiles
        self.appendtxt = "_F.csv"
        self.orientfrombw = orientfrombw
        self.overwrite = overwrite
        self.fulldata = fulldata
        self.convert = convert
        self.datflip = datflip
        self.changefps = changefps
        self.mask_margin = mask_margin
        self.max_traj_gap = max_traj_gap
        self.min_traj_len = min_traj_len
        self.roi_edge_margin = roi_edge_margin
        self.interp_gap_com = interp_gap_com
        self.interp_gap_orient = interp_gap_orient
        self.smoothwin = smoothwin
        self.orient_min_speed = orient_min_speed
        self.interpolate = interpolate
        self.compute_movement = compute_movement
        self.compute_distances = compute_distances


    # ------------------------------------------------------------------
    # Setup and preparation
    # ------------------------------------------------------------------

    def _log(self, msg):
        print(f"  [{self.nr}] {msg}", flush=True)

    def _unpack_coords(self, column, newcols):
        """Unpack a tuple column (e.g. 'com') into two separate float columns."""
        if column in self.data:
            mask = pd.isnull(self.data[column])
            self.data.loc[mask, column] = [(np.nan, np.nan)] * mask.sum()
            if self.data[column].notna().sum() == 0:
                self.data[newcols[0]] = np.nan
                self.data[newcols[1]] = np.nan
            else:
                self.data[newcols[0]], self.data[newcols[1]] = zip(*self.data[column])
                self.data[newcols[0]] = pd.to_numeric(self.data[newcols[0]], errors="coerce").astype(float)
                self.data[newcols[1]] = pd.to_numeric(self.data[newcols[1]], errors="coerce").astype(float)
            self.data = self.data.drop(columns=[column])


    def setup(self, trackedfile, thread_id):

        self.trackedfile = trackedfile
        self.thread_id = thread_id
        self.nr = str(self.trackedfiles.index(self.trackedfile))
        self.filebase = os.path.splitext(os.path.basename(self.trackedfile))[0]
        if self.appendtxt in self.trackedfile:
            self.procfile = self.trackedfile
        else:
            if "_E" in self.filebase:
                self.procfile = os.path.join(self.dirs["processed"],
                                             self.filebase.replace("_E", "") + self.appendtxt)
            else:
                self.procfile = os.path.join(self.dirs["processed"],
                                             self.filebase + self.appendtxt)

        pr_procid = multiprocessing.current_process()._identity
        pr_procid = "{P" + str(pr_procid[0]) + "}" if len(pr_procid) > 0 else ""
        self.pr_comm = pr_procid + "[" + self.nr + "] " + self.filebase + " "

        if os.path.isfile(self.procfile) and not self.overwrite:
            return

        print(f"{self.pr_comm.strip()} — processing", flush=True)

        # Locate this file's overview row
        if "_E" not in self.filebase[-5:] and "_Z" not in self.filebase[-5:] and "_R" not in self.filebase[-5:]:
            self.ind = list(self.overview[(self.overview.video == self.filebase)].index.values)[0]
        if "_E" in self.filebase[-5:]:
            self.ind = list(self.overview[(self.overview.video == self.filebase[:-2])].index.values)[0]
        if "_Z" in self.filebase[-5:]:
            self.ind = list(self.overview[(self.overview.video == self.filebase[:-3]) &
                                          (self.overview.zone == int(self.filebase[-1:]))].index.values)[0]
        if "_R" in self.filebase[-5:]:
            self.ind = list(self.overview[(self.overview.video == self.filebase[:-3]) &
                                          (self.overview.region == int(self.filebase[-1:]))].index.values)[0]

        # Load tracking data
        self.data = pd.read_csv(self.trackedfile, header=0)
        if len(self.data) > 0 and "_E" not in self.filebase[-5:]:
            converters = {c: lit_converter for c in self.data.columns
                          if str(self.data[c].iloc[0]).startswith("(")}
            self.data = pd.read_csv(self.trackedfile, header=0, converters=converters)
            self.emptydat = False
        else:
            self.emptydat = True
        self._log(f"loaded {len(self.data)} rows")

        self.ids = [i for i in pd.unique(self.data.id)]

        # File metadata
        self.fileinfo = self.overview.loc[self.ind]
        if self.fulldata:
            self.minfr = int(self.fileinfo.frame_start
                             if self.fileinfo.frame_start == self.fileinfo.frame_start
                             else 1)
            self.maxfr = int(self.fileinfo.fcount
                             if self.fileinfo.frame_stop != self.fileinfo.frame_stop
                             else self.fileinfo.frame_stop)
        else:
            self.minfr, self.maxfr = int(min(self.data["frame"])), int(max(self.data["frame"]))
        self.fps = self.fileinfo.fps
        self.conv = float(self.fileinfo.conv) if (self.convert and pd.notna(self.fileinfo.conv)) else 1
        if "ID" not in self.fileinfo or pd.isna(self.fileinfo.ID):
            self.IDs = list(self.data.id.unique())
        else:
            IDval = self.fileinfo.ID
            if isinstance(IDval, str):
                if IDval.startswith(("(", "[")):
                    try:
                        self.IDs = list(literal_eval(IDval))
                    except Exception:
                        self.IDs = IDval.strip("()[]").split(",")
                elif "," in IDval:
                    self.IDs = [s.strip() for s in IDval.split(",")]
                else:
                    self.IDs = [IDval.strip()]
            else:
                self.IDs = list(IDval) if hasattr(IDval, "__iter__") else [str(IDval)]
        self.res = literal_eval(self.fileinfo.resolution)
        self.cover = pd.notna(self.fileinfo["maskimg"])
        self.height = self.fileinfo.roi[1][1] - self.fileinfo.roi[0][1]

        # Detect actual frame sampling interval (1 = every frame, 2 = every other, etc.)
        if not self.emptydat and len(self.data) >= 2:
            diffs = np.diff(sorted(self.data["frame"].unique())).tolist()
            self.frame_step = int(reduce(gcd, diffs)) if diffs else 1
        else:
            self.frame_step = 1

        self.prep()

        # Load reference images
        self.bg_path       = valid_img_path(self.fileinfo, "bgimg",       self.dirs["originals"])
        self.mask_path     = valid_img_path(self.fileinfo, "maskimg",     self.dirs["originals"])
        self.wall_path     = valid_img_path(self.fileinfo, "wallimg",     self.dirs["originals"])
        self.zone_path     = valid_img_path(self.fileinfo, "zoneimg",     self.dirs["originals"])
        self.maskzone_path = valid_img_path(self.fileinfo, "maskzoneimg", self.dirs["originals"])
        self.maskcoords     = []
        self.wallcoords     = []
        self.zonecoords     = {}
        self.maskzonecoords = []
        if self.mask_path:
            _, self.maskcoords = coordsfrommask(self.mask_path)
        if self.wall_path:
            _, self.wallcoords = coordsfrommask(self.wall_path)
        if self.zone_path:
            self.zonecoords = coordsfromzones(self.zone_path)
        if self.maskzone_path:
            _, self.maskzonecoords = coordsfrommask(self.maskzone_path)

        self.process()


    def prep(self):
        """Prepare raw tracking data: unpack coordinates, filter frames, expand to
        full frame range, and remove short isolated fragments."""

        # Implicitly drop bw-contour rows when colour IDs are also present
        if not self.emptydat:
            str_ids = self.data["id"].astype(str)
            has_colour = (~str_ids.str.isnumeric()).any()
            has_bw = str_ids.str.isnumeric().any()
            if has_colour and has_bw:
                self.data = self.data[~str_ids.str.isnumeric()].reset_index(drop=True)
                self.ids = [i for i in pd.unique(self.data.id)]

        # Drop columns that are no longer needed downstream
        self.data = self.data.drop(columns=["com", "consmerged"], errors="ignore")

        # Unpack legacy packed coordinate columns if present
        self._unpack_coords("com", ["cx", "cy"])

        # Rename modern head columns to the fx/fy convention used downstream
        if "hx" in self.data.columns and "hy" in self.data.columns:
            self.data.rename(columns={"hx": "fx", "hy": "fy"}, inplace=True)

        # Unpack legacy head/tail tuple columns
        if "head" in self.data.columns:
            self._unpack_coords("head", ["fx", "fy"])
        if "tail" in self.data.columns:
            self._unpack_coords("tail", ["tx", "ty"])

        # When using bw angle as orientation source, rename it now
        if self.orientfrombw and "angle" in self.data and \
                "hx" not in self.data and "fx" not in self.data:
            self.data.rename(columns={"angle": "orient"}, inplace=True)

        # Set frame as index for alignment
        self.data.index = list(self.data.frame)

        # Remove frames outside the tracked window
        valid_frames = list(range(self.minfr, self.maxfr + 1, self.frame_step))
        invalid = self.data.index[~self.data.index.isin(valid_frames)]
        if len(invalid) > 0:
            self._log(f"removed {len(invalid)} frames outside {self.minfr}–{self.maxfr}")
            self.data = self.data[self.data.index.isin(valid_frames)]

        # If nothing remains, create a minimal skeleton
        if len(self.data) == 0:
            self.ids = [0]
            self.data = pd.DataFrame({
                "frame": list(range(self.minfr, self.maxfr + 1, self.frame_step)),
                "id": 0,
            })

        self.data = ensure_columns(self.data,
                                   {"cx": np.nan, "cy": np.nan,
                                    "ID": np.nan, "traj": np.nan,
                                    "inroi": 1, "inmask": 1})

        # Expand each ID to the full frame range and remove short fragments
        final = None
        for idind, id in enumerate(self.ids):

            alones = getalones(self.data[self.data.id == id], "cx", self.min_traj_len)
            if len(alones) > 0:
                coord_cols = (["cx", "cy"] +
                              (["fx", "fy"] if "fx" in self.data else []) +
                              (["tx", "ty"] if "tx" in self.data else []))
                self.data.loc[alones, coord_cols] = np.nan
                self._log(f"removed {len(alones)} outlier frames for obj {id}")
            if self.data["cx"].isna().all():
                self.emptydat = True

            # Build a full-range frame skeleton and fill in tracked rows
            newcols = [[frame, id] + list(np.repeat(np.nan, len(self.data.columns) - 2))
                       for frame in range(self.minfr, self.maxfr + 1, self.frame_step)]
            newdat = pd.DataFrame(newcols, columns=self.data.columns, dtype=object)
            newdat.index = list(newdat.frame)
            self.data.index = self.data["frame"]
            newdat.loc[self.data[self.data.id == id].index] = self.data[self.data.id == id]
            final = newdat if final is None else pd.concat([final, newdat])

        self.data = final.sort_values(["id", "frame"])
        self.data.index = list(range(len(self.data)))
        self.data["frame"] = self.data["frame"].astype(int)

        self._log("prepared")


    # ------------------------------------------------------------------
    # Private processing helpers
    # ------------------------------------------------------------------

    def _fix_headtail(self, sub):
        """Clean head/tail coordinate data per trajectory.

        Blanks coordinates near the ROI border, removes isolated detections,
        and interpolates short gaps. Does not attempt head/tail swap correction
        (fixheadtail requires a working orient column — that is computed later).
        """
        newroi = ((1, 1), (self.fileinfo.roi[1][0] - self.fileinfo.roi[0][0],
                           self.fileinfo.roi[1][1] - self.fileinfo.roi[0][1]))
        corrected = []
        total_missing = 0

        for t in sub.traj.dropna().unique():
            tf = sub[sub.traj == t].copy().reset_index(drop=True)

            # Blank head/tail within roi_edge_margin of the ROI boundary
            borderdist = calc_borderdistdf(tf, newroi)
            # Negative inside: only blank positions near or outside the edge.
            tf.loc[borderdist > -self.roi_edge_margin, ["fx", "fy", "tx", "ty"]] = np.nan

            # Remove isolated head/tail detections shorter than min_traj_len
            alones = getalones(tf, "fx", self.min_traj_len)
            if alones:
                tf.loc[alones, ["fx", "fy", "tx", "ty"]] = np.nan

            # Interpolate short head/tail gaps
            if self.interpolate:
                for cols in [["fx", "fy"], ["tx", "ty"]]:
                    if cols[0] in tf.columns:
                        tf, miss = fillmissing(tf, cols, None, self.fileinfo.roi,
                                               win=5, lenthresh=self.interp_gap_orient)
                        total_missing += miss

            corrected.append(tf)

        if total_missing:
            self._log(f"step 4: filled {total_missing} head/tail frames")

        return pd.concat(corrected).reset_index(drop=True) if corrected else sub


    def _compute_orientation(self, final):
        """Compute, fill, smooth, and finalise orientation for all IDs and trajectories.

        Called after movement variables so that heading/speed are available as
        fallback sources for missing orientation.

        Two modes:
        - orientfrombw: use bw tracking angle directly as orient
        - fx/fy present: derive orient from head direction relative to centroid
        """
        if self.orientfrombw and "angle" in final.columns:
            # Simple mode: angle column → orient
            final["orient"] = final["angle"]
            valid = final["orient"].dropna().index
            result = final.loc[valid, "orient"].apply(angle_to_vec)
            final.loc[valid, ["vx", "vy"]] = result.apply(pd.Series, index=["vx", "vy"])
            for col in ["vx", "vy"]:
                final[col] = pd.to_numeric(final[col], errors="coerce")
            final["turnspeed"] = get_anglediff(final["orient"])
            final["turnaccel"] = get_anglediff(final["turnspeed"])
            final.drop(columns=["fx", "fy", "fx_c", "fy_c"], inplace=True, errors="ignore")
            return final

        if "fx" not in final.columns:
            return final

        final["orient"] = np.nan

        for ID in final.ID.unique():
            for t in final.loc[final.ID == ID].traj.dropna().unique():
                inds = final.loc[(final.ID == ID) & (final.traj == t)].index

                # Derive orient from head position relative to centroid
                pt1 = series_to_point_tuple(final.loc[inds], ["cx", "cy"])
                pt2 = series_to_point_tuple(final.loc[inds], ["fx", "fy"])
                final.loc[inds, "orient"] = points_to_angle(pt1, pt2, flip=True)

                # Fall back to heading when orient is missing and speed is sufficient
                final.loc[inds, "orient"] = final.loc[inds].apply(
                    lambda row: row["heading"]
                    if pd.isnull(row["orient"])
                    and row.get("speed", 0) >= self.orient_min_speed
                    else row["orient"], axis=1)

                final.loc[inds, "orient"] = pd.to_numeric(
                    final.loc[inds, "orient"], errors="coerce")

                # Compute orientation vectors and fill/smooth them
                valid = final.loc[inds, "orient"].dropna().index
                result = final.loc[valid, "orient"].apply(angle_to_vec)
                final.loc[valid, ["vx", "vy"]] = result.apply(
                    pd.Series, index=["vx", "vy"])
                for col in ["vx", "vy"]:
                    final[col] = pd.to_numeric(final[col], errors="coerce")

                if self.interpolate:
                    final.loc[inds], _ = fillmissing(
                        final.loc[inds], ["vx", "vy"], None, None,
                        lenthresh=self.interp_gap_orient)
                final.loc[inds] = smooth(
                    final.loc[inds], [t], ["vx", "vy"], self.smoothwin)

                # Recompute orient and turn rates from smoothed vectors
                final.loc[inds, "orient"] = points_to_angle(
                    (final.loc[inds, "vx"], final.loc[inds, "vy"]), flip=True)
                final.loc[inds, "turnspeed"] = get_anglediff(final.loc[inds, "orient"])
                final.loc[inds, "turnaccel"] = get_anglediff(final.loc[inds, "turnspeed"])

        return final


    def _calc_distances(self, final, xs_global, ys_global):
        """Compute all distance measures and attach them to final."""
        xmin, ymin = self.fileinfo.roi[0]
        xmax, ymax = self.fileinfo.roi[1]
        xs_full = xs_global + xmin
        ys_full = ys_global + ymin

        final["rdist"] = dist_to_rect(xs_full, ys_full, xmin, xmax, ymin, ymax) * self.conv * -1

        valid = np.isfinite(xs_full) & np.isfinite(ys_full)

        if self.maskcoords:
            final["mdist"] = dist_to_poly(xs_full, ys_full, self.maskcoords) * self.conv
            mx = np.full(len(xs_full), np.nan)
            my = np.full(len(xs_full), np.nan)
            if valid.any():
                tree = KDTree(self.maskcoords)
                _, idxs = tree.query(np.column_stack([xs_full[valid], ys_full[valid]]))
                pts = np.array(self.maskcoords)
                mx[valid] = pts[idxs, 0] - xmin
                my[valid] = pts[idxs, 1] - ymin
            final["mx"] = mx
            final["my"] = my

        if self.wallcoords:
            final["wdist"] = dist_to_poly(xs_full, ys_full, self.wallcoords) * self.conv
            wx = np.full(len(xs_full), np.nan)
            wy = np.full(len(xs_full), np.nan)
            if valid.any():
                tree = KDTree(self.wallcoords)
                _, idxs = tree.query(np.column_stack([xs_full[valid], ys_full[valid]]))
                pts = np.array(self.wallcoords)
                wx[valid] = pts[idxs, 0] - xmin
                wy[valid] = pts[idxs, 1] - ymin
            final["wx"] = wx
            final["wy"] = wy
        if self.maskzonecoords:
            final["mzdist"] = dist_to_poly(xs_full, ys_full, self.maskzonecoords) * self.conv
            final["inmaskzone"] = (final["mzdist"] <= 0).astype(int)

        for zidx, coords in self.zonecoords.items():
            d = dist_to_zone(xs_full, ys_full, coords, self.conv)
            if d is not None:
                final[f"z{zidx}dist"] = d

        ptcols = [c for c in self.overview.columns if re.match(r"pt\d+$", c)
                  and pd.notna(self.fileinfo.get(c))]
        if ptcols:
            computed = []
            for ptcol in ptcols:
                ptstr = self.fileinfo.get(ptcol)
                try:
                    pt = literal_eval(ptstr)
                    if isinstance(pt, tuple) and len(pt) == 2:
                        pt_cm = (convert(pt[0], pt[1], self.conv, self.height,
                                         self.datflip, roi=None, already_relative=False)
                                 if self.convert else pt)
                        final[f"{ptcol}dist"] = [ptsToDist((x, y), pt_cm)
                                                  for x, y in zip(final.cx_c, final.cy_c)]
                        computed.append(ptcol[2:])
                except Exception as e:
                    self._log(f"warning: could not parse {ptcol} → {ptstr}: {e}")
            if computed:
                self._log(f"step 9: point distances — {', '.join(computed)}")

        return final


    def _finalise(self, final):
        """Blank in-mask coordinates, order columns, round, and save."""

        # Blank positional columns for in-mask frames
        positional = ["cx", "cy", "fx", "fy", "fx_c", "fy_c", "tx", "ty", "tx_c", "ty_c",
                      "mx", "my", "wx", "wy"]
        final.loc[final["inmask"] == 1,
                  [c for c in positional if c in final.columns]] = np.nan

        # Drop converted coordinates when conversion was not applied
        if not (self.convert and pd.notna(self.fileinfo.conv)):
            final.drop(columns=["cx_c", "cy_c", "fx_c", "fy_c"],
                       inplace=True, errors="ignore")

        # Drop head coordinates when using bw angle as orientation source
        if self.orientfrombw:
            final.drop(columns=["fx", "fy", "fx_c", "fy_c"],
                       inplace=True, errors="ignore")

        final.drop(columns=["inroi"], inplace=True, errors="ignore")

        # Column ordering
        base_order = [
            "frame", "time", "ID", "traj", "inmask",
            "cx", "cy", "cx_c", "cy_c",
            "fx", "fy", "fx_c", "fy_c",
            "displ", "speed", "accel", "cumdispl",
            "heading", "orient", "turnspeed", "turnaccel", "cumturn", "abscumturn",
            "rdist", "mdist", "wdist",
            "mx", "my", "wx", "wy",
        ]
        ordered = [c for c in base_order if c in final.columns]
        zone_cols = sorted([c for c in final.columns if re.match(r"z\d+dist", c)],
                           key=lambda c: int(re.findall(r"z(\d+)dist", c)[0]))
        pt_cols = sorted([c for c in final.columns if re.match(r"pt\d+dist", c)],
                         key=lambda c: int(re.findall(r"pt(\d+)dist", c)[0]))
        # Own the selected columns before rounding them in place.
        final = final[[c for c in ordered + zone_cols + pt_cols if c in final.columns]].copy()

        # Rounding
        round_map = {
            3: ["turnspeed", "turnaccel"],
            2: ["time", "cx", "cy", "fx", "fy", "cx_c", "cy_c", "fx_c", "fy_c",
                "displ", "speed", "accel"],
            1: ["cumdispl", "cumturn", "abscumturn", "heading", "orient",
                "mdist", "wdist", "rdist"],
        }
        for decimals, cols in round_map.items():
            present = [c for c in cols if c in final.columns]
            if present:
                final[present] = final[present].round(decimals)
        for col in [c for c in ["mx", "my", "wx", "wy"] if c in final.columns]:
            final[col] = final[col].round(0)
        for col in zone_cols + pt_cols:
            if col in final.columns:
                final[col] = final[col].round(1)

        return final


    def _write_empty(self):
        """Write a minimal placeholder CSV when no tracking data is available."""
        out = self.data.copy()
        out["cx_c"] = out["cx"]
        out["cy_c"] = out["cy"]
        out["rdist"] = np.nan
        out["inmask"] = 1
        out.to_csv(self.procfile, index=False)
        self._log("no tracking data — empty file written")


    # ------------------------------------------------------------------
    # Main processing pipeline
    # ------------------------------------------------------------------

    def process(self):
        """
        Post-processing pipeline. Steps:
          1.  Add time column; optionally resample to a lower fps
          2.  Per ID: interpolate short centroid gaps
          3.  Per ID: remove mask-covered data; assign trajectory numbers
          4.  Per ID: clean head/tail coordinates (if orientation was tracked)
          5.  Per ID: smooth centroid trajectories
          6.  Convert pixel coordinates to real-world units
          7.  Compute per-trajectory movement variables
          8.  Compute and smooth orientation (requires speed/heading from step 7)
          9.  Compute distance measures (if compute_distances=True)
         10.  Finalise: blank in-mask rows, order columns, round, save
        """
        if self.emptydat:
            self._write_empty()
            return

        # Step 1: time column and optional fps resampling
        self.data["time"] = np.round(self.data.frame / self.fileinfo.fps, 3)
        if isinstance(self.changefps, int):
            if self.changefps > self.fileinfo.fps:
                self._log("provided fps exceeds file fps — skipping")
                return
            self.fps = self.changefps
            framelist = list(np.round(np.linspace(
                1, self.maxfr,
                int(self.maxfr / float(self.fileinfo.fps) * self.changefps))))
            self.data = self.data[self.data["frame"].isin(framelist)]
            self._log(f"step 1: resampled to {self.fps} fps")

        # Initialise output columns
        assign_cols = dict(ID=np.nan, traj=np.nan, inroi=1, inmask=1)
        if self.compute_movement:
            assign_cols.update(
                displ=np.nan, cumdispl=np.nan,
                speed=np.nan, accel=np.nan,
                heading=np.nan, orient=np.nan,
                turnspeed=np.nan, turnaccel=np.nan,
                cumturn=np.nan, abscumturn=np.nan,
            )
        self.data = self.data.assign(**assign_cols)

        # Steps 2–5: per-ID processing
        final_parts = []
        for idind, id in enumerate(self.ids):
            sub = self.data[self.data.id == id].copy().reset_index(drop=True)
            sub["ID"] = self.IDs[idind]

            # Step 2: interpolate centroid gaps
            if self.interpolate:
                sub, n_filled = fillmissing(
                    sub, ["cx", "cy"], None, self.fileinfo.roi,
                    win=5, nearmaskdis=self.mask_margin,
                    edgedis=self.roi_edge_margin, lenthresh=self.interp_gap_com)
                self._log(f"[obj {id}] step 2: {n_filled} centroid gaps filled")

            # Step 3: remove near-mask data; assign trajectory numbers
            sub, ntraj, nremoved = process_trajectories(
                sub, mask=self.maskcoords, cover=self.cover, win=5,
                trajgap=self.max_traj_gap, inmaskdis=self.mask_margin,
                mintrajlength=self.min_traj_len,
                erase_coords=True, interpolate=self.interpolate)
            sub["inmask"] = sub["traj"].isna().astype(int)
            self._log(f"[obj {id}] step 3: {ntraj} traj, {nremoved} near-mask removed")

            # Step 4: head/tail coordinate cleanup
            if {"fx", "fy"}.issubset(sub.columns):
                sub = self._fix_headtail(sub)

            # Step 5: smooth centroid trajectories
            if self.smoothwin > 1:
                sub = smooth(sub, sub.traj.dropna().unique(), ["cx", "cy"], self.smoothwin)
                self._log(f"[obj {id}] step 5: smoothed")

            final_parts.append(sub)

        final = pd.concat(final_parts).sort_values(["ID", "frame"]).reset_index(drop=True)

        # Step 6: coordinate conversion
        xs_global = final["cx"].values.copy()
        ys_global = final["cy"].values.copy()
        if self.convert and pd.notna(self.fileinfo.conv):
            final["cx_c"], final["cy_c"] = convert(
                final["cx"], final["cy"], self.conv, self.height,
                self.datflip, self.fileinfo.roi, already_relative=True)
            if {"fx", "fy"}.issubset(final.columns):
                final["fx_c"], final["fy_c"] = convert(
                    final["fx"], final["fy"], self.conv, self.height,
                    self.datflip, self.fileinfo.roi, already_relative=True)
            self._log("step 6: converted")
        else:
            final["cx_c"], final["cy_c"] = final["cx"].copy(), final["cy"].copy()
            if {"fx", "fy"}.issubset(final.columns):
                final["fx_c"], final["fy_c"] = final["fx"].copy(), final["fy"].copy()

        if self.compute_movement:
            # Step 7: movement variables (displacement, speed, heading, turn rates)
            for ID in final.ID.unique():
                id_mask = final.ID == ID
                for t in final.loc[id_mask, "traj"].dropna().unique():
                    traj_mask = id_mask & (final.traj == t)
                    fx = final.loc[traj_mask, "cx_c"]
                    fy = final.loc[traj_mask, "cy_c"]
                    final.loc[traj_mask, "displ"] = calcudiff(fx, fy)
                    # speed: displ (mm/frame) × fps → mm/s ÷ 10 → cm/s
                    final.loc[traj_mask, "speed"] = final.loc[traj_mask, "displ"] * self.fps / 10
                    final.loc[traj_mask, "accel"] = (
                        differentiate(final.loc[traj_mask, "speed"]) * self.fps)
                    final.loc[traj_mask, "heading"] = calcudiff(fx, fy, angle=True)
                    final.loc[traj_mask, "turnspeed"] = get_anglediff(
                        final.loc[traj_mask, "heading"])
                    final.loc[traj_mask, "turnaccel"] = get_anglediff(
                        final.loc[traj_mask, "turnspeed"])
                final.loc[id_mask, "cumdispl"] = np.nancumsum(final.loc[id_mask, "displ"])
            self._log("step 7: movement done")

            # Ensure coordinate columns are numeric before orientation step
            for col in ["fx", "fy", "cx", "cy", "vx", "vy"]:
                if col in final.columns:
                    final[col] = pd.to_numeric(final[col], errors="coerce")

            # Step 8: orientation (needs speed/heading from step 7)
            final = self._compute_orientation(final)
            for ID in final.ID.unique():
                final.loc[final.ID == ID, "cumturn"] = np.nancumsum(
                    final.loc[final.ID == ID, "turnspeed"])
                final.loc[final.ID == ID, "abscumturn"] = np.nancumsum(
                    abs(final.loc[final.ID == ID, "turnspeed"]))
            self._log("step 8: orientation done")

        # Ensure converted coordinates are numeric
        for col in ["cx_c", "cy_c", "fx_c", "fy_c"]:
            if col in final.columns:
                final[col] = pd.to_numeric(final[col], errors="coerce")

        if self.compute_distances:
            # Step 9: distance measures
            final = self._calc_distances(final, xs_global, ys_global)
            self._log("step 9: distances done")

        # Step 10: finalise and save
        final = self._finalise(final)
        final.to_csv(self.procfile, index=False)
        self._log("step 10: saved")
