#! /usr/bin/env python

import re
import os
import cv2
import pandas as pd
from scipy.spatial import cKDTree
from ast import literal_eval

import multiprocessing
from pythutils.mathutils import points_to_angle, angle_to_vec, ptsToDist

from .utils import *

class Processor:

    def __init__(self, dirs, config, overview, trackedfiles, orientfrombw, overwrite, addIDs, removeoutliers,
                 alonewindow=5, changefps=None, datflip=True, fulldata=True,
                 convert=True, nearmaskdis=20, inmaskdis=10, trajgap = 50, edgedis=25, smoothwin=10, centertype=None,
                 centralise=False, filllenthresh_com=500, filllenthresh_headtail=50, mintrajlength=10, fillmissingorientdiffthresh=100, headtoorientspeedthresh=1,
                 delcontdata=False, powermate=False, force_single_traj=False):

        """
        Performs a number of processing steps on the data.
        """

        self.dirs = dirs
        self.config = config
        self.overview = overview
        self.overview.roi = [literal_eval(i) if isinstance(i,str) else i for i in self.overview.roi]
        self.trackedfiles = trackedfiles
        self.appendtxt = "_F.csv"
        self.orientfrombw = orientfrombw
        self.overwrite = overwrite
        self.addIDs = addIDs
        self.removeoutliers = removeoutliers
        self.alonewindow = alonewindow
        # self.manfix = manfix
        # self.manonly = manonly
        self.fulldata = fulldata
        self.changefps = changefps
        self.datflip = datflip
        self.convert = convert
        self.nearmaskdis = nearmaskdis
        self.inmaskdis = inmaskdis
        self.trajgap = trajgap
        self.edgedis = edgedis
        self.smoothwin = smoothwin
        self.centertype = centertype
        self.centralise = centralise
        self.filllenthresh_com = filllenthresh_com
        self.filllenthresh_headtail = filllenthresh_headtail
        # self.man_vidresizeval = man_vidresizeval
        # self.man_types = man_types
        # self.man_customstep = man_customstep
        self.mintrajlength = mintrajlength
        self.fillmissingorientdiffthresh = fillmissingorientdiffthresh
        self.headtoorientspeedthresh = headtoorientspeedthresh
        # self.man_frameloc = man_frameloc 
        # self.man_timewindow = man_timewindow
        # self.man_threshold_speed = man_threshold_speed
        self.delcontdata = delcontdata
        self.powermate = powermate
        self.force_single_traj = force_single_traj


    def _unpack_coords(self, column, newcols):
        """Unpack tuple column (like 'com') into two numeric float columns."""
        if column in self.data:
            # Ensure all empty or null entries are tuples
            mask = pd.isnull(self.data[column])
            self.data.loc[mask, column] = [(np.nan, np.nan)] * mask.sum()

            if self.data[column].notna().sum() == 0:
                # If no data to unpack, just fill with NaN columns
                self.data[newcols[0]] = np.nan
                self.data[newcols[1]] = np.nan
            else:
                # Unpack and convert
                self.data[newcols[0]], self.data[newcols[1]] = zip(*self.data[column])
                self.data[newcols[0]] = pd.to_numeric(self.data[newcols[0]], errors="coerce").astype(float)
                self.data[newcols[1]] = pd.to_numeric(self.data[newcols[1]], errors="coerce").astype(float)

            # Drop original
            self.data = self.data.drop(columns=[column])

    
    def setup(self, trackedfile, thread_id):

        self.trackedfile = trackedfile
        self.tread_id = thread_id
        self.nr = str(self.trackedfiles.index(self.trackedfile))
        self.filebase = os.path.splitext(os.path.basename(self.trackedfile))[0]
        if self.appendtxt in self.trackedfile:
            self.procfile = self.trackedfile
        else:
            if "_E" in self.filebase:
                self.procfile = os.path.join(self.dirs["processed"], self.filebase.replace("_E", "") + self.appendtxt)
            else:
                self.procfile = os.path.join(self.dirs["processed"], self.filebase + self.appendtxt)

        pr_procid = multiprocessing.current_process()._identity
        pr_procid = "{P"+str(pr_procid[0])+"}" if len(pr_procid)>0 else ""
        self.pr_comm = pr_procid+"["+self.nr+"] "+self.filebase+" "

        # Preparation of processing file (P1-P3)
        if os.path.isfile(self.procfile) and not self.overwrite:
            return

        print(self.pr_comm+"Processing started..", end=" ", flush=True)

        # Get the corresponding entry of the overview file
        if "_E" not in self.filebase[-5:] and "_Z" not in self.filebase[-5:] and "_R" not in self.filebase[-5:]: 
            self.ind = list(self.overview[(self.overview.video==self.filebase)].index.values)[0]
        if "_E" in self.filebase[-5:]: 
            self.ind = list(self.overview[(self.overview.video==self.filebase[:-2])].index.values)[0]
        if "_Z" in self.filebase[-5:]:
            self.ind = list(self.overview[(self.overview.video==self.filebase[:-3])&(self.overview.zone==int(self.filebase[-1:]))].index.values)[0]
        if "_R" in self.filebase[-5:]:
            self.ind = list(self.overview[(self.overview.video==self.filebase[:-3])&(self.overview.region==int(self.filebase[-1:]))].index.values)[0]

        # Load the datafile
        self.data = pd.read_csv(self.trackedfile, header=0)
        if len(self.data) > 0 and "_E" not in self.filebase[-5:]:
            converters = {c: lit_converter for c in self.data.columns if str(self.data[c].iloc[0]).startswith("(")}
            self.data = pd.read_csv(self.trackedfile, header=0, converters=converters)
            self.emptydat = False
        else:
            self.emptydat = True
        print("["+self.nr+"] "+"Data loaded, "+str(len(self.data))+" rows..", end=" ", flush=True)

        # Remove bw contours if needed
        if self.delcontdata:
            idnr = len(self.data['id'].unique())
            self.data = self.data[~self.data['id'].str.isnumeric()]
            self.data.reset_index(drop=True, inplace=True)
            if idnr > len(self.data['id'].unique()):
                print("["+self.nr+"] "+"Removed bw contour ids..", flush=True)

        # Get combinations of threshtype and identities to get unique objects
        self.ids = [i for i in pd.unique(self.data.id)]

        # Get further file information
        self.fileinfo = self.overview.loc[self.ind]
        if self.fulldata:
            self.minfr = int(self.fileinfo.frame_start if self.fileinfo.frame_start==self.fileinfo.frame_start else self.config.track.startframe)
            self.maxfr = int(self.fileinfo.fcount if self.fileinfo.frame_stop != self.fileinfo.frame_stop else self.fileinfo.frame_stop)
        else:
            self.minfr, self.maxfr = (min(self.data["frame"]),max(self.data["frame"]))
        self.fps = self.fileinfo.fps
        if self.convert and self.fileinfo.conv == self.fileinfo.conv:
            self.conv = float(self.fileinfo.conv)
        else:
            self.conv = 1
        if "ID" not in self.fileinfo or self.fileinfo.ID!=self.fileinfo.ID:
            self.IDs = self.data.id.unique()
        else:
            IDval = self.fileinfo.ID
            if isinstance(IDval, str):
                if IDval.startswith("(") or IDval.startswith("["):
                    try:
                        self.IDs = list(literal_eval(IDval))
                    except Exception:
                        self.IDs = IDval.strip('()[]').split(',')
                elif "," in IDval:
                    self.IDs = [s.strip() for s in IDval.split(",")]
                else:
                    self.IDs = [IDval.strip()]
            else:
                self.IDs = list(IDval) if hasattr(IDval, "__iter__") else [str(IDval)]
        self.res = literal_eval(self.fileinfo.resolution)
        self.cover = self.fileinfo["maskimg"]==self.fileinfo["maskimg"]
        self.height = self.fileinfo.roi[1][1]-self.fileinfo.roi[0][1]

        self.prep()

        # # Run manual tracking function
        # if self.manfix:
        #     vidfile = self.trackedfile[:-6]+"_TR.mp4" if self.trackedfile.endswith(("_E.csv","_R.csv","_Z.csv")) else self.trackedfile[:-4]+"_TR.mp4"
        #     TM = Tracker_man(vidfile = vidfile,
        #                      data = self.data,
        #                      fileaction = "append",
        #                      ids = self.ids,
        #                      ptypes = self.man_types,
        #                      resizeval = self.man_vidresizeval,
        #                      internal = True,
        #                      customstep = self.man_customstep,
        #                      conv = self.conv,
        #                      smoothwin = self.smoothwin,
        #                      frameloc = self.man_frameloc, 
        #                      timewindow = self.man_timewindow, 
        #                      threshold_speed = self.man_threshold_speed,
        #                      powermate = self.powermate)
        #     TM.track()

        #     if self.manonly:
        #         return

        # Load all relevant images
        self.bg_path   = valid_img_path(self.fileinfo, "bgimg",   self.dirs["originals"])
        self.mask_path = valid_img_path(self.fileinfo, "maskimg", self.dirs["originals"])
        self.wall_path = valid_img_path(self.fileinfo, "wallimg", self.dirs["originals"])
        self.zone_path = valid_img_path(self.fileinfo, "zoneimg", self.dirs["originals"])
        if self.bg_path:
            self.img_bg = cv2.imread(self.bg_path)
        else:
            print(f"[{self.nr}] bg img does not exist..", end=" ", flush=True)
        if self.mask_path:
            _, self.maskcoords = coordsfrommask(self.mask_path)
        else:
            print(f"[{self.nr}] Maskimg does not exist..", end=" ", flush=True)
            self.maskcoords = []
        if self.wall_path:
            _, self.wallcoords = coordsfrommask(self.wall_path)
        else:
            print(f"[{self.nr}] Wallimg does not exist..", end=" ", flush=True)
            self.wallcoords = []
        if self.zone_path:
            self.zonecoords = coordsfromzones(self.zone_path)
        else:
            print(f"[{self.nr}] Zoneimg does not exist..", end=" ", flush=True)
            self.zonecoords = {}
        self.center = None
        if self.centertype == "pt" and "pt" in self.fileinfo and pd.notna(self.fileinfo["pt"]):
            x, y = literal_eval(self.fileinfo["pt"])
            self.center = (x, y)
        elif self.centertype == "walls" and self.wallcoords:
            wall_xs, wall_ys = zip(*self.wallcoords)
            self.center = (np.mean(wall_xs), np.mean(wall_ys))
        elif self.centertype == "roi":
            xmin, ymin = self.fileinfo.roi[0]
            xmax, ymax = self.fileinfo.roi[1]
            width = xmax - xmin
            height = ymax - ymin
            self.center = (width / 2, height / 2)
        else:
            self.center = None
        self.process()


    def prep(self):
        # -- Coordinate cleanup and unpacking --
        
        # Remove unnecessary columns
        self.data = self.data.drop(columns=["com", "consmerged"], errors="ignore")

        # If we have old-style packed coords, unpack them
        self._unpack_coords("com", ["cx", "cy"])

        # Handle modern hx/hy
        if "hx" in self.data.columns and "hy" in self.data.columns:
            # use hx,hy as head coordinates for downstream code (fx,fy)
            self.data.rename(columns={"hx": "fx", "hy": "fy"}, inplace=True)

        # Only unpack "head"/"tail" if those columns actually exist (legacy case)
        if "head" in self.data.columns:
            self._unpack_coords("head", ["fx", "fy"])
        if "tail" in self.data.columns:
            self._unpack_coords("tail", ["tx", "ty"])

        # Only rename angle to orient if we really want to use angle as primary source
        if self.orientfrombw and "angle" in self.data and "hx" not in self.data and "fx" not in self.data:
            self.data.rename(columns={'angle': 'orient'}, inplace=True)

        # Set frame index
        self.data.index = list(self.data.frame)

        # Remove frames outside the allowed start/stop window
        valid_frames = list(range(self.minfr, self.maxfr + 1))
        invalid_frames = self.data.index[~self.data.index.isin(valid_frames)]
        if len(invalid_frames) > 0:
            print(f"[{self.nr}] Removed {len(invalid_frames)} invalid frames "
                f"(outside {self.minfr}-{self.maxfr}).", flush=True)
            self.data = self.data[self.data.index.isin(valid_frames)]

        # If no data, ensure at least ID=0
        if len(self.data) == 0:
            self.ids = [0]
            self.data = pd.DataFrame({
                "frame": list(range(self.minfr, self.maxfr + 1)),
                "id": 0
            })
        
        # Add core columns
        self.data = ensure_columns(self.data, {"cx": np.nan, "cy": np.nan, "ID": np.nan, 
                                               "traj": np.nan, "inroi": 1, "inmask": 1})

        # Process each ID
        for idind, id in enumerate(self.ids):
            
            # Remove short fragments of tracking data
            if self.removeoutliers:
                alones = getalones(self.data[self.data.id == id], "cx", self.alonewindow)
                if len(alones) > 0:
                    cols = ["cx", "cy"] + (["fx", "fy"] if "fx" in self.data else []) + (["tx", "ty"] if "tx" in self.data else [])
                    self.data.loc[alones, cols] = np.nan
                    print("[" + self.nr + "] Removed", len(alones), "outliers", flush=True)
                
                # Re-check for emptiness after removing outliers
                if self.data["cx"].isna().all() or len(self.data.dropna(subset=["cx", "cy"])) == 0:
                    self.emptydat = True

            # Add empty rows for each frame and ID combination
            newcols = [
                [frame, id] + list(np.repeat(np.nan, len(self.data.columns) - 2))
                for frame in range(self.minfr, self.maxfr + 1)
            ]
            newdat = pd.DataFrame(newcols, columns=self.data.columns, dtype=object)
            newdat.index = list(newdat.frame)
            # Reindex self.data to match newdat's frame-based index
            self.data.index = self.data["frame"]          

            # Assign matching rows by frame
            newdat.loc[self.data[self.data.id == id].index] = self.data[self.data.id == id]
            final = newdat if idind == 0 else pd.concat([final, newdat])

        # Finalize the dataset
        self.data = final.sort_values(["id", "frame"])
        self.data.index = list(range(len(self.data)))
        self.data["frame"] = self.data["frame"].astype(int)

        print("[" + self.nr + "] Data prepared..", end=" ", flush=True)


    def fixids(self):

        """
        This seems to be a semi-standalone function to check ids and link them
        before conflicts. But not sure how to call the function as it is nowhere
        used at the moment
        """

        # Load files
        dat = pd.read_csv(datafile)
        cap = cv2.VideoCapture(trackedvid)
        _,img = cap.read()
        h,w = img.shape[:2]

        # Get indices where contours were merged
        mergedinds = dat.index[dat["consmerged"]==1].values
        indsecs = getindsections(mergedinds)

        # Create windows
        cv2.namedWindow("Compilation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Compilation", w*2, h)
        cv2.moveWindow("Compilation", 50, 0)

        # For each merging event show prev and next frames
        for i,sec in enumerate(indsecs):
            print(str(i+1)+"|"+str(len(indsecs)), end=" ")
            bef = dat.loc[sec[0],"frame"]-5
            mid = dat.loc[int(sec[1]-((sec[1]-sec[0])/2)),"frame"]
            aft = dat.loc[sec[1],"frame"]+5
            cap.set(cv2.CAP_PROP_POS_FRAMES, bef)
            _,imga = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
            _,imgb = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, aft)
            _,imgc = cap.read()
            compilation = cv2.hconcat([imga, imgb, imgc])
            cv2.imshow("Compilation", compilation)
            key = cv2.waitKey(0) & 0xff
            if key == 27:
                break
            if key == ord('b'):
                dat.loc[sec[0]:sec[1]+1,"com"] = np.nan
                print("dats removed..", end=" ")

        if key != 27:
            dat.to_csv(datafile, index=False)


    def process(self):

        # Add real time column
        self.data["time"] = np.round(list(self.data.frame/self.fileinfo.fps),3)

        # Potentially subset data to a lower fps
        if type(self.changefps) == int:
            if(self.changefps > self.fileinfo.fps):
                print("["+self.nr+"] "+"Provided fps is higher than of file, skipping..")
                return
            self.fps = self.changefps
            framelist = list(np.round(np.linspace(1, self.maxfr,
                             int(self.maxfr/float(fileinfo.fps)*self.changefps))))
            self.data = self.data[self.data["frame"].isin(framelist)]
            print("["+self.nr+"] "+"Data subsetted to "+self.fps+" fps..", end=" ")

        # Assign emtpy data columns
        self.data = self.data.assign(ID = np.nan,
                                     traj = np.nan,
                                     displ = np.nan,
                                     cumdispl = np.nan,
                                     speed = np.nan,
                                     accel = np.nan,
                                     heading = np.nan,
                                     orient = np.nan,
                                     turnspeed = np.nan,
                                     turnaccel = np.nan,
                                     cumturn = np.nan,
                                     abscumturn = np.nan,
                                     inroi = 1,
                                     inmask=1)

        # Check each tracked object
        if not self.emptydat:
            final = []

            for idind, id in enumerate(self.ids):
                print(f"[{self.nr}] [obj {id}]:", end=" ", flush=True)

                # Subset and prepare data
                sub = self.data[self.data.id == id].copy().reset_index(drop=True)
                sub["ID"] = self.IDs[idind] if self.addIDs else id

                # Interpolate all gaps, even if they pass through the mask
                sub, nmissing = fillmissing(sub, ["cx", "cy"], None, self.fileinfo.roi,
                                            win=5, nearmaskdis=self.nearmaskdis,
                                            edgedis=self.edgedis,
                                            lenthresh=self.filllenthresh_com)
                print(f"[{self.nr}] Filled {nmissing} centroid frames..", end=" ")
                
                # Now that all points exist, remove ones under or near the mask
                sub, ntraj, nremoved = process_trajectories(
                    sub,
                    mask=self.maskcoords,
                    cover=self.cover,
                    win=5,
                    trajgap=self.trajgap,
                    inmaskdis=self.inmaskdis,
                    mintrajlength=self.mintrajlength,
                    erase_coords=True,
                    interpolate=True,
                    force_single_traj=self.force_single_traj
                )
                sub["inmask"] = sub["traj"].isna().astype(int)
                print(f"[{self.nr}] Removed {nremoved} points near cover and split into {ntraj} trajector{'ies' if ntraj != 1 else 'y'}..", end=" ")

                # Fix head/tail orientation if applicable
                if {"head", "tail"}.issubset(self.data.columns):
                    missing = 0
                    total_swaps = 0
                    medarea = np.nanmedian(sub["area"]) if "area" in sub else None
                    corrected = []

                    for t in sub.traj.dropna().unique():
                        traj_df = sub[sub.traj == t].copy().reset_index(drop=True)

                        # Exclude border regions
                        traj_df["orienttoexcl"] = np.nan
                        newroi = ((1, 1), (self.fileinfo.roi[1][0] - self.fileinfo.roi[0][0], self.fileinfo.roi[1][1] - self.fileinfo.roi[0][1]))
                        borderdist = calc_borderdistdf(traj_df, newroi)
                        traj_df.loc[borderdist < self.edgedis, "orienttoexcl"] = 1

                        # Fix head/tail swap if area allows
                        if medarea:
                            traj_df, swapped = fixheadtail(traj_df, thresharea=medarea)
                            traj_df.loc[traj_df.orienttoexcl == 1, ["head", "fx", "fy", "tail", "tx", "ty"]] = np.nan
                            total_swaps += len(swapped)

                        # Remove outlier head/tail frames
                        alones = getalones(traj_df, "fx", self.alonewindow)
                        if alones:
                            traj_df.loc[alones, ["fx", "fy", "tx", "ty"]] = np.nan
                            print(f"[{self.nr}] Removed {len(alones)} alone head-tail frames..", end=" ", flush=True)

                        # Fill missing head/tail data
                        for cols in [["fx", "fy"], ["tx", "ty"]]:
                            traj_df, miss = fillmissing(traj_df, cols, None, self.fileinfo.roi, win=5, lenthresh=self.filllenthresh_headtail)
                            missing += miss

                        corrected.append(traj_df)

                    sub = pd.concat(corrected).reset_index(drop=True)
                    print(f"[{self.nr}] Swapped {total_swaps} head and tails..", end=" ")
                    print(f"[{self.nr}] Filled in front data for {missing} NaN frames..", end=" ")

                # Smooth trajectory if enabled
                if self.smoothwin > 1:
                    sub = smooth(sub, sub.traj.dropna().unique(), ["cx", "cy"], self.smoothwin)
                    print(f"[{self.nr}] Data smoothed..", end=" ")

                # Merge with other objects
                final.append(sub)

            final = pd.concat(final).sort_values(["ID", "frame"]).reset_index(drop=True)

        else:
            print(f"[{self.nr}] No valid tracking data (emptydat=True) — skipping processing.")
            final = self.data.copy()
            final["cx_c"] = final["cx"]
            final["cy_c"] = final["cy"]
            final["rdist"] = np.nan
            final["inmask"] = 1
            final.to_csv(self.procfile, index=False)
            print(f"[{self.nr}] Empty file written due to missing coordinates.")
            return

        # --- Coordinate conversion ---
        xs_global, ys_global = final["cx"].values, final["cy"].values
        if self.convert and self.fileinfo.conv == self.fileinfo.conv:
            final["cx_c"], final["cy_c"] = convert(
                final["cx"], final["cy"], self.conv, self.height, self.datflip, self.fileinfo.roi, already_relative=True
            )
            if {"fx", "fy"}.issubset(final.columns):
                final["fx_c"], final["fy_c"] = convert(
                    final["fx"], final["fy"], self.conv, self.height, self.datflip, self.fileinfo.roi, already_relative=True
                )
            print(f"[{self.nr}] Data converted..", end=" ", flush=True)
        else:
            final["cx_c"], final["cy_c"] = final["cx"], final["cy"]
            if {"fx", "fy"}.issubset(final.columns):
                final["fx_c"], final["fy_c"] = final["fx"], final["fy"]

        # --- Movement variable calculations ---
        for ID in final.ID.unique():
            id_mask = final.ID == ID
            for t in final.loc[id_mask, "traj"].dropna().unique():
                traj_mask = id_mask & (final.traj == t)
                fx = final.loc[traj_mask, "cx_c"]
                fy = final.loc[traj_mask, "cy_c"]

                final.loc[traj_mask, "displ"] = calcudiff(fx, fy)
                final.loc[traj_mask, "speed"] = final.loc[traj_mask, "displ"] * self.fps / 10  # cm/s
                final.loc[traj_mask, "accel"] = differentiate(final.loc[traj_mask, "speed"]) * self.fps
                final.loc[traj_mask, "heading"] = calcudiff(fx, fy, angle=True)
                final.loc[traj_mask, "turnspeed"] = get_anglediff(final.loc[traj_mask, "heading"])
                final.loc[traj_mask, "turnaccel"] = get_anglediff(final.loc[traj_mask, "turnspeed"])

            # Cumulative displacement across all trajectories per ID
            final.loc[id_mask, "cumdispl"] = np.nancumsum(final.loc[id_mask, "displ"])

        # --- Final work for orientation data ---
        for col in ["fx", "fy", "tx", "ty", "cx", "cy", "vx", "vy"]:
            if col in final.columns:
                final[col] = pd.to_numeric(final[col], errors="coerce")

        if self.orientfrombw and "angle" in final.columns:
            # Simple mode: use angle from tracking as orient
            print(f"[{self.nr}] Using angle column as orient..", end=" ", flush=True)
            final["orient"] = final["angle"]

            # derive vx,vy from orient, smooth if you want, etc.
            valid_inds = final["orient"].dropna().index
            result = final.loc[valid_inds, "orient"].apply(angle_to_vec)
            final.loc[valid_inds, ["vx","vy"]] = result.apply(pd.Series, index=["vx", "vy"])
            final.loc[valid_inds, "vx"] = pd.to_numeric(final.loc[valid_inds, "vx"], errors="coerce")
            final.loc[valid_inds, "vy"] = pd.to_numeric(final.loc[valid_inds, "vy"], errors="coerce")

            # (optional) smooth vx,vy here
            # final = smooth(...)

            # recompute turnspeed/turnaccel if needed
            final["turnspeed"] = get_anglediff(final["orient"])
            final["turnaccel"] = get_anglediff(final["turnspeed"])

            # if in this mode you don't care about head coords, you can safely drop them
            final.drop(columns=["fx", "fy", "fx_c", "fy_c"], inplace=True, errors='ignore')
    
        elif "fx" in final.columns:
            print("["+self.nr+"] "+"Improving orient..", end=" ", flush=True)
            final["orient"] = None

            for ID in final.ID.unique():
                for t in final.loc[final.ID==ID].traj.dropna().unique():
                    # Get the subset indices for this ID and trajectory
                    inds = final.loc[(final.ID == ID) & (final.traj == t)].index

                    # Compute orient from head coordinates (fx, fy)
                    pt1 = series_to_point_tuple(final.loc[inds], ["cx", "cy"])
                    pt2 = series_to_point_tuple(final.loc[inds], ["fx", "fy"])
                    final.loc[inds, "orient"] = points_to_angle(pt1, pt2, flip=True)

                    # Add heading values to empty orient cells if speed >= threshold
                    final.loc[inds, "orient"] = final.loc[inds].apply(
                        lambda row: row['heading'] if pd.isnull(row['orient']) and row['speed'] >= self.headtoorientspeedthresh else row['orient'],
                        axis=1)

                    # Ensure 'orient' is numeric and handle NaN values
                    # Convert 'orient' to numeric for the entire selection (including 'inds' rows)
                    final.loc[inds, 'orient'] = pd.to_numeric(final.loc[inds, 'orient'], errors='coerce')

                    # Filter valid rows where 'orient' is a valid number (not NaN)
                    valid_inds = final.loc[inds, 'orient'].dropna().index

                    # Apply your 'angle_to_vec' formula for valid rows only
                    result = final.loc[valid_inds, "orient"].apply(angle_to_vec)
                    final.loc[valid_inds, ["vx","vy"]] = result.apply(pd.Series, index=["vx", "vy"])
                    final["vx"] = pd.to_numeric(final["vx"], errors="coerce")
                    final["vy"] = pd.to_numeric(final["vy"], errors="coerce")

                    # Fill in missing values as needed
                    final.loc[inds], missing = fillmissing(final.loc[inds], ["vx", "vy"], None, None, lenthresh=self.fillmissingorientdiffthresh)

                    # Smooth the data (if the smoothing window is large enough)
                    final.loc[inds] = smooth(final.loc[inds], [t], ["vx", "vy"], self.smoothwin)

                    # Recompute final orientation from smoothed vectors (vx, vy)
                    final.loc[inds,"orient"] = points_to_angle((final.loc[inds,"vx"], final.loc[inds,"vy"]), flip=True)

                    # Overwrite turnspeed
                    final.loc[inds,"turnspeed"] = get_anglediff(final.loc[inds,"orient"]) 

                    # Compute turning acceleration 
                    final.loc[inds,"turnaccel"] = get_anglediff(final.loc[inds,"turnspeed"]) 

        # Calculate additional turning speed variables
        for ID in final.ID.unique():
            final.loc[final.ID==ID,"cumturn"] = np.nancumsum(final.loc[final.ID==ID,"turnspeed"])
            final.loc[final.ID==ID,"abscumturn"] = np.nancumsum(abs(final.loc[final.ID==ID,"turnspeed"]))

        # --- Prepare coordinates ---
        final["cx_c"] = pd.to_numeric(final["cx_c"], errors='coerce')
        final["cy_c"] = pd.to_numeric(final["cy_c"], errors='coerce')
        if "fx_c" in final:
            final["fx_c"] = pd.to_numeric(final["fx_c"], errors='coerce')
            final["fy_c"] = pd.to_numeric(final["fy_c"], errors='coerce')

        # --- DISTANCE MEASURES ---
        xmin, ymin = self.fileinfo.roi[0]
        xmax, ymax = self.fileinfo.roi[1]
        xs_full = xs_global + xmin
        ys_full = ys_global + ymin

        # Distances to full-image features
        final["rdist"] = dist_to_rect(xs_full, ys_full, xmin, xmax, ymin, ymax) * self.conv * -1
        if self.maskcoords:
            final["mdist"] = dist_to_poly(xs_full, ys_full, self.maskcoords) * self.conv
        if self.wallcoords:
            final["wdist"] = dist_to_poly(xs_full, ys_full, self.wallcoords) * self.conv
        if self.center is not None:
            final["cdist"] = dist_to_point(xs_full, ys_full, *self.center) * self.conv
        if self.zonecoords:
            for zidx, coords in self.zonecoords.items():
                if not coords:
                    continue
                if len(coords) == 1:
                    dists = dist_to_point(xs_full, ys_full, coords[0][0], coords[0][1]) * self.conv
                elif len(coords) == 4 and is_axis_aligned_rectangle(coords):
                    coords_arr = np.asarray(coords)
                    xmin_z, xmax_z = coords_arr[:, 0].min(), coords_arr[:, 0].max()
                    ymin_z, ymax_z = coords_arr[:, 1].min(), coords_arr[:, 1].max()
                    dists = dist_to_rect(xs_full, ys_full, xmin_z, xmax_z, ymin_z, ymax_z) * self.conv
                elif len(coords) >= 3:
                    dists = dist_to_poly(xs_full, ys_full, coords) * self.conv
                else:
                    dists = np.nan
                final[f"z{zidx}dist"] = dists
        ptcols = [col for col in self.overview.columns if re.match(r"pt\d+$", col)]
        ptcols = [col for col in ptcols if pd.notna(self.fileinfo.get(col))]
        if ptcols:
            print(f"[{self.nr}] Distance computed to points:", end=" ", flush=True)
            for ptcol in ptcols:
                ptstr = self.fileinfo.get(ptcol)
                try:
                    pt = literal_eval(ptstr)
                    if isinstance(pt, tuple) and len(pt) == 2:
                        pt_cm = convert(pt[0], pt[1], self.conv, self.height, self.datflip, roi=None, already_relative=False) if self.convert else pt
                        distname = f"{ptcol}dist"
                        final[distname] = [ptsToDist((x, y), pt_cm) for x, y in zip(final.cx_c, final.cy_c)]
                        print(f"{ptcol[2:]}", end=" ", flush=True)
                except Exception as e:
                    print(f"\n[{self.nr}] Warning: Could not parse {ptcol} → {ptstr}: {e}", flush=True)

        # --- Centralise data ---
        if self.centralise and self.center is not None:
            cx0, cy0 = convert(self.center[0], self.center[1], self.conv, self.height, self.datflip, self.fileinfo.roi)
            final["cx_c"] -= float(cx0)
            final["cy_c"] -= float(cy0)
            if "fx_c" in final:
                final["fx_c"] -= float(cx0)
                final["fy_c"] -= float(cy0)
            print(f"[{self.nr}] Data centralised..", end=" ", flush=True)

        # --- Final data cleanup ---
        # Define columns to blank when inmask == 1 (only those that exist)
        positional_cols = ["cx", "cy", "fx", "fy", "fx_c", "fy_c", "tx", "ty", "tx_c", "ty_c"]
        cols_to_nan = [col for col in positional_cols if col in final.columns]
        final.loc[final["inmask"] == 1, cols_to_nan] = np.nan

        # Drop converted coordinates if not needed
        if not (self.convert and self.fileinfo.conv == self.fileinfo.conv):
            final.drop(columns=["cx_c", "cy_c", "fx_c", "fy_c"], inplace=True, errors='ignore')

        # Drop inroi as rdist and mdist can be used
        final.drop(columns=["inroi"], inplace=True, errors='ignore')

        # Drop head coords if orientation was estimated from binary image
        if self.orientfrombw:
            final.drop(columns=["fx", "fy", "fx_c", "fy_c"], inplace=True, errors='ignore')

        # Define static part of column order
        column_order = [
            "frame", "time", "ID", "traj", "inmask",
            "cx", "cy", "cx_c", "cy_c",
            "fx", "fy", "fx_c", "fy_c",
            "displ", "speed", "accel", "cumdispl",
            "heading", "orient", "turnspeed", "turnaccel", "cumturn", "abscumturn",
            "rdist", "mdist", "wdist", "cdist"
        ]

        # --- Final column ordering ---
        # Start with static base columns
        full_order = [col for col in column_order if col in final.columns]
        # Find all zone*_dist columns and sort numerically
        zone_dis_cols = sorted(
            [col for col in final.columns if re.match(r"z\d+dist", col)],
            key=lambda c: int(re.findall(r"z(\d+)dist", c)[0])
        )
        # Find all pt*_dist columns and sort numerically
        pt_dis_cols = sorted(
            [col for col in final.columns if re.match(r"pt\d+dist", col)],
            key=lambda c: int(re.findall(r"pt(\d+)dist", c)[0])
        )
        # Append zone and point distance columns in order
        full_order += zone_dis_cols + pt_dis_cols

        # Apply the final column order (only keep those that exist)
        final = final[[col for col in full_order if col in final.columns]]

        # Rounding: define by decimal level
        round_by = {
            3: ["turnspeed", "turnaccel"],
            2: ["time", "cx", "cy", "fx", "fy", "mx", "my", "wx", "wy", "cx_c", "cy_c", "fx_c", "fy_c", "displ", "speed", "accel", "turnspeed", "turnaccel"],
            1: ["cumdispl", "cumturn", "abscumturn", "heading", "orient", "dist", "mdist", "wdist", "cdist", "rdist"]
        }
        for decimals, cols in round_by.items():
            cols_in_df = [c for c in cols if c in final.columns]
            final[cols_in_df] = final[cols_in_df].round(decimals)
        zone_dis_cols = [col for col in final.columns if re.match(r"z\d+dist", col)]
        if zone_dis_cols:
            final[zone_dis_cols] = final[zone_dis_cols].round(1)
        if pt_dis_cols:
            final[pt_dis_cols] = final[pt_dis_cols].round(1)

        # Save
        final.to_csv(self.procfile, index=False)
        print(f"[{self.nr}] File written..", end=" ", flush=True)