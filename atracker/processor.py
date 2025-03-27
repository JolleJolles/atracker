#! /usr/bin/env python

import os
import cv2
import sys
import pandas as pd
from ast import literal_eval

import multiprocessing
from pythutils.sysutils import lineprint
from pythutils.mathutils import points_to_angle, angle_to_vec, ptsToDist

from .utils import *
from .tracker_man import Tracker_man

class Processor:

    def __init__(self, dirs, config, overview, trackedfiles, orientfrombw,overwrite, addIDs, removeoutliers,
                 alonewindow=5, manfix=False, changefps=None, datflip=True, fulldata=True,
                 convert=True, nearmaskdis=20, inmaskdis=10, trajgap = 50, edgedis=25, smoothwin=10, centertype=None,
                 centralise=False, filllentresh_com=500, filllentresh_headtail=50, man_vidresizeval=0.5, man_types=["c"], 
                 man_customstep=100, manonly=True, mintrajlength=10, fillmissingorientdifftresh=100, headtoorientspeedtresh=1,
                 man_frameloc=0, man_timewindow=0, man_treshold_speed=0, delcontdata=False, powermate=False):

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
        self.manfix = manfix
        self.manonly = manonly
        self.fulldata = fulldata
        self.changefps = changefps
        self.datflip = datflip
        self.convert = convert
        self.nearmaskdis = nearmaskdis
        self.inmaskdis = inmaskdis
        self.removenearmask = removenearmask
        self.trajgap = trajgap
        self.edgedis = edgedis
        self.smoothwin = smoothwin
        self.centertype = centertype
        self.centralise = centralise
        self.filllentresh_com = filllentresh_com
        self.filllentresh_headtail = filllentresh_headtail
        self.man_vidresizeval = man_vidresizeval
        self.man_types = man_types
        self.man_customstep = man_customstep
        self.mintrajlength = mintrajlength
        self.fillmissingorientdifftresh = fillmissingorientdifftresh
        self.headtoorientspeedtresh = headtoorientspeedtresh
        self.man_frameloc = man_frameloc 
        self.man_timewindow = man_timewindow
        self.man_treshold_speed = man_treshold_speed
        self.delcontdata = delcontdata
        self.powermate = powermate


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
            converters = {c:lit_converter for c in self.data.columns if str(self.data[c][0])[0]=="("}
            self.data = pd.read_csv(self.trackedfile, header=0, converters=converters)
            self.emptydat = False
        else:
            self.emptydat = True
        print("["+self.nr+"] "+"Data loaded, "+str(len(self.data))+" rows..", end=" ", flush=True)

        # Convert to value columns
        for i in self.data.loc[pd.isnull(self.data["icom"])].index:
            self.data.at[i,"icom"] = (np.nan,np.nan)
        self.data["icom"] = self.data["icom"].apply(safe_literal_eval)
        #> if there is not yet any cx,cy coordinate data, extract from icom 
        if "cx" not in self.data:
            self.data["cx"],self.data["cy"] = zip(*self.data["icom"]) if len(self.data)>0 else (np.nan,np.nan)
        #> if there is a 'head' column but no corresponding coordinates yet, create those 
        if "head" in self.data:
            for i in self.data.loc[pd.isnull(self.data["head"])].index:
                self.data.at[i,"head"] = (np.nan,np.nan)
            self.data["fx"],self.data["fy"] = zip(*self.data["head"])  if len(self.data)>0 else (np.nan,np.nan)
         #> if there is a 'tail' column but no corresponding coordinates yet, create those 
        if "tail" in self.data:
            for i in self.data.loc[pd.isnull(self.data["tail"])].index:
                self.data.at[i,"tail"] = (np.nan,np.nan)
            self.data["tx"],self.data["ty"] = zip(*self.data["tail"])  if len(self.data)>0 else (np.nan,np.nan)
        #> if there is a angle column, which arises from using orientfrombw, then change name to orient 
        if self.orientfrombw:
            self.data.rename(columns={'angle': 'orient'}, inplace=True)

        # Remove bw contours if needed
        if self.delcontdata:
            idnr = len(self.data['id'].unique())
            self.data = self.data[~self.data['id'].str.isnumeric()]
            self.data.reset_index(drop=True, inplace=True)
            if idnr > len(self.data['id'].unique()):
                print("["+self.nr+"] "+"Removed bw contour ids..", flush=True)

        # Get combinations of treshtype and identities to get unique objects
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
            if self.fileinfo.ID[0]=="(":
                self.IDs = self.fileinfo.ID.strip('()').split(',')
            else:
                self.IDs = literal_eval(self.fileinfo.ID) if self.fileinfo.ID[0]=="[" else [self.fileinfo.ID]
        self.res = literal_eval(self.fileinfo.resolution)
        self.cover = self.fileinfo["maskimg"]==self.fileinfo["maskimg"]
        self.height = self.fileinfo.roi[1][1]-self.fileinfo.roi[0][1]
        self.nearmaskdisf = self.nearmaskdis * self.conv
        self.inmaskdis = self.inmaskdis * self.conv

        self.prep()

        # Run manual tracking function
        if self.manfix:
            vidfile = self.trackedfile[:-6]+"_TR.mp4" if self.trackedfile.endswith(("_E.csv","_R.csv","_Z.csv")) else self.trackedfile[:-4]+"_TR.mp4"
            TM = Tracker_man(vidfile = vidfile,
                             data = self.data,
                             fileaction = "append",
                             ids = self.ids,
                             ptypes = self.man_types,
                             resizeval = self.man_vidresizeval,
                             internal = True,
                             customstep = self.man_customstep,
                             conv = self.conv,
                             smoothwin = self.smoothwin,
                             frameloc = self.man_frameloc, 
                             timewindow = self.man_timewindow, 
                             treshold_speed = self.man_treshold_speed,
                             powermate = self.powermate)
            TM.track()

            if self.manonly:
                return

        # Further preps for auto processing
        if self.fileinfo.bgimg == self.fileinfo.bgimg:
            self.img_bg = cv2.imread(os.path.join(self.dirs["originals"], self.fileinfo.bgimg))
        else:
            print("["+self.nr+"] "+"bg img does not exist..", end=" ", flush=True)
        if self.fileinfo.maskimg == self.fileinfo.maskimg:
            _,self.maskcoords = coordsfrommask(os.path.join(self.dirs["originals"], self.fileinfo.maskimg))
            self.maskcoords = [(x-self.fileinfo.roi[0][0],y-self.fileinfo.roi[0][1]) for x,y in self.maskcoords]
        else:
            print("["+self.nr+"] "+"Maskimg does not exist..", end=" ", flush=True)
            self.maskcoords = []
        if "wallimg" not in self.fileinfo or self.fileinfo.wallimg!=self.fileinfo.wallimg:
            self.wallcoords = []
            print("["+self.nr+"] "+"Wallimg does not exist..", end=" ", flush=True)
        else:
            _,self.wallcoords = coordsfrommask(os.path.join(self.dirs["originals"], self.fileinfo.wallimg))
        #print("")
        self.process()


    def prep(self):
        
        self.data.index = list(self.data.frame)
        if len(self.data)==0:
            self.ids = [0]
        for idind,id in enumerate(self.ids):
            
            # Remove short fragments of tracking data
            if self.removeoutliers:
                alones = getalones(self.data[self.data.id == id], "icom", self.alonewindow)
                if len(alones)>0:
                    cols = ["cx", "cy"] + (["fx", "fy"] if "fx" in self.data else []) + (["tx", "ty"] if "tail" in self.data else [])
                    self.data.loc[alones,cols] = np.nan
                    print("["+self.nr+"] "+"Removed",len(alones),"outliers", flush=True)

            # Add empty rows for each frame and id combination
            newcols = [([frame, id] + list(np.repeat(np.nan,len(self.data.columns)-2))) for frame in list(range(self.minfr,self.maxfr+1))]
            newdat = pd.DataFrame(newcols, columns = self.data.columns)
            newdat.index = list(newdat.frame)
            newdat.loc[self.data.index[self.data.id==id]] = self.data[self.data.id==id]
            final = newdat if idind==0 else pd.concat([final,newdat])

        # Sort the data by frame and id
        self.data = final.sort_values(["id","frame"])
        self.data.index = list(range(0,len(self.data)))
        self.data["frame"] = self.data["frame"].astype(int)

        # Add proper nans for com and icom data
        for i in self.data.loc[pd.isnull(self.data["icom"])].index:
            self.data.at[i,"icom"] = (np.nan,np.nan)

        print("["+self.nr+"] "+"Data prepared..", end=" ", flush=True)


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
                dat.loc[sec[0]:sec[1]+1,"icom"] = np.nan
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

        # Remove unnecessary columns 
        self.data = self.data.drop(['com', 'consmerged'], axis=1)

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
                                     inroi = 1)

        # Check each tracked object
        if not self.emptydat:
            for idind, id in enumerate(self.ids):
                print("["+self.nr+"] "+"[obj "+str(id)+"]:", end=" ", flush=True)

                # Subset data
                sub = self.data.copy()
                sub = sub.loc[sub.id == id,]
                sub.index = list(range(0, sub.index.size,1))

                # Add ID data
                sub.ID = self.IDs[idind] if self.addIDs else id

                # Check and fill in missing coordinate data and add roi status
                sub, nmissing = fillmissing(sub, ["cx","cy"], self.maskcoords, self.fileinfo.roi,
                                    5, self.nearmaskdisf, self.edgedis, self.filllentresh_com)
                print("["+self.nr+"] "+"Filled in centroid data for "+str(nmissing)+" NaN frames..", end=" ", flush=True)

                # Add trajectory numbers and mask status
                addtrajsnmaskstate(sub, self.maskcoords, self.cover, win = 5,
                                            trajgap = self.trajgap, nearmaskdis = self.nearmaskdisf)
                if self.cover and self.inmaskdis>0 and len(self.maskcoords)>0:
                    _,_,missing = removenearmask(sub, self.maskcoords, inmaskdis = self.inmaskdis, mintrajlength = self.mintrajlength)
                    print("["+self.nr+"] "+"Removed "+str(missing)+" frames near cover..",end=' ', flush=True)
                trajs = sub.loc[~np.isnan(sub.traj),"traj"].unique()
                word = " Trajs.." if len(trajs)>1 else " Traj"
                print("["+self.nr+"] "+str(len(trajs))+word+"..", end=" ", flush=True)

                # Improve front/back allocation where possible
                if "head" and "tail" in self.data:
                    missing = 0
                    totfixes = 0
                    try:
                        medarea = np.nanmedian(sub.area)
                    except:
                        medarea = None
                    for i,t in enumerate(sub.traj.unique()):
                        if t!=t:
                            subsub = sub[sub.traj!=sub.traj].copy()
                        else:
                            subsub = sub[sub.traj==t].copy()
                            subsub.index = list(range(0,len(subsub)))

                            # Remove heads and tails near roi
                            subsub.loc[:,"orienttoexcl"] = np.nan
                            newroi = ((1,1),(self.fileinfo.roi[1][0]-self.fileinfo.roi[0][0],self.fileinfo.roi[1][1]-self.fileinfo.roi[0][1]))
                            borderdist = calc_borderdistvec(subsub, newroi)
                            subsub.loc[borderdist < self.edgedis, ["orienttoexcl"]] = 1

                            # Fix heads and tails as good as possible
                            if medarea != None:
                                subsub, swappedinds = fixheadtail(subsub, tresharea=medarea)
                                subsub.loc[subsub.orienttoexcl==1, ["head","fx","fy","tail","tx","ty"]] = np.nan

                            # Get alone frames
                            alones = getalones(subsub, "fx", self.alonewindow)
                            if len(alones)>0:
                                print("["+self.nr+"] "+"Removed",len(alones),"alone head-tail frames..", end=" ", flush=True)
                                subsub.loc[alones,["fx","fy","tx","ty"]] = np.nan

                            # Fill in mising head and tail data
                            subsub, submissing = fillmissing(subsub, ["fx","fy"], None, self.fileinfo.roi, 5, lentresh=self.filllentresh_headtail)
                            subsub, submissing = fillmissing(subsub, ["tx","ty"], None, self.fileinfo.roi, 5, lentresh=self.filllentresh_headtail)
                            missing += submissing
                            totfixes += len(swappedinds)

                        temp = subsub if i == 0 else pd.concat([temp, subsub])

                    temp.index = list(range(0,len(temp)))
                    sub = temp.copy()
                    print("["+self.nr+"] "+"Swapped "+str(totfixes)+" head and tails..", end=" ", flush=True)
                    print("["+self.nr+"] "+"Filled in front data for "+str(missing)+" NaN frames..", end=" ", flush=True)

                # Smooth positional coordinates and orientation
                if self.smoothwin>1:
                    sub = smooth(sub, trajs, ["cx","cy"], self.smoothwin)
                    # Remove rows where smoothed columns are NaN
                    sub = sub.dropna(subset=["cx", "cy"])
                    print("["+self.nr+"] "+"Data smoothed..", end=" ", flush=True)

                # Merge sub with complete dataset
                final = sub if idind == 0 else pd.concat([final, sub])
        else:
            final = self.data
            
        # Re-sort dataframe before conversion operations
        final = final.sort_values(["ID","frame"])
        final = final.reset_index(drop=True)

         # Convert positional coordinates
        if self.convert and self.fileinfo.conv == self.fileinfo.conv:
            final["cx_c"],final["cy_c"] = convert(final.cx, final.cy, self.conv, self.height, self.datflip, None)
            if "fx" in self.data:
                final["fx_c"],final["fy_c"] = convert(final.fx, final.fy, self.conv, self.height, self.datflip, None)
            print("["+self.nr+"] "+"Data converted..", end=" ", flush=True)
        else:
            final["cx_c"],final["cy_c"]  = (final["cx"],final["cy"])
            if "fx" in self.data:
                final["fx_c"],final["fy_c"]  = (final["fx"],final["fy"])

        # After all centroid data is corrected, calculate key movement variables 
        for ID in final.ID.unique():
            for t in final.loc[final.ID==ID].traj.dropna().unique():
                inds = final.loc[(final.ID==ID)&(final.traj==t)].index
                final.loc[inds,"displ"] = calcudiff(final.loc[inds,"cx_c"], final.loc[inds,"cy_c"])
                final.loc[inds,"speed"] = final.loc[inds,"displ"] * self.fps / 10 #cm/s
                final.loc[inds,"accel"] = differentiate(final.loc[inds,"speed"]) * self.fps
                final.loc[inds,"heading"] = calcudiff(final.loc[inds,"cx_c"], final.loc[inds,"cy_c"], angle=True)
                final.loc[inds,"turnspeed"] = get_anglediff(final.loc[inds,"heading"])
                final.loc[inds,"turnaccel"] = get_anglediff(final.loc[inds,"turnspeed"]) 
            final.loc[final.ID==ID,"cumdispl"] = np.nancumsum(final.loc[final.ID==ID,"displ"])

        if "fx" in final.columns:
            print("["+self.nr+"] "+"Improving orient..", end=" ", flush=True)
            final["orient"] = None

            for ID in final.ID.unique():
                for t in final.loc[final.ID==ID].traj.dropna().unique():
                    # Get the subset indices for this ID and trajectory
                    inds = final.loc[(final.ID == ID) & (final.traj == t)].index

                    # Compute orient from head coordinates (fx, fy)
                    final.loc[inds,"orient"] = points_to_angle(
                        (final.loc[inds,"cx"], final.loc[inds,"cy"]),
                        (final.loc[inds,"fx"], final.loc[inds,"fy"]), 
                        flip=True)

                    # Add heading values to empty orient cells if speed >= threshold
                    final.loc[inds, "orient"] = final.loc[inds].apply(
                        lambda row: row['heading'] if pd.isnull(row['orient']) and row['speed'] >= self.headtoorientspeedtresh else row['orient'],
                        axis=1)

                    # Ensure 'orient' is numeric and handle NaN values
                    # Convert 'orient' to numeric for the entire selection (including 'inds' rows)
                    final.loc[inds, 'orient'] = pd.to_numeric(final.loc[inds, 'orient'], errors='coerce')

                    # Filter valid rows where 'orient' is a valid number (not NaN)
                    valid_inds = final.loc[inds, 'orient'].dropna().index

                    # Apply your 'angle_to_vec' formula for valid rows only
                    result = final.loc[valid_inds, "orient"].apply(angle_to_vec)
                    final.loc[valid_inds, ["vx","vy"]] = result.apply(pd.Series, index=["vx", "vy"])

                    # Fill in missing values as needed
                    final.loc[inds], missing = fillmissing(final.loc[inds], ["vx", "vy"], None, None, lentresh=self.fillmissingorientdifftresh)

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

        # # Fix any remaining issues with orientation by setting those rows to NA
        # if "fx" in self.data:
        #     indstona = final.index.values[abs(final.turnspeed)>450]
        #     for ind in indstona:
        #         final.loc[ind-3:ind+3,["orient","turnspeed","fx","fy","tx","ty"]] = np.nan
        #     print(str(len(indstona))+" rows with too high turning speeds set to NA..", end=" ")

        # Add structural distance measures
        ## To region of interest
        roi = tuple([(pt[0]*self.conv,pt[1]*self.conv) for pt in self.fileinfo.roi])
        final["rdist"] = [calc_borderdist((x,y),roi) for (x,y) in zip(final.cx_c,final.cy_c)]
        final.loc[final.inroi==0,"rdist"] = 0

        ## To mask
        if self.maskcoords not in (None,[]):
            if self.convert:
                self.mask_c = [convert(x,y,self.conv,self.height,self.datflip,None) for x,y in self.maskcoords]
            else:
                self.mask_c = self.maskcoords
            self.maskcoords = [convert(x,y,1,self.height,False,None) for x,y in self.maskcoords]
            if len(final.cx_c[~np.isnan(final.cx_c)])==0:
                final["mdist"] = 0
                final["mx"],final["my"] = np.nan, np.nan
            else:
                final["mdist"], pixid = KDTree(self.mask_c).query(list(zip(final.cx_c,final.cy_c)))
                final.loc[final["mdist"]==np.Inf,"mdist"] = np.nan
                final["mx"],final["my"] = zip(*[self.maskcoords[i] if i<len(self.maskcoords) else (np.nan,np.nan) for i in pixid])
                if self.cover:
                    final.loc[final.inmask==1,"mdist"] = 0
            print("["+self.nr+"] "+"Mask distances computed..", end=" ", flush=True)

        ## To walls
        if self.wallcoords not in (None,[]):
            if self.convert:
                self.wall_c = [convert(x,y,self.conv,self.height,self.datflip,self.fileinfo.roi) for x,y in self.wallcoords]
            else:
                self.wall_c = self.wallcoords
            self.wallcoords = [convert(x,y,1,self.height,False,self.fileinfo.roi) for x,y in self.wallcoords]
            invalid_rows = final[
                final.cx_c.isna() | final.cy_c.isna() | np.isinf(final.cx_c) | np.isinf(final.cy_c)
            ]
            # # Print detailed information about the invalid rows
            # if not invalid_rows.empty:
            #     print("["+self.nr+"] "+"Invalid rows detected:", flush=True)
            #     print("["+self.nr+"] ", invalid_rows[["frame", "cx_c", "cy_c"]], flush=True)  # Include relevant columns
            valid_rows = ~final.cx_c.isna() & ~final.cy_c.isna() & ~np.isinf(final.cx_c) & ~np.isinf(final.cy_c)
            filtered_points = list(zip(final.cx_c[valid_rows], final.cy_c[valid_rows]))
            if len(filtered_points) == 0:
                final["wdist"] = 0
                final["wx"],final["wy"] = np.nan, np.nan
            else:
                final["wdist"], pixid = KDTree(self.wall_c).query(filtered_points)
                final.loc[final["wdist"] == np.Inf,"wdist"] = np.nan
                final["wx"],final["wy"] = zip(*[self.wallcoords[i] if i<len(self.wallcoords) else (np.nan,np.nan) for i in pixid])
            print("["+self.nr+"] "+"Wall distances computed..", end=" ", flush=True)

        ## To center (either based on roi or walls)
        if self.centertype is not None:
            center = None
            try:
                x,y = literal_eval(self.fileinfo.pt)
                center = convert(x, y, self.conv, self.fileinfo.roi[1][1]-self.fileinfo.roi[0][1], True, self.fileinfo.roi)
            except:
                if self.centertype == "roi":
                    center = ((roi[1][0]-roi[0][0])/2,(roi[1][1]-roi[0][1])/2)
                if hasattr(self,"wall_c"):
                    center = (np.average(list(zip(*self.wall_c))[0]), np.average(list(zip(*self.wall_c))[1]))
            if center is not None:
                final["cdist"] = [ptsToDist((x,y),center) for x,y in zip(final.cx_c,final.cy_c)]
                print("["+self.nr+"] "+"Center distances computed..", end=" ", flush=True)
            else:
                print("["+self.nr+"] "+"Center/pt distance computation could not be performed..", end=" ", flush=True)

        # Centralise data
        if self.centralise and center is not None:
            final.cx_c = final.cx_c - center[0]
            final.cy_c = final.cy_c - center[1]
            if "fx_c" in final:
                final.fx_c = final.fx_c - center[0]
                final.fy_c = final.fy_c - center[1]
            print("["+self.nr+"] "+"Data centralised..", end=" ", flush=True)

        # Eye positions and visual field information
        ## todo

        # Final data organisation
        if not (self.convert and self.fileinfo.conv == self.fileinfo.conv):
            final = final.drop(columns=["cx_c","cy_c","fx_c","fy_c"], errors='ignore')
        if self.orientfrombw:
            final = final.drop(columns=["fx","fy","fx_c","fy_c"])

        # Reorganise colums
        datacols = ["frame","time","ID","traj","inroi","inmask","cx","cy","cx_c",
                    "cy_c","fx","fy","fx_c","fy_c","displ","speed","cumdispl","accel",
                    "heading","orient","turnspeed","turnaccel","cumturn","abscumturn",
                    "rdist","mdist","wdist","cdist","mx","my","wx","wy"]
        final = final[[col for col in datacols if col in final.columns]]

        ## Round data columns
        cols = ["turnspeed","turnaccel"]
        final = final.round({i:3 for i in cols if i in final.columns})
        cols = ["time","cx","cy","fx","fy","mx","my","wx","wy","cx_c","cy_c","fx_c","fy_c","displ","speed","accel","turnspeed","turnaccel"]
        final = final.round({i:2 for i in cols if i in final.columns})
        cols = ["cumdispl","cumturn","abscumturn","heading","orient","dist","mdist","wdist","cdist","rdist"]
        final = final.round({i:1 for i in cols if i in final.columns})

        ## Write data to file
        final.to_csv(self.procfile, index=False)
        print("["+self.nr+"] "+"File written..", flush=True)