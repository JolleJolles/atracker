#! /usr/bin/env python

import os
import cv2
import pandas as pd
import numpy as np
from itertools import cycle

from pythutils.mediautils import check_media, get_vid_params, imgresize
from pythutils.drawutils import namedcols, draw_text, draw_crosshair
from pythutils.fileutils import name
from pythutils.datutils import create_emptydf, pd_to_coords

from .utils import *

class Tracker_man:

    """
    Initializes a manual tracking instance

    Parameters
    ----------
    vidfile : str; no default
        Name of the video file to be manually tracked.
    fileaction : ["replace","newfile","append"]; default = "newfile"
        In the case the trackingfile already exists, either replace the
        file, make a new file with a sequence number, or append to the file.
    ids : list; default = ["1"]
        A list of animal IDs that will be tracked.
    ptypes : list of ["c","h","t"]; default = ["c"]
        One or multiple point locations that need to be tracked, can be "c",
        coordinate in the centre of the animal's body; "f", tip coordinate
        on the front of the animal; "b", bottom coordinate on the back of
        the animal.
    safecount : boolean; default = False
        If an extra careful count should be made for the total number of
        frames. This is necessary when videos likely contain dropped or
        replicated frames, which will otherwise likely result in an error.
        >! left this in for now, but likely obsolete due to use of find_max_working_pyframe function
    datacrop : boolean; default = False
        If the data should be cropped to the first and last manually tracked
        data point. When start and stopframes are provided datacrop will be
        automatically set to True.
    firstframe : int; default = None
        Frame before which the manually tracked datafile should be cropped.
    lastframe : int; default = None
        Frame after which the manually tracked datafile should be cropped.
    resizeval : float; default = 1
        Value with which the video should be resized. Values larger than 1
        will increase the video size, facilitating careful tracking, while
        those smaller than 1 will decrease video size, facilitating faster
        manual tracking.
    statevar : str; default = None
        Name of a potential state variable that can be coded as 0 and 1.
    customstep : int; default = None
        Custom (forward) step size in frames.
    """

    def __init__(self, vidfile, data = [], fileaction = "newfile", ids = ["0"],
                 ptypes = ["c"], safecount = False, datacrop = False,
                 firstframe = None, lastframe = None, resizeval = 1, arrowl = 40,
                 statevar = None, customstep = None, indir = True, internal=False, 
                 conv=1, min_speed=0, max_speed=100, smoothwin=25,
                 frameloc = 0, timewindow = 0, threshold_speed = 0, powermate = False):

        check_media(vidfile)

        self.resizeval = resizeval
        self.multiplier = 1./self.resizeval
        self.datacrop = datacrop
        self.fileaction = fileaction
        self.internal = internal
        self.vidfile = vidfile
        self.data = data
        self.statevar = statevar
        self.arrowl = arrowl
        self.smoothwin = smoothwin

        types = {"c": ("com", ["cx","cy"], "orange"),
                 "h": ("head", ["fx","fy"], "green"),
                 "t": ("tail", ["tx","ty"], "blue"),
                 "o": ("orient", ["fx","fy"], "green")}
        self.ids = ids
        self.idpool = cycle(self.ids)
        self.types = [types[x] for x in ptypes]
        self.typepool = cycle(self.types)
        self.ustep = 1 if customstep is None else customstep

        self.cap = cv2.VideoCapture(self.vidfile)
        self.fps, self.width, self.height, self.fcount = get_vid_params(self.cap)
        max_pyframe = find_max_working_pyframe(self.cap)
        self.fcount = max_pyframe+1
        self.width = int(self.width * self.resizeval)
        self.height = int(self.height * self.resizeval)

        self.conv = conv
        print("Using conversion measure", self.conv, end='\r')

        self.min_speed = min_speed
        self.max_speed = max_speed 

        # Load or create datafile
        def name(filename, ext = "", action = "newfile", suffix="", indir = indir):

            """
            Gets the name for a file with required extension, and will either overwrite
            the existing file or generate a new file with a sequence.
            """

            dirname, filename = os.path.split(filename)
            dirname = '.' if dirname == '' else dirname
            filename, fileext = os.path.splitext(filename)
            if "_TR" in filename:
                filename = filename[:-3]
            names = [x for x in os.listdir(dirname) if x.startswith(filename)]
            names = [x for x in names if os.path.splitext(x)[1]==ext]

            if len(names) == 0 or action == "append":
                return os.path.join(dirname, filename+ext)

            elif action == "overwrite":
                if os.path.exists(filename+ext):
                    os.remove(filename+ext)
                return os.path.join(dirname, filename+ext)

            elif action == "newfile":
                names = [os.path.splitext(x)[0] for x in names]
                if suffix != "":
                    filename = os.path.join(dirname, filename+suffix+ext)
                else:
                    suffixes = [x.replace(filename, '') for x in names]
                    suffixes = [int(x[1]) for x in suffixes if x.startswith('_')]
                    suffix = 2 if len(suffixes)==0 else max(suffixes)+1
                    filename = os.path.join(dirname, filename)
                    filename = '%s_%d%s' % (filename, suffix, ext)
                return filename

        if len(self.data)==0:
            self.datafile = name(self.vidfile, ".csv", fileaction, indir=indir)
            if os.path.isfile(self.datafile):
                self.data = pd.read_csv(self.datafile, header = 0)
                print("Datafile "+os.path.split(self.datafile)[1]+" loaded..")
                ind = self.data.loc[(self.data.frame == self.data.frame[0])].index[0]
                self.data.index = list(range(ind, ind+len(self.data)))
            else:
                if action=="newfile":
                    if safecount:
                        print("Running safe frame count..",end='')
                        self.fcount = safe_count(vidfile)
                    self.firstframe = firstframe if firstframe is not None else 1
                    self.lastframe = lastframe if lastframe is not None else self.fcount
                    self.columns = sum([i[1] for i in self.types], [])
                    if statevar is not None:
                        self.columns = self.columns + [statevar]
                    self.data = create_emptydf(self.columns, self.ids, self.firstframe, self.lastframe)
                    #self.data = data.combine_first(self.data)
                    print("Empty datafile '"+os.path.split(self.datafile)[1]+"' created..")
        else:
            if self.internal:
                self.datafile = name(self.vidfile, ".csv", "newfile", suffix="_E", indir=indir)
            if os.path.isfile(self.datafile):
                self.data = pd.read_csv(self.datafile, header = 0)
                print("Edited datafile "+os.path.split(self.datafile)[1]+" loaded..")
            self.firstframe = int(self.data.frame[0])
            self.lastframe = int(self.data.frame[len(self.data)-1])

        # Compute orientation points
        if "orient" in self.data.columns and "fx" not in self.data.columns:
            def compute_hpt(row):
                if pd.notna(row['orient']) and pd.notna(row['cx']) and pd.notna(row['cy']):
                    hvec = angle_to_vec(row['orient'])
                    fx = int(row['cx'] + hvec[0] * self.arrowl)
                    fy = int(row['cy'] + hvec[1] * self.arrowl*-1)
                    return fx, fy
                else:
                    return np.nan, np.nan
            self.data[['fx', 'fy']] = self.data.apply(lambda row: pd.Series(compute_hpt(row)), axis=1)
            print("Orientation coordinates computed")

        self.datacopy = self.data.copy()
        
        self.add = False
        self.mousept = None
        self.ind_start = None
        self.ind_stop = None
        self.drawcoords = True
        self.showboth = True
        self.showmouse = False
        self.show_speedcoords = False
        self.drawframes = False
        self.drawallframes = True
        self.dragging = False
        self.rectangle = None
        self.drawrectangle = False
        self.selected_points = None
        self.crop_firstframe = self.firstframe
        self.crop_lastframe = self.lastframe

        self.framediff = self.lastframe-self.firstframe
        self.nsteps, self.stepsize = self.framediff, 1 #maxsteps(self.framediff, 200)

        self.timewindow = timewindow
        self.frameloc = frameloc 
        self.current_frame = None
        self.smoothed_angles = None
        self.show_smoothed_angles = False 
        self.threshold_speed = threshold_speed

        self.powermate = powermate
        
        self.uset_id(False)
        self.uset_type(True)

        self.show_windows()
        cv2.setMouseCallback('Video', self.drawpoint)
        
        if self.timewindow != 0:
            self.uset_frameloc(1)    

    def show_windows(self):

        def nothing(x):
            pass

        ipanel_width = 300
        cv2.namedWindow('Frame position', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Frame position', int(self.width+ipanel_width), 30)
        cv2.moveWindow('Frame position', 0, 0)
        self.barpos = self.frameloc
        self.trackpos = 0
        cv2.createTrackbar('PyFrame', 'Frame position', self.barpos, self.nsteps, nothing)

        cv2.namedWindow('Time window', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Time window', int(self.width+ipanel_width), 50)
        cv2.moveWindow('Time window', 0, 137)
        cv2.createTrackbar('Frames', 'Time window', self.timewindow, self.framediff, nothing)

        cv2.namedWindow('Info panel', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Info panel', ipanel_width, 160)
        cv2.moveWindow('Info panel', 0, 249)
        
        cv2.namedWindow('Speed threshold', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Speed threshold', ipanel_width, 30)
        cv2.moveWindow('Speed threshold', 0, 437)
        cv2.createTrackbar('Speed threshold', 'Speed threshold', int(self.threshold_speed), 100, nothing)

        cv2.namedWindow('Smooth window', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Smooth window', ipanel_width, 30)
        cv2.moveWindow('Smooth window', 0, 549)
        cv2.createTrackbar('Smooth window', 'Smooth window', int(self.smoothwin), 200, nothing)

        cv2.namedWindow('Video', cv2.WINDOW_GUI_NORMAL)
        screen_height = get_screen_dimensions()[1]
        ymove = 249
        cv2.resizeWindow('Video', self.width, min(self.height, int((screen_height-ymove)/2)))
        cv2.imshow("Video", self.draw_frame)
        cv2.moveWindow('Video', ipanel_width, ymove)


    def drawpoint(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.drawrectangle:
                self.pt = (x,y)
                self.draw()
                self.add = True
            else:
                self.rectangle = (x,y,x,y)
                self.dragging = True
        elif event == cv2.EVENT_MOUSEMOVE:
            self.mousept = (x,y)
            self.draw()
            if self.drawrectangle:
                if self.dragging:
                    self.rectangle = (self.rectangle[0], self.rectangle[1], x, y)
                    self.draw()
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawrectangle:
                if self.dragging:
                    self.rectangle = (self.rectangle[0], self.rectangle[1], x, y)
                    self.dragging = False
                    self.draw()
                    self.select_points()


    def select_points(self):

        if self.rectangle is not None:
            x1,y1,x2,y2 = tuple(int(c*self.multiplier) for c in self.rectangle)
            x1,x2 = min(x1,x2),max(x1,x2)
            y1,y2 = min(y1,y2),max(y1,y2)
            self.selected_points = self.data[(self.data["cx"] >= x1) & 
                                        (self.data["cx"] <= x2) & 
                                        (self.data["cy"] >= y1) & 
                                        (self.data["cy"] <= y2)]
            self.selected_points =  self.selected_points[(self.selected_points.frame > self.crop_firstframe) &
                                                        (self.selected_points.frame < self.crop_lastframe)]


    def uset_id(self, reset = True):

        self.id = next(self.idpool)
        if reset:
            self.reset()


    def uset_type(self, reset = True):  

        self.type = next(self.typepool)
        self.label = self.type[0]
        self.subcolumns = self.type[1]
        self.col = namedcols(self.type[2])
        if reset:
            self.reset()


    def uset_frameloc(self, change):

        self.frameloc = self.frameloc+change
        if self.frameloc >= self.lastframe-self.firstframe:
            self.frameloc = self.lastframe-self.firstframe
        if self.frameloc <= self.firstframe-self.firstframe:
            self.frameloc = self.firstframe-self.firstframe
        trackpos = int(self.frameloc/self.stepsize)
        self.trackpos = 0 if trackpos < 0 else trackpos
        cv2.setTrackbarPos('PyFrame','Frame position', self.trackpos)
        
        self.crop_firstframe = max(self.firstframe, self.frameloc+self.firstframe-self.timewindow)
        self.crop_lastframe = min(self.framediff+self.firstframe, self.frameloc+self.firstframe+self.timewindow)
        self.select_points()
        self.reset()


    def reset(self):

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frameloc)
        _, self.frame = self.cap.read()
        self.frame = imgresize(self.frame, self.resizeval)
        self.pt = None
        self.loc = self.data.index[(self.data.frame == self.firstframe+self.frameloc) & (self.data.id == self.id)][0]
        self.draw()


    def draw(self):

        # Draw frame
        self.draw_frame = self.frame.copy()

        # Draw points
        if self.drawcoords:
            
            if self.timewindow>1:
                self.sub = self.data[self.data.id == self.id].reset_index(drop=True)
                self.sub = self.sub[(self.sub.frame >= self.crop_firstframe) &
                          (self.sub.frame < self.crop_lastframe)].reset_index(drop=True)

                # Centroid coordinates    
                self.sub = smooth(self.sub, None, ["cx","cy"], self.smoothwin)
                coords, framelist = pd_to_coords(self.sub, None, True, ["cx","cy"], self.resizeval)
                plotcoords = coords
                if self.show_speedcoords:
                    self.sub['distances'] = np.sqrt((self.sub['cx'].diff() ** 2) + (self.sub['cy'].diff() ** 2))
                    self.sub['distances_mm'] = self.sub['distances'] * self.conv
                    self.sub['time_diffs'] = self.sub['frame'].diff()
                    self.sub['speed'] = self.sub['distances_mm'] / self.sub['time_diffs'] * self.fps

                # Draw orientations
                if self.label=="orient":

                    head_coords, head_framelist = pd_to_coords(self.sub, None, True, ["fx","fy"], self.resizeval)               

                    # Check if frame has changed 
                    if (self.frameloc != self.current_frame) or self.force_redraw:
                        self.current_frame = self.frameloc
                        self.show_smoothed_angles = False
                        self.smoothed_angles = None
                        self.force_redraw = False

                        # Compute non-smoothed arrows for this frame
                        self.angledata = []
                        for i, head_coord in enumerate(head_coords):
                            head_coord = tuple(head_coord[0])
                            frame_number = head_framelist[i]
                            if frame_number in framelist:
                                coord_index = framelist.index(frame_number)
                                coord = tuple(coords[coord_index][0])
                                direction_vector = np.array(head_coord) - np.array(coord)
                                angle = np.degrees(np.arctan2(direction_vector[1], direction_vector[0]))
                                self.angledata.append({"coord": coord, "angle": angle})
                    
                    if self.show_smoothed_angles and self.smoothed_angles is None:
                        # Compute smoothed arrows
                        angles = [entry["angle"] for entry in self.angledata]
                        angles_rad = np.radians(angles)
                        sin_vals = np.sin(angles_rad)
                        cos_vals = np.cos(angles_rad)
                        valid_min_periods = min(self.smoothwin, len(sin_vals))
                        smoothed_sin = pd.Series(sin_vals).rolling(window=self.smoothwin, center=True, min_periods=valid_min_periods).mean().fillna(0)
                        smoothed_cos = pd.Series(cos_vals).rolling(window=self.smoothwin, center=True, min_periods=valid_min_periods).mean().fillna(0)
                        smoothed_angles = np.degrees(np.arctan2(smoothed_sin, smoothed_cos))
                        
                        self.smoothed_angles = [
                            {"coord": entry["coord"], "angle": smoothed_angles[i]} 
                            for i, entry in enumerate(self.angledata)
                        ]

                    # Select the appropriate list of angles to draw
                    if self.show_smoothed_angles and isinstance(self.smoothed_angles, list):
                        arrows_to_draw = self.smoothed_angles
                    else:
                        arrows_to_draw = self.angledata                   

                    self.draw_frame = self.frame.copy()  # Clear previous non-smoothed drawings
                    for entry in arrows_to_draw:
                        coord = entry["coord"]
                        angle = entry["angle"]
                        angle_rad = np.radians(angle)
                        direction_vector = np.array([np.cos(angle_rad), np.sin(angle_rad)]) * 15
                        if np.isnan(coord[0]) or np.isnan(coord[1]) or np.any(np.isnan(direction_vector)):
                            continue    
                        end_point = (int(coord[0] + direction_vector[0]), int(coord[1] + direction_vector[1]))

                        # Map angle to red-to-green color scale
                        normalized_angle = (180 - angle) % 360  # Reverse the angle for clockwise direction
                        if normalized_angle > 180:
                            normalized_angle -= 360  # Wrap angles > 180 back to [-180, 180]
                        normalized_magnitude = abs(normalized_angle) / 180.0  # Scale |angle| to [0, 1]
                        red_intensity = int(255 * normalized_magnitude)
                        green_intensity = int(255 * (1 - normalized_magnitude))
                        color = (0, green_intensity, red_intensity)

                        cv2.arrowedLine(self.draw_frame, coord, end_point, color, 2, tipLength=0.6)               
               
                # Draw frame numbers
                if self.drawframes:
                    if self.drawallframes:
                        draw_text(self.draw_frame, str(int(framelist[i])), (coord[0]-8,coord[1]), size =  0.3)
                    else:
                        if not frameshown:
                            if (self.mousept[0] - 5 <= coord[0] <= self.mousept[0] + 5 and
                                self.mousept[1] - 5 <= coord[1] <= self.mousept[1] + 5):
                                frametext = str(int(framelist[i]))
                                (text_width, text_height), baseline = cv2.getTextSize(frametext, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                                x = max(0, min(coord[0], self.width - text_width))
                                y = max(0, min(coord[1]-2*text_height, self.height - (baseline + 2*text_height)))
                                frameshown = True
                
                # Compute speeds
                if self.showboth or not self.label=="orient":
                    if not self.label=="orient":
                        if self.show_speedcoords:
                            if len(self.sub['distances_mm']) > 1:
                                inds = self.sub.index[self.sub["frame"].isin(framelist)].tolist()
                                speeds = self.sub.loc[inds, "speed"].dropna().to_numpy()
                                valid_inds = self.sub.loc[inds, "speed"].notna()
                                plotcoords = coords[valid_inds]
                                norm_speeds = 1 - (speeds - self.min_speed) / (self.max_speed - self.min_speed)
                                circle_radii = [max(int(1 + 6 * ns), 1) for ns in norm_speeds]                      

                    cv2.polylines(self.draw_frame, [plotcoords], False, namedcols("black"), 1)
                    frameshown = False
                    if not self.label=="orient":
                        for i,coord in enumerate(plotcoords[1:]):
                            coord = tuple(coord[0])
                            if self.show_speedcoords:
                                circle_radius = circle_radii[i]
                            
                            # Check if speed is below or above the threshold
                            color = "orange" 
                            if self.show_speedcoords:
                                color = "orange" if speeds[i - 1] >= self.threshold_speed else "red"
                                cv2.circle(self.draw_frame, coord, circle_radius, namedcols(color), -1)
                            else:
                                cv2.circle(self.draw_frame, coord, 3, namedcols(color), -1)

                            
                            # Draw framenumbers
                            if self.drawframes:
                                if self.drawallframes:
                                    draw_text(self.draw_frame, str(int(framelist[i])), (coord[0]-8,coord[1]), size =  0.3)
                                else:
                                    if not frameshown:
                                        if (self.mousept[0] - 5 <= coord[0] <= self.mousept[0] + 5 and
                                            self.mousept[1] - 5 <= coord[1] <= self.mousept[1] + 5):
                                            frametext = str(int(framelist[i]))
                                            (text_width, text_height), baseline = cv2.getTextSize(frametext, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                                            x = max(0, min(coord[0], self.width - text_width))
                                            y = max(0, min(coord[1]-2*text_height, self.height - (baseline + 2*text_height)))
                                            frameshown = True
                
                # Draw frame numbers
                if self.drawframes and not self.drawallframes and frameshown:
                    draw_text(self.draw_frame, frametext, (x,y), size=0.3, bgcol="white", margin=1)
                               
           
        # Draw final things on top                    
        ## Draw centroid
        pt = pd_to_coords(self.data, self.loc, True, ["cx","cy"], self.resizeval)
        com = pt
        if pt is not None:
            cv2.circle(self.draw_frame, pt, 0, namedcols("orange"), 8)
        if self.pt is not None:
            col = namedcols("orange") if self.label=="com" else namedcols("blue")
            cv2.circle(self.draw_frame, self.pt, 0, col, 5)
            pt = self.pt
      
        # Draw head point
        if self.drawcoords and self.showboth:
            if "fx" in self.data.columns:
                if pd.notna(self.data.loc[self.loc,"fx"]) and com is not None:
                    direction_vector = np.array(tuple(self.data.loc[self.loc, ["fx", "fy"]].astype(int))) - np.array(com)
                    direction_vector = direction_vector / np.linalg.norm(direction_vector) * 10
                    end_point = (int(com[0] + direction_vector[0]), int(com[1] + direction_vector[1]))
                    cv2.arrowedLine(self.draw_frame, com, end_point, namedcols("black"), 2, tipLength=0.5)
        
        ## Draw mouse coordinates
        if self.showmouse:
            if self.mousept is not None:
                draw_text(self.draw_frame, str(self.mousept), (self.mousept[0]-10, self.mousept[1]+10), size=0.3, bgcol="white", margin=1)
        
        ## Draw rectangle
        if self.rectangle is not None:
            x1, y1, x2, y2 = self.rectangle
            cv2.rectangle(self.draw_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        ## Draw mouse pointer
        if self.mousept is not None:
            draw_crosshair(self.draw_frame, self.mousept, col="black")

        # Draw Params
        fontsize = 0.5
        tab = 72
        self.draw_params = np.zeros((120,300,3), np.uint8)+255
        selector_text = "rect" if self.drawrectangle else "point"
        selector_col = "green" if self.drawrectangle else "red"
        draw_text(self.draw_params, "Selector:", (0,5), size = fontsize)
        draw_text(self.draw_params, selector_text, (tab, 5), size =  fontsize, col = selector_col)
        draw_text(self.draw_params, "Frame:", (0,25), size = fontsize)
        frame_info = f"{self.firstframe+self.frameloc} [{self.crop_firstframe}-{self.crop_lastframe}]"
        draw_text(self.draw_params, frame_info, (tab, 25), size = fontsize)
        draw_text(self.draw_params, "ID:", (0, 45), size =  fontsize)
        draw_text(self.draw_params, str(self.id), (tab, 45), size =  fontsize)
        draw_text(self.draw_params, "Type:", (0, 65), size = fontsize)
        draw_text(self.draw_params, self.label, (tab, 65), size =  fontsize, col = self.col)
        draw_text(self.draw_params, "Coords:", (0, 85), size =  fontsize)
        if self.label == "com":
            drawpt = pd_to_coords(self.data, self.loc, True, ["cx","cy"], self.resizeval)
        elif self.label == "head" or self.label == "orient":
            drawpt = pd_to_coords(self.data, self.loc, True, ["fx","fy"], self.resizeval)
        else: 
            drawpt = None
        if drawpt is not None:
            draw_text(self.draw_params, str(drawpt), (tab, 85), size =  fontsize)
        if self.statevar is not None:
            state = self.data.loc[self.loc, self.statevar]
            state = str(int(state)) if state == state else "nan"
            draw_text(self.draw_params, "State ("+self.statevar+"): "+state, (0, 105), size =  fontsize)


    def savedat(self):

        changes = ~self.datacopy["cx"].eq(self.data["cx"]) & ~(self.datacopy["cx"].isna() & self.data["cx"].isna())
        changes = changes.sum()
        print(changes)
        if "fx" in self.datacopy.columns:
            changes2 = ~self.datacopy["fx"].eq(self.data["fx"]) & ~(self.datacopy["fx"].isna() & self.data["fx"].isna())
            changes = changes + changes2.sum()
        print(changes)
        temp = "change" if changes == 1 else "changes"
        changes = print("User saved..", changes, "row", temp, "recorded..", end= " ")
        if self.datacrop:
            inds = list(self.data.dropna(thresh = 4).index)
            if len(inds)==0:
                print("dataset was emtpy..")
            else:
                if len(inds)<3:
                    inds = [self.firstframe, self.lastframe]
                inds = [inds[0],inds[-1]]
                if self.ind_start != None:
                    inds[0] = self.ind_start
                if self.ind_stop != None:
                    inds[1] = self.ind_stop
                self.data = self.data.loc[inds[0]:inds[-1],]
                print("dataset cropped to frames " + str(min(self.data.frame)) + ":" + str(max(self.data.frame)) + "..", end='')
        if len(self.data) > 2:
            self.data.to_csv(self.datafile, index = False)


    def keypress(self):

        '''
        This function enables the user to use simple keypresses to
        command most of the manual tracking functionality.

        Parameters
        ----------
        Keys q,w,e,r,t,y are used for controlling frame position:
        q : go to first frame
        w : go one second back
        e : go one frame back
        r : go one frame forward
        t : go one second forward
        y : go to the last frame
        u : custom step size forward

        Change selection:
        c : point <> rectangle drawing

        Control the type of data tracked:
        i : flip to next animal ID
        p : flip to next point type
        d : deletes current point
        f : forgets selected_points

        Change the animal state:
        x : change state animal is in (0 <> 1).
            As the default state is NaN, make sure 
            to press x twice to set state to 0

        Visualisations:
        a : toggle showing all currently tracked data
        z : toggle showing framenumbers for current data
        ` : toggle showing all framenumbers/only near the mouse
        l : toggle showing only orientation data
        h : toggle showing mouse coordinates
        g : toggle showing smoothed angles
        j : toggle showing coordinates based on speed
        v : print changes made so far
        b : print tracking data of current time window

        Storing and exiting:
        n : saves data and creates new datafile
        s : saves data and exits
        esc : exits the program without saving
        '''

        if self.key == 255:
            return None

        else:
            
            # save and create new file
            if self.key == ord("n"):
                self.savedat()
                self.data = self.datacopy.copy()
                print("File saved, created additional datafile..")
                self.datafile = name(self.vidfile, ".csv", self.fileaction)
                return True

            # save and exit
            elif self.key == ord("s"):
                self.savedat()
                print("File saved, exiting..")
                return False

            # exit without saving
            elif self.key == 27:
                self.data = self.datacopy.copy()
                print("User exited.. changes discarded..")
                return False
            
            # change drawing point <> rectangle
            elif self.key == ord("c"):
                self.drawrectangle = not self.drawrectangle
                self.rectangle = None
                self.reset()

            # change showing all frames or only near the mouse
            elif self.key == ord("`"):
                self.drawallframes = not self.drawallframes
                self.reset()

            # display changes
            else:
                if self.key == ord("v"):
                    # Create a mask where 'cx' is NaN in one DataFrame and not NaN in the other
                    mask = ~self.datacopy["cx"].eq(self.data["cx"]) & ~(self.datacopy["cx"].isna() & self.data["cx"].isna())
                    row_changes = mask.sum()
                    changed_frames = self.data[mask]["frame"].tolist()
                    print(row_changes, "rows changed so far:", changed_frames, end='\n')
                
                elif self.key == ord("b"):
                    if "fx" in self.sub: 
                        print(self.sub.loc[:,["frame","cx","cy","fx","fy"]])
                    else: 
                        print(self.sub.loc[:,["frame","cx","cy"]])

                # change display
                else:

                    # change id
                    if self.key == ord("i"):
                        self.force_redraw = True
                        self.uset_id()

                    # change point type
                    if self.key == ord("p"):
                        self.uset_type()
                        #print("Changed to",self.label)

                    # remove selected points from memory
                    if self.key == ord("f"):
                        print("selected points forgotten")
                        self.selected_points = None
                        
                    # Delete current point or selected points
                    if self.key == ord("d"):
                        if self.selected_points is None: # single point deletion
                            self.data.loc[self.loc, self.subcolumns] = np.nan
                            delframes = str(self.firstframe+self.frameloc)
                            self.pt = None
                        else:
                            delframes = ",".join(map(str,self.selected_points.frame.tolist()))
                            self.data.loc[self.selected_points.index, self.subcolumns] = np.nan
                            self.selected_points = None
                        temp = "Frame" if delframes == 1 else "Frames" 
                        print(temp, "deleted:", delframes)
                        self.force_redraw = True
                        self.reset()

                    # Toggle showing data
                    if self.key == ord("a"):
                        self.drawcoords = not self.drawcoords
                        self.reset()

                    # Toggle showing centroid data when showing orientation data
                    if self.key == ord("l"):
                        self.showboth = not self.showboth
                        self.reset()
                    
                    # Toggle showing smoothed angles
                    if self.key == ord("g"):
                        self.show_smoothed_angles = not self.show_smoothed_angles
                        self.reset()
                    
                    # Toggle showing normal coords or based on speed
                    if self.key == ord("j"):
                        self.show_speedcoords = not self.show_speedcoords
                        self.reset()

                    # Toggle showing mouse position
                    if self.key == ord("h"):
                        self.showmouse = not self.showmouse
                        self.reset()

                    # Toggle showing frame numbers
                    if self.key == ord("z"):
                        self.drawframes = not self.drawframes
                        self.reset()

                    # set start and end of data
                    if self.key in [ord("["),ord("]")]:
                        ind = self.data.loc[(self.data.frame == self.frameloc+self.firstframe)].index[0]
                        self.datacrop = True
                        if self.key == ord("["):
                            print("Data will be cropped to start at frame",self.frameloc+self.firstframe)
                            self.ind_start = ind
                        if self.key == ord("]"):
                            print("Data will be cropped to stop at frame",self.frameloc+self.firstframe)
                            self.ind_stop = ind

                    # change state
                    if self.key == ord("x") and self.statevar is not None:
                        state = self.data.loc[self.loc, self.statevar]
                        state = 1 if (state != state or state == 0) else 0
                        self.data.loc[self.loc, self.statevar] = state
                        print("Frame", "%5s" % str(self.frameloc+self.firstframe), "|", self.id, "| ", end='')
                        print("%10s" % self.statevar, "%1s" % str(int(state)))
                        self.draw()

                    # change frame
                    if self.key in [ord(x) for x in "qwertyu"]:
                        if self.key == ord("q"):
                            change = -self.lastframe
                        if self.key == ord("w"):
                            change = -self.fps
                        if self.key == ord("e"):
                            change = -1
                        if self.key == ord("r"):
                            change = 1
                        if self.key == ord("t"):
                            change = self.fps
                        if self.key == ord("y"):
                            change = self.lastframe
                        if self.key == ord("u"):
                            change = self.ustep
                        self.uset_frameloc(change)

                return True


    def movebar(self):

        barpos = int(cv2.getTrackbarPos('PyFrame','Frame position'))
        if barpos != self.barpos:
            self.barpos = barpos
            if self.barpos != self.trackpos:
                self.frameloc = int(self.barpos * self.stepsize)
                self.crop_firstframe = max(self.firstframe, self.frameloc+self.firstframe-self.timewindow)
                self.crop_lastframe = min(self.framediff+self.firstframe, self.frameloc+self.firstframe+self.timewindow)
                self.select_points()
                self.reset()
    

    def movetimewindow(self):

        timewindow = int(cv2.getTrackbarPos('Frames','Time window'))
        if timewindow != self.timewindow:
            self.timewindow = timewindow
            self.crop_firstframe = max(self.firstframe, self.frameloc+self.firstframe-self.timewindow)
            self.crop_lastframe = min(self.framediff+self.firstframe, self.frameloc+self.firstframe+self.timewindow)
            self.select_points()
            self.force_redraw = True
            self.reset()


    def movethreshold(self):

        threshold_speed = cv2.getTrackbarPos('Speed threshold','Speed threshold')
        if threshold_speed != self.threshold_speed:
            self.threshold_speed = threshold_speed
            self.reset()


    def movesmoothwindow(self):

        smoothwin = cv2.getTrackbarPos('Smooth window','Smooth window')
        if smoothwin != self.smoothwin:
            self.smoothwin = smoothwin
            self.reset()


    def track(self):
        
        if self.powermate:
            import threading
            self.running_flag = threading.Event()
            self.running_flag.set()  # Enable the listener
            self.pmate_key = None
            self.powermate_thread = start_powermate_listener(
                update_key_function=lambda key: setattr(self, 'pmate_key', key),
                running_flag=self.running_flag,
            )

        try:
            while True:

                cv2.imshow("Info panel", self.draw_params)
                cv2.imshow("Video", self.draw_frame)
                self.key = cv2.waitKey(1) & 0xff
                self.movebar()
                self.movetimewindow()
                self.movethreshold()
                self.movesmoothwindow()

                if self.powermate:
                    if self.pmate_key is not None:
                        self.key = self.pmate_key  # Use PowerMate input as the key
                        self.pmate_key = None  # Clear PowerMate key after processing

                if self.keypress() is False:
                    if self.powermate:
                        self.running_flag.clear()
                    break

                if self.add and self.pt is not None:
                    realpt = (int(self.pt[0] * self.multiplier), int(self.pt[1] * self.multiplier))
                    com = self.data.loc[self.loc, ["cx","cy"]]
                    if self.subcolumns[0] == "fx" and pd.notna(com[0]):
                        hvec = angle_to_vec(points_to_angle(com, realpt, flip=True))
                        fx = int(com[0] + hvec[0] * self.arrowl)
                        fy = int(com[1] + hvec[1] * self.arrowl*-1)
                        realpt = (fx,fy)
                    self.data.loc[self.loc, self.subcolumns] = realpt
                    print("[Changed:", str(self.firstframe+self.frameloc), end=" ")
                    self.add = False
                    self.force_redraw = True
                    self.draw()

        except KeyboardInterrupt:
            print("Keyboard interrupt received.")
        finally:
            if self.powermate:
                stop_powermate_listener(self.powermate_thread, self.running_flag)
                self.powermate_thread = None
            cv2.destroyAllWindows()
            cv2.waitKey(1)