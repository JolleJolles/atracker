#! /usr/bin/env python

import os
import cv2
import numpy as np

from .utils import get_screen_resolution

from pythutils.mediautils import check_media, get_vid_params, crop
from pythutils.drawutils import draw_crosshair, draw_text, mouse_events, namedcols, draw_cross, draw_hcross
from pythutils.mathutils import midpoint, ptsToDist, uneven
from pythutils.sysutils import objectview

from .imgprocessor import ImgProcessor
from .utils import *

class ivideo:

    def __init__(self, source, firstframe=None, lastframe=None,
                 showFramenr=True, showCrosshair=True, showHelp=False,
                 showCross=False, drawLine=False, drawRect=False,
                 drawPoly=False, drawCircle=False, drawEllipse=False, drawPt=False,
                 treshtype=None, treshinfo=None, fullrange=False,
                 showHelperlines=False, displaysize=1, orientation=True,
                 orient_minconvex=0.4, objboxsize=200, simple=True, roi=None):

        """Opens a video with dynamic user interaction interface"""

        self.source = source
        if type(self.source) == str:
            check_media(self.source, internal=True)

        self.stream = True if type(self.source) == int else False
        self.showFramenr = False if self.stream else showFramenr
        self.showCrosshair = showCrosshair
        self.showHelp = showHelp
        self.showHelperlines = showHelperlines
        self.showCross = showCross
        self.showFullScreen = False
        self.showMask = False
        self.inverted = False

        self.drawLine = drawLine
        self.drawRect = drawRect
        self.drawPoly = drawPoly
        self.drawCircle = drawCircle
        self.drawEllipse = drawEllipse
        self.drawPt = drawPt
        self.treshtype = treshtype
        self.treshinfo = treshinfo

        self.orientation = orientation
        self.orient_minconvex = orient_minconvex

        self.col_allcons = namedcols("red")
        self.col_conssub = namedcols("blue")
        self.col_centre = namedcols("white")
        self.lwith_centre = 15
        self.objboxsize = objboxsize
        self.simple = simple
        self.roi = roi

        self.cap = cv2.VideoCapture(source)
        self.fps, self.vidw, self.vidh, _ = get_vid_params(self.cap)
        max_pyframe = find_max_working_pyframe(self.cap)
        self.fcount = max_pyframe+1

        self.firstframe = firstframe if firstframe is not None else 1
        self.lastframe = lastframe if lastframe is not None else self.fcount
        self.missingframes = []

        self.coords = {}

        self.frameloc = self.firstframe - 1
        self.listening = False
        self.fullrange = fullrange
        self.displaysize = displaysize

        self.tempcol = namedcols("orange")
        self.col = namedcols("red")
        self.maskcol = namedcols("black")

        if self.treshtype != None:
            if self.treshinfo == None:
                ti = []
            elif treshtype not in self.treshinfo:
                ti = []
            else:
                ti = self.treshinfo[treshtype]
            self.d_blur = 1 if ti == [] else ti["blur"]
            self.d_min_area = 100 if ti == [] else ti["min_area"]
            self.d_max_area = 1000 if ti == [] else ti["max_area"]
            if self.treshtype.startswith("bw"):
                self.d_blur2 = 1 if ti == [] or "blur2" not in ti else ti["blur2"]
                self.d_erosion = 1 if ti == [] else ti["erosion"]
                self.d_treshold = 60 if ti == [] else ti["treshold"]
            else:
                colmin = (0,0,0) if ti == [] else literal_eval(ti["colmin"])
                colmax = (179,255,255) if ti == [] else literal_eval(ti["colmax"])
                self.d_hue_lo, self.d_sat_lo, self.d_val_lo = colmin
                self.d_hue_hi, self.d_sat_hi, self.d_val_hi = colmax

        self.create_gui()
        self.defaults(mask=True)
        self.keys()


    def defaults(self, mask=False):

        self.start_frame = None
        if self.roi is not None:
            self.vidw  = self.roi[1][0] - self.roi[0][0]
            self.vidh =  self.roi[1][1] - self.roi[0][1]
        self.stop_frame = None
        self.m = mouse_events()
        cv2.setMouseCallback(self.source, self.m.draw)
        if mask:
            self.mask = np.zeros((self.vidh,self.vidw,3), np.uint8)+255
            self.maskbak = self.mask.copy()


    def create_gui(self):

        def nothing(x):
            pass

        self.winwidth = 400
        self.screen = get_screen_resolution()

        cv2.namedWindow("Frame position", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Frame position", self.winwidth, 0)
        cv2.moveWindow("Frame position", 0, 0)
        self.stepsize = 1
        self.barpos = 0
        self.trackpos = 0
        cv2.createTrackbar("PyFrame", "Frame position", self.barpos, self.lastframe-self.firstframe, nothing)

        cv2.namedWindow(self.source, cv2.WINDOW_NORMAL)
        vidw = int(min(self.screen[0]-self.winwidth, self.displaysize * self.vidw))
        vidh = int(self.vidh*(vidw/self.vidw)+50)
        cv2.resizeWindow(self.source, vidw, vidh)
        cv2.moveWindow(self.source, self.winwidth, 0)
        self.m = mouse_events()
        cv2.setMouseCallback(self.source, self.m.draw)
        #self.toggle_fullscreen(False)
        self.reset()
        cv2.imshow(self.source, self.drawFrame)

        if self.treshtype is not None:
            movedown = 0
            if not self.treshtype.startswith("bw"):
                cv2.namedWindow("Hue", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Hue", self.winwidth, 70)
                cv2.moveWindow("Hue", 0, 137)
                self.hue_panel = create_hue_gradient(10)
                movedown = 48
            cv2.namedWindow("Tresholding", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Tresholding", self.winwidth, 0)
            cv2.moveWindow("Tresholding", 0, movedown + 157)
            cv2.createTrackbar("Blur", "Tresholding", self.d_blur, 40, nothing)
            cv2.createTrackbar("Min area", "Tresholding", self.d_min_area, self.vidw*3, nothing)
            cv2.createTrackbar("Max area", "Tresholding", self.d_max_area, self.vidw*3, nothing)
            if self.treshtype.startswith("bw"):
                cv2.createTrackbar("Erode", "Tresholding", self.d_erosion , 40, nothing)
                cv2.createTrackbar("Blur2", "Tresholding", self.d_blur2, 40, nothing)
                cv2.createTrackbar("Treshold", "Tresholding", self.d_treshold, 255, nothing)
            else:
                cv2.createTrackbar("Hue low", "Tresholding", self.d_hue_lo, 179, nothing)
                cv2.createTrackbar("Hue high", "Tresholding", self.d_hue_hi, 179, nothing)
                cv2.createTrackbar("Sat low", "Tresholding", self.d_sat_lo, 255, nothing)
                cv2.createTrackbar("Sat high", "Tresholding", self.d_sat_hi, 255, nothing)
                cv2.createTrackbar("Val low", "Tresholding", self.d_val_lo, 255, nothing)
                cv2.createTrackbar("Val high", "Tresholding", self.d_val_hi, 255, nothing)
                movedown = movedown + 104

            cv2.namedWindow("Tresholded", cv2.WINDOW_NORMAL)
            winheight = int(self.winwidth/self.vidw*self.vidh)+20
            cv2.resizeWindow("Tresholded", self.winwidth, winheight)
            cv2.moveWindow("Tresholded", 0, movedown + 391)

            self.infoheight = 10
            self.infopanel = np.zeros((10, self.winwidth), np.uint8)+255
            cv2.namedWindow("Contour info", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Contour info", self.winwidth, self.infoheight)
            cv2.moveWindow("Contour info", 0, movedown + winheight + 420)


    def keys(self, save="s", quit=27, help="h", fullscreen="f", goto="g",
             firstframe="q", backsec="w", backframe="e", forwardframe="r",
             forwardsec="t", lastframe="y", cross="x", mask="m", setstart="[",
             setstop="]", line="l", rectangle="k", polygon="p", erase="d",
             add = "a", invert="i", circle="c", ellipse="v", helperlines="z",
             getcoords = "o", simple="u"):


        """
        User keys for interacting with a video

        general
        ------------
        save: default="s",
            Saves the current data, as initiated, and closes video
        quit: default=27 (ESC key)
            Closes the video without saving any data
        help: default="h"
            Toggles the help panel for the keypresses
        fullscreen: default="f"
            Toggles fullscreen
        goto: default="g"
            Enables the user to go to a specific frame in the video, which can
            be typed in using the numeric keys and confirmed with ENTER or
            cancelled with ESC or "g" key.
        cross: default="x"
            Toggles a horizontal and vertical cross to help with camera
            positioning
        helperlines: default="z"
            Toggles a horizontal and vertical cross at the mouse pointer to help
            with positioning of shapes
        mask: default="m"
            Toggles mask window
        invert: default="i"
            Inverts mask
        simple: default="u"
            Toggles drawing of simple and complex contour information

        config
        ------------
        setstart: default="["
            Stores the frame as frame to start video
        setstop: default="]"
            Stores the frame as frame to stop video

        functions
        ------------
        line: default="l"
            Draws a line between two points
        rectangle: default="k"
            Draws a rectangle between two points
        polygon: default="p"
            Draws a multi-point polygon
        circle: default="c"
            Draws a circle between two points
        ellipse: default="v"
            Draws an ellipse within rectangle encompassing two points
        erase: default="d"
            Erases the points
        add: default="a"
            Creates a new list of points
        getcoords: default="o"
            Get the simple coordiates from the mask contour

        videocontrol
        ------------
        firstframe: default="q"
            Go to the first frame
        backsec: default="w"
            Go one second back in time
        backframe: default="e"
            Go one frame back in time
        forwardframe: default="r"
            Go one frame forward in time
        forwardsec: default="t"
            Go one sec forward in time
        lastframe: default="y"
            Go to the last frame
        """

        def _ASCII(value):
            return value if type(value)==int else ord(value)

        self.ks = {"nothing":255,
                  "save":_ASCII(save),
                  "quit":_ASCII(quit),
                   "help": _ASCII(help),
                   "fullscreen": _ASCII(fullscreen),
                   "goto": _ASCII(goto),
                   "cross": _ASCII(cross),
                   "helperlines": _ASCII(helperlines),
                   "mask": _ASCII(mask),
                   "invert": _ASCII(invert),
                   "simple": _ASCII(simple),
                   "setstart": _ASCII(setstart),
                   "setstop": _ASCII(setstop),
                   "line": _ASCII(line),
                   "rectangle": _ASCII(rectangle),
                   "polygon": _ASCII(polygon),
                   "circle": _ASCII(circle),
                   "ellipse": _ASCII(ellipse),
                   "erase": _ASCII(erase),
                   "add": _ASCII(add),
                   "getcoords": _ASCII(getcoords),
                   "firstframe": _ASCII(firstframe),
                   "backsec": _ASCII(backsec),
                   "backframe": _ASCII(backframe),
                   "forwardframe": _ASCII(forwardframe),
                   "forwardsec": _ASCII(forwardsec),
                   "lastframe": _ASCII(lastframe)
                   }

        self.k = objectview(self.ks)

        dl = self.keys.__doc__.split("\n")
        self.keyinfo = {key.split(":")[0].split(" ")[-1]:dl[i+1].split("  ")[-1]
                        for i,key in enumerate(dl) if ":" in key}


    def key_event(self):

        """
        Enables the user to use simple keypresses to interact with a video.
        Keypresses can be set with the "keys" function.
        """

        if self.key == self.k.nothing:
            return None

        else:

            if not self.listening and self.key in (self.k.save, self.k.quit):

                if self.key == self.k.save:
                    print("data stored..")

                if self.key == self.k.quit:
                    self.defaults()
                    print("User exited.. data discarded..")

                return False

            else:

                if self.key == self.k.help:
                    self.showHelp = not self.showHelp
                    self.reset()

                if self.key == self.k.fullscreen:
                    self.showFullScreen = not self.showFullScreen
                    self.toggle_fullscreen()

                if not self.stream and self.key == self.k.goto:
                    self.listening = not self.listening
                    self.goframe = ""
                    self.reset()
                if self.listening:
                    if self.key in range(48,57+1): #numeric keys
                        self.goframe += chr(self.key)
                    if self.key == 13: #enter key
                        self.uset_frameloc(int(self.goframe)-1-self.frameloc)
                    if self.key in (13,27): #enter or escape key
                        self.goframe = ""
                        self.listening = False
                    if self.key != self.k.goto:
                        self.reset()

                if self.key == self.k.mask:
                    self.showMask = not self.showMask
                    self.reset()

                if self.key == self.k.cross:
                    self.showCross = not self.showCross
                    self.reset()

                if self.key == self.k.helperlines:
                    self.showHelperlines = not self.showHelperlines
                    self.reset()

                if not self.stream and self.key == self.k.setstart:
                    print("Start frame set at", self.frameloc+1, end=".. ")
                    self.start_frame = self.frameloc+1

                if not self.stream and self.key == self.k.setstop:
                    print("Stop frame set at", self.frameloc+1, end=".. ")
                    self.stop_frame = self.frameloc+1

                if self.key == self.k.line:
                    self.drawLine = not self.drawLine
                    self.defaults()
                    if self.drawLine:
                        self.drawRect = False
                        self.drawPoly = False
                        self.drawCircle = False
                        self.drawEllipse = False
                    self.reset()

                if self.key == self.k.rectangle:
                    self.drawRect = not self.drawRect
                    self.defaults()
                    if self.drawRect:
                        self.drawLine = False
                        self.drawPoly = False
                        self.drawCircle = False
                        self.drawEllipse = False
                    self.reset()

                if self.key == self.k.polygon:
                    self.drawPoly = not self.drawPoly
                    self.defaults()
                    if self.drawPoly:
                        self.drawRect = False
                        self.drawLine = False
                        self.drawCircle = False
                        self.drawEllipse = False
                    self.reset()

                if self.key == self.k.circle:
                    self.drawCircle = not self.drawCircle
                    self.defaults()
                    if self.drawCircle:
                        self.drawRect = False
                        self.drawLine = False
                        self.drawPoly = False
                        self.drawEllipse = False
                    self.reset()

                if self.key == self.k.ellipse:
                    self.drawEllipse = not self.drawEllipse
                    self.defaults()
                    if self.drawEllipse:
                        self.drawCircle = False
                        self.drawRect = False
                        self.drawLine = False
                        self.drawPoly = False
                    self.reset()

                if self.key == self.k.invert:
                    self.mask = np.invert(self.mask)
                    self.inverted = not self.inverted
                    self.reset()

                if self.key == self.k.simple:
                    self.simple = not self.simple
                    self.reset()

                if self.key == self.k.erase:
                    self.defaults()
                    self.reset()

                if self.key == self.k.add:
                    if self.drawRect and self.m.twoPoint is not None:
                        cv2.rectangle(self.mask, self.m.twoPoint[0],
                                      self.m.twoPoint[1], self.maskcol, -1)
                    if self.drawPoly and len(self.m.pts)>0:
                        cv2.fillPoly(self.mask, np.array([self.m.pts]),
                                     self.maskcol)
                    if self.drawCircle and self.m.twoPoint is not None:
                        cv2.circle(self.mask, self.cloc, self.cr, self.maskcol, -1)
                    if self.drawEllipse and self.m.twoPoint is not None:
                        cv2.ellipse(self.mask,self.cloc,self.lenaxes,0,0,360,self.maskcol,-1)
                    self.defaults()
                    self.reset()

                if self.key == self.k.getcoords:
                    _,self.maskcoords = coordsfrommask(self.mask)
                    self.coords[len(self.coords)+1] = self.maskcoords
                    self.mask = self.maskbak.copy()

                if not self.stream and self.key == self.k.firstframe:
                    change = -self.lastframe
                    self.uset_frameloc(change)

                if not self.stream and self.key == self.k.backsec:
                    change = -self.fps
                    self.uset_frameloc(change)

                if not self.stream and self.key == self.k.backframe:
                    change = -1
                    self.uset_frameloc(change)

                if not self.stream and self.key == self.k.forwardframe:
                    change = 1
                    self.uset_frameloc(change)

                if not self.stream and self.key == self.k.forwardsec:
                    change = self.fps
                    self.uset_frameloc(change)

                if not self.stream and self.key == self.k.lastframe:
                    change = self.fcount-1 if self.fullrange else self.lastframe
                    self.uset_frameloc(change)

                return True


    def movebar(self):

        barpos = int(cv2.getTrackbarPos("PyFrame","Frame position"))
        if barpos != self.barpos:
            self.barpos = barpos
            if self.barpos != self.trackpos:
                if self.fullrange:
                    self.frameloc = int(self.barpos*self.stepsize)
                else:
                    self.frameloc = int(self.firstframe+(self.barpos*self.stepsize))
                self.reset()
        if self.stream:
            self.reset()
        if self.m.drawing:
            self.m.drawing = False
            self.reset()


    def uset_frameloc(self, change):

        self.frameloc = self.frameloc+change
        if self.frameloc >= self.lastframe-self.firstframe:
            self.frameloc = self.lastframe-self.firstframe
        if self.frameloc <= self.firstframe-self.firstframe:
            self.frameloc = self.firstframe-self.firstframe
        trackpos = int(self.frameloc/self.stepsize)
        self.trackpos = 0 if trackpos < 0 else trackpos
        cv2.setTrackbarPos("PyFrame","Frame position", self.trackpos)
        self.reset()


    def toggle_fullscreen(self, value=None):

        if value != None:
            self.showFullScreen = value

        if self.showFullScreen:
            cv2.namedWindow("Full", cv2.WND_PROP_FULLSCREEN)
            cv2.imshow("Full", self.drawFrame)
            cv2.setWindowProperty("Full", cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
        else:
            cv2.destroyWindow("Full")
            cv2.setWindowProperty(self.source, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN)
            cv2.setWindowProperty(self.source, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_NORMAL)
        self.reset()


    def exit(self):

        cv2.destroyWindow("Frame position")
        cv2.destroyWindow(self.source)
        try:
            cv2.destroyWindow("Hue")
            cv2.destroyWindow("Tresholded")
            cv2.destroyWindow("Tresholding")
            cv2.destroyWindow("Contour info")
            for i in range(0,50):
                cv2.destroyWindow(str(i))
        except Exception:
            pass
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        for i in range(15):
            cv2.waitKey(1)


    def reset(self):

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frameloc)
        frameOK, self.frame = self.cap.read()
        if self.roi is not None:
            self.frame = crop(self.frame, self.roi[0], self.roi[1])
        if not frameOK:
            if self.frameloc < self.lastframe-(self.fps*5):
                self.missingframes.append(self.frameloc+1)
                self.frameloc += 1
            if self.frameloc > self.lastframe-(self.fps*5):
                self.frameloc -= 1
            self.reset()
        if len(self.missingframes)>0:
            missing = str(min(self.missingframes))+":"+str(max(self.missingframes))
            print("Frames "+missing+" could not be read and were skipped..")
            self.missingframes = []
        self.pt = None
        self.draw()


    def draw(self):

        self.drawFrame = self.frame.copy()

        if self.showMask:
            cv2.addWeighted(self.mask, 0.7, self.drawFrame, 0.3, 0, self.drawFrame)

        if self.showCrosshair and not self.showFullScreen:
            ll = 3000 if self.showHelperlines else 10
            draw_crosshair(self.drawFrame, self.m.pos, ll)

        if self.drawRect:
            start, end, color = None, None, None

            if self.m.posDown is not None and not self.showFullScreen:
                start, end, color = self.m.posDown, self.m.pos, self.tempcol
            elif self.m.posUp is not None:
                start, end, color = self.m.pts[-1], self.m.posUp, self.col

            if start and end:
                area = abs((end[0] - start[0]) * (end[1] - start[1]))
                cv2.rectangle(self.drawFrame, start, end, color, 2)
                draw_text(self.drawFrame, f"a: {round(area)} px", loc=(0, 30), size=1, margin=5, bgcol="white")


        if self.drawLine:
            if not self.showFullScreen and self.m.posDown is not None:
                cv2.line(self.drawFrame, self.m.posDown, self.m.pos,
                         self.tempcol, 2)
            if self.m.posUp is not None:
                cv2.line(self.drawFrame, self.m.pts[-1], self.m.posUp,
                         self.col, 2)

        if self.drawPoly:
            if not self.showFullScreen and len(self.m.pts)>0:
                cv2.line(self.drawFrame, self.m.pts[-1], self.m.pos,
                         self.tempcol, 2)
            if len(self.m.pts)>1:
                cv2.polylines(self.drawFrame, np.array([self.m.pts]), False,
                              self.col, 2)

        if self.drawCircle:
            if not self.showFullScreen and self.m.posDown is not None:
                loc = midpoint(self.m.pos, self.m.posDown)
                radius = ptsToDist(self.m.pos , self.m.posDown)/2
                cv2.circle(self.drawFrame, loc, int(radius), self.tempcol, 2)
                draw_text(self.drawFrame, f"r: {round(radius)} px", loc=(0, 30), size=1, margin=5, bgcol="white")
            if self.m.posUp is not None:
                self.cloc = midpoint(self.m.twoPoint[0],self.m.twoPoint[1])
                self.cr = int(ptsToDist(self.m.twoPoint[0],self.m.twoPoint[1])/2)
                cv2.circle(self.drawFrame, self.cloc, self.cr, self.col, 2)
                draw_text(self.drawFrame, f"r: {round(self.cr)} px", loc=(0, 30), size=1, margin=5, bgcol="white")

        if self.drawEllipse:
            if not self.showFullScreen and self.m.posDown is not None:
                cloc = midpoint(self.m.pos, self.m.posDown)
                lenaxes = (int(abs((self.m.posDown[0]-self.m.pos[0]))/2),int(abs((self.m.posDown[1]-self.m.pos[1]))/2))
                cv2.circle(self.drawFrame, cloc, 2, self.col, 2)
                cv2.ellipse(self.drawFrame,cloc,lenaxes,0,0,360,self.tempcol,2)
            if self.m.posUp is not None:
                self.cloc = midpoint(self.m.twoPoint[0],self.m.twoPoint[1])
                self.lenaxes = (int(abs((self.m.twoPoint[1][0]-self.m.twoPoint[0][0]))/2),int(abs((self.m.twoPoint[1][1]-self.m.twoPoint[0][1]))/2))
                cv2.ellipse(self.drawFrame,self.cloc,self.lenaxes,0,0,360,self.col,2)
        
        if self.drawPt:
            if len(self.m.pts)>0 and not any([self.drawEllipse, self.drawCircle, self.drawPoly, self.drawPoly, self.drawRect]):
                for pt in self.m.pts:
                    cv2.circle(self.drawFrame, pt, 6, namedcols("black"), -1)
                    cv2.circle(self.drawFrame, pt, 4, namedcols("pink"), -1)

        if self.showCross:
            draw_cross(self.drawFrame)
            draw_hcross(self.drawFrame)

        if self.showFramenr:
            draw_text(self.drawFrame, str(self.frameloc+1), (0,0), 0.8, margin=5,
                      bgcol="white")

        if self.listening:
            txt = "Go to frame: "+self.goframe
            draw_text(self.drawFrame, txt, (10,80), 1, "lightgreen",
                      thickness=2, shadow=True)

        if self.showHelp:
            vpos = 0
            for par,key in self.ks.items():
                if par in self.keyinfo:
                    vpos += 25
                    key = chr(key) if len(repr(chr(key)))<=3 else str(key)
                    helptxt = par+": "+key+", "+self.keyinfo[par]
                    draw_text(self.drawFrame, helptxt, (10, vpos), 0.75, "orange",
                              thickness=1, shadow=True)


    def show(self):

        while True:

            if self.treshtype is not None:
                self.reset()
                win = "Tresholding"
                t_blur = uneven(int(cv2.getTrackbarPos("Blur", win)))
                t_minarea = int(cv2.getTrackbarPos("Min area", win))
                t_maxarea = int(cv2.getTrackbarPos("Max area", win))

                if self.treshtype.startswith("bw"):
                    t_tresh = int(cv2.getTrackbarPos("Treshold", win))
                    t_erode = uneven(int(cv2.getTrackbarPos("Erode", win)))
                    t_blur2 = uneven(int(cv2.getTrackbarPos("Blur2", win)))

                    IP = ImgProcessor(self.frame, self.img_bg, self.img_mask, self.treshtype,
                        t_blur, t_erode, t_blur2, t_tresh, t_minarea, t_maxarea, simple=self.simple)
                    self.img_tresh, allcons, conlist = IP.process()
                    self.treshinfo = {"min_area": t_minarea, "max_area": t_maxarea,
                                    "erosion": t_erode, "treshold" : t_tresh, "blur": t_blur,
                                    "blur2": t_blur2}

                else:
                    hue_lo = cv2.getTrackbarPos("Hue low", win)
                    hue_hi = cv2.getTrackbarPos("Hue high", win)
                    sat_lo = cv2.getTrackbarPos("Sat low", win)
                    sat_hi = cv2.getTrackbarPos("Sat high", win)
                    val_lo = cv2.getTrackbarPos("Val low", win)
                    val_hi = cv2.getTrackbarPos("Val high", win)

                    # show hue panel 
                    display_panel = self.hue_panel.copy()
                    display_panel[:, :hue_lo] = display_panel[:, :hue_lo] // 3  # Darken the left region
                    display_panel[:, hue_hi:] = display_panel[:, hue_hi:] // 3  # Darken the right region
                    cv2.line(display_panel, (hue_lo, 0), (hue_lo, display_panel.shape[0] - 1), (0, 0, 0), 1)
                    cv2.line(display_panel, (hue_hi, 0), (hue_hi, display_panel.shape[0] - 1), (0, 0, 0), 1)
                    cv2.imshow("Hue", display_panel)

                    colmin = (hue_lo, sat_lo, val_lo)
                    colmax = (hue_hi, sat_hi, val_hi)
                    colBGR = namedcols(self.treshtype)

                    IP = ImgProcessor(self.frame, self.img_bg, self.img_mask, self.treshtype,
                        blur=t_blur, min_area=t_minarea, max_area=t_maxarea,
                        colmin=colmin, colmax=colmax, simple=self.simple)
                    self.img_tresh, allcons, conlist = IP.process()
                    self.treshinfo = {"min_area": t_minarea, "max_area": t_maxarea,
                                    "blur": t_blur, "colmin" : str(colmin),
                                    "colmax" : str(colmax), "colBGR" : str(colBGR)}

                # Draw all contours
                cv2.drawContours(self.drawFrame, allcons, -1, self.col_allcons, 1)

                # Draw contours within size range
                _,cons = dic_exclnan(conlist,"contour","com")
                cv2.drawContours(self.drawFrame, cons, -1, self.col_conssub, 1)

                # Draw more complex information
                inds,_ = dic_exclnan(conlist,"fed")
                for i in inds:
                    arrowtip = get_coord(conlist["head"][i][0], conlist["head"][i][1], conlist["angle"][i], 13, astuple=True)
                    cv2.polylines(self.drawFrame, np.array([conlist["skeleton"][i]]), False, namedcols("orange"), 1)
                    cv2.arrowedLine(self.drawFrame, conlist["head"][i], arrowtip, namedcols("white"), 1, tipLength = 0.4)
                    cv2.circle(self.drawFrame, conlist["head"][i], 0, namedcols("lightgreen"), 6)
                    cv2.circle(self.drawFrame, conlist["tail"][i], 0, namedcols("red"), 6)
                    cv2.circle(self.drawFrame, conlist["fed"][i], 0, namedcols("yellow"), 6)
                    cv2.circle(self.drawFrame, conlist["icom"][i], 0, namedcols("black"), 6)

                # Draw object coms and id's
                inds,coms = dic_exclnan(conlist,"com")
                self.infoheight = min((len(coms)*2*7+20), 400)
                self.infopanel = np.zeros((self.infoheight, self.winwidth), np.uint8)+255
                for i,com in enumerate(coms):
                    #cv2.circle(self.drawFrame, com, 0, self.col_centre, self.lwith_centre)
                    draw_text(self.drawFrame, str(i+1), (com[0]-15,com[1]-15), 0.3, "black", 0, 1)
                    cv2.putText(self.infopanel, str(i+1)+" Area: " + str(conlist["area"][i]), (5, 12+((i+i)*8)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, namedcols("black"), 1, cv2.LINE_AA)
                    if not self.simple:
                        cv2.putText(self.infopanel, "Length: " + str(conlist["length"][i]), (200, 12+((i+i)*8)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, namedcols("black"), 1, cv2.LINE_AA)

                # Draw object boxes
                win=0
                for ind in inds:
                    if(conlist["objbox"][ind]==conlist["objbox"][ind]):
                        objbox = crop(self.drawFrame, conlist["objbox"][ind])
                        win += 1
                        cv2.namedWindow(str(ind), cv2.WINDOW_NORMAL)
                        cv2.imshow(str(ind), objbox)
                        cv2.resizeWindow(str(ind), self.objboxsize, self.objboxsize)
                        cv2.moveWindow(str(ind), self.objboxsize*win, self.screen[1]-self.objboxsize)

                cv2.imshow("Tresholded", self.img_tresh)
                cv2.imshow("Contour info", self.infopanel)
                cv2.resizeWindow("Contour info", self.winwidth, self.infoheight+50)

            w = int(self.displaysize * self.vidw)
            h = int(self.displaysize * self.vidh)
            cv2.imshow(self.source, cv2.resize(self.drawFrame, (w, h)))
            self.key = cv2.waitKey(1) & 0xff
            self.movebar()

            if self.key_event() is False:
                break

        self.exit()
