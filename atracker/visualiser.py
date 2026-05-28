#! /usr/bin/env python

import os
import cv2
import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from pythutils.sysutils import lineprint
from pythutils.mediautils import crop, videowriter, add_transimg, imgresize
from pythutils.drawutils import namedcols, draw_text, draw_traj, uniqcols
from pythutils.mathutils import points_to_angle

from .media import framechecks
from .geometry import get_coord
from .contour_utils import draw_coordlist


def addcanvas(img, dims, color):

    # Create background canvas
    bgcanvas = np.zeros((dims[1], dims[0], 3), dtype="uint8")
    bgcanvas = bgcanvas + [color]

    # Get ratio canvas width and height to that of image
    wratio = dims[0]/float(img.shape[1])
    hratio = dims[1]/float(img.shape[0])

    # Now resize image to be able to fit in canvas with maximum possible size
    resize = min(wratio, hratio)
    img = imgresize(img, resize)

    # Place image in center of canvas if ratios are not identical
    if wratio == hratio:
        bgcanvas = img

    # Image occupies 100% of width of canvas, thus change yspace
    elif wratio < hratio:
        extra = int((bgcanvas.shape[0] - img.shape[0]) / 2)
        bgcanvas[extra:extra+img.shape[0],:] = img

    # Image occupies 100% of height of canvas, thus change wspace
    elif wratio>hratio:
        extra = int((bgcanvas.shape[1] - img.shape[1]) / 2)
        bgcanvas[:,extra:extra+img.shape[1]] = img

    bgcanvas = bgcanvas.astype(np.uint8)

    return bgcanvas


_ZONE_COLORS = [
    (0, 210, 255),   # orange
    (255, 255, 0),   # cyan
    (0, 220, 80),    # lime green
    (200, 0, 230),   # pink/magenta
    (255, 180, 0),   # teal/azure
    (50, 50, 255),   # red
]


class TrackVisualiser:
    """Drawing helper for the real-time tracking loop (tracksingle)."""

    def __init__(self, config, thresh_types, objects, threshcolors, orientfrombw,
                 mask_contours=None, wall_contours=None, zone_coords=None):
        self.config = config
        self.thresh_types = thresh_types
        self.objects = objects
        self.threshcolors = threshcolors
        self.orientfrombw = orientfrombw

        # Pre-compute colours once per video
        self.col_red       = namedcols("red")
        self.col_lightgreen = namedcols("lightgreen")
        self.col_orange    = namedcols("orange")
        self.col_contour   = eval(config.vis.contour_col)
        self.col_centre    = eval(config.vis.centre_col)
        self.col_orient    = eval(config.vis.orient_col)
        self.col_traj      = eval(config.vis.traj_col)
        self.thresh_cols   = {t: namedcols(t) for t in thresh_types if not t.startswith("bw")}
        self.cols          = uniqcols(max(1, objects))
        self.mask_contours = mask_contours
        self.wall_contours = wall_contours
        self.zone_coords   = zone_coords

    def draw_thresh_pass(self, img_draw, img, img_thresh, conlist, allcons,
                         ids, filtered_coms, traj_history, tracked_ids, thresh_type):
        """
        Draw one threshold-type pass onto img_draw.
        conlist must already be subsetted to the ID'ed contours (same length as ids).
        """
        # Trajectories
        draw_ids = sorted(tracked_ids) if thresh_type.startswith("bw") else [thresh_type]
        for i, id in enumerate(draw_ids):
            trajdat = list(reversed(traj_history.get(id, [])))
            if len(trajdat) >= 2:
                col = (self.cols[(id - 1) % len(self.cols)] if self.config.vis.idcol else self.col_traj
                       if thresh_type.startswith("bw") else self.thresh_cols.get(thresh_type, self.col_traj))
                draw_traj(img_draw, trajdat, col,
                          self.config.vis.traj_minthick,
                          self.config.vis.traj_maxthick,
                          self.config.vis.traj_opacity)

        if self.config.vis.trajs_below:
            img_draw[img_thresh == 255] = img[img_thresh == 255]

        # All detected contours (thin red)
        cv2.drawContours(img_draw, allcons, -1, self.col_red, 1)

        # ID'ed contours
        is_merged = (self.objects > 1 and thresh_type.startswith("bw")
                     and conlist.get("consmerged"))
        contour_col = (128 if not thresh_type.startswith("bw")
                       else self.col_contour if not is_merged else 128)
        cv2.drawContours(img_draw, conlist["contour"], -1, contour_col, 1)

        # Per-object details: skeleton, tail, head/arrow, centroid, ID text
        for i, id in enumerate(conlist["id"]):
            if conlist["skeleton"][i] == conlist["skeleton"][i]:
                img_draw = draw_coordlist(img_draw, conlist["skeleton"][i], self.col_orange)
            if conlist["tail"][i] == conlist["tail"][i]:
                cv2.circle(img_draw, conlist["tail"][i], 0, self.col_red, 6)
            if conlist["head"][i] == conlist["head"][i]:
                arrowtip = get_coord(conlist["head"][i][0], conlist["head"][i][1],
                                     conlist["angle"][i], 13, True)
                cv2.arrowedLine(img_draw, conlist["head"][i], arrowtip,
                                self.col_orient, 1, tipLength=0.4)
                cv2.circle(img_draw, conlist["head"][i], 0, self.col_lightgreen, 6)
            if conlist["com"][i] == conlist["com"][i]:
                idcol = (self.thresh_cols.get(thresh_type, self.col_centre)
                         if not thresh_type.startswith("bw") else self.col_centre)
                cv2.circle(img_draw, conlist["com"][i], 0, idcol,
                           self.config.vis.centre_lwidth)
                if thresh_type.startswith("bw"):
                    draw_text(img_draw, str(id),
                              (conlist["com"][i][0] - 4, conlist["com"][i][1] - 4),
                              0.3, "black", 0, 1)

    def draw_orient_link(self, img_draw, fulldat, frame_nr):
        """
        Compute and draw orientation arrows that link colour contours to their
        paired bw contours. Also writes computed angles back into fulldat["angle"].
        """
        currframe_inds = [i for i, f in enumerate(fulldat["frame"]) if f == frame_nr]
        ids = [id for i, id in enumerate(fulldat["id"]) if i in currframe_inds]

        if "angle" not in fulldat:
            fulldat["angle"] = [None] * len(fulldat["frame"])

        if not ids or len(ids) != 2 * len([i for i in ids if type(i) == str]):
            return

        coms = [(fulldat["cx"][i], fulldat["cy"][i]) for i in currframe_inds]
        colids, colcoms = zip(*[(ids[i], com) for i, com in enumerate(coms)
                                if type(ids[i]) == str])
        bwids, bwcoms = zip(*[(ids[i], com) for i, com in enumerate(coms)
                               if type(ids[i]) != str])
        _, bw_inds = linear_sum_assignment(cdist(colcoms, bwcoms))

        for i, id in enumerate(colids):
            angle = int(points_to_angle(colcoms[i], bwcoms[bw_inds[i]], flip=True))
            target_index = currframe_inds[ids.index(id)]
            if target_index >= len(fulldat["angle"]):
                fulldat["angle"].extend(
                    [None] * (target_index - len(fulldat["angle"]) + 1))
            fulldat["angle"][target_index] = angle
            arrowtip = get_coord(colcoms[i][0], colcoms[i][1], angle, 5, True)
            col = (255, 255, 255) if id in ["blue", "black"] else (0, 0, 0)
            cv2.arrowedLine(img_draw, colcoms[i], arrowtip, col,
                            self.config.vis.orient_lwidth, tipLength=0.4)

    def draw_scene_overlays(self, img_draw):
        """Draw wall contours, zone polygons with labels, and mask border on img_draw."""
        if self.wall_contours is not None:
            overlay = img_draw.copy()
            cv2.drawContours(overlay, self.wall_contours, -1, (160, 50, 160), -1)
            cv2.addWeighted(overlay, 0.35, img_draw, 0.65, 0, img_draw)
            cv2.drawContours(img_draw, self.wall_contours, -1, (110, 30, 110), 2)

        if self.zone_coords is not None:
            overlay = img_draw.copy()
            for zone_idx, coords in self.zone_coords.items():
                contour = np.array([[[x, y]] for x, y in coords], dtype=np.int32)
                col = _ZONE_COLORS[(zone_idx - 1) % len(_ZONE_COLORS)]
                cv2.drawContours(overlay, [contour], -1, col, -1)
            cv2.addWeighted(overlay, 0.18, img_draw, 0.82, 0, img_draw)
            font = cv2.FONT_HERSHEY_SIMPLEX
            for zone_idx, coords in self.zone_coords.items():
                contour = np.array([[[x, y]] for x, y in coords], dtype=np.int32)
                col = _ZONE_COLORS[(zone_idx - 1) % len(_ZONE_COLORS)]
                cv2.drawContours(img_draw, [contour], -1, col, 2)
                cx = int(np.mean([x for x, y in coords]))
                cy = int(np.mean([y for x, y in coords]))
                label = f"Z{zone_idx}"
                cv2.putText(img_draw, label, (cx - 9, cy + 5), font, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(img_draw, label, (cx - 9, cy + 5), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        if self.mask_contours is not None:
            cv2.drawContours(img_draw, self.mask_contours, -1, (180, 180, 180), 1)

    def _line_colour(self, line):
        """Dark BGR colour for an info-box line, colour-coded by object ID."""
        if line.startswith("frame "):
            return (50, 50, 50)
        if line.startswith("ID"):
            id_str = line[2:line.index(":")] if ":" in line else line[2:]
            try:
                id_int = int(id_str)
                base = self.cols[(id_int - 1) % len(self.cols)]
                return tuple(max(0, int(v * 0.55)) for v in base)
            except ValueError:
                try:
                    base = namedcols(id_str)
                    return tuple(max(0, int(v * 0.55)) for v in base)
                except Exception:
                    pass
        return (80, 80, 80)

    def draw_info_overlay(self, img_draw, img_mask, frame_nr, frame_info):
        """Draw mask overlay and a semi-transparent info box with per-ID colour coding."""
        if img_mask is not None:
            img_masked = cv2.bitwise_and(img_draw, img_draw, mask=img_mask)
            cv2.addWeighted(img_masked, self.config.vis.mask_opacity,
                            img_draw, 1 - self.config.vis.mask_opacity, 0, img_draw)

        lines = [f"frame {frame_nr}"] + frame_info
        font = cv2.FONT_HERSHEY_SIMPLEX
        fsize, pad = 0.38, 5
        dims = [cv2.getTextSize(ln, font, fsize, 1)[0] for ln in lines]
        box_w = max(w for w, _ in dims) + 2 * pad
        box_h = sum(h + pad for _, h in dims) + pad

        overlay = img_draw.copy()
        cv2.rectangle(overlay, (0, 0), (box_w, box_h), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.7, img_draw, 0.3, 0, img_draw)
        cv2.rectangle(img_draw, (0, 0), (box_w - 1, box_h - 1), (140, 140, 140), 1)

        y = pad
        for ln, (_, th) in zip(lines, dims):
            cv2.putText(img_draw, ln, (pad, y + th), font, fsize,
                        self._line_colour(ln), 1, cv2.LINE_AA)
            y += th + pad


def visualise(data, videofile, img_bg = None, img_mask = None, img_thresh=None,
              wallconts = None, zone_coords = None, roi = None, outfile = None,
              framestep = 1,
              displaystep = 25,
              cropimg=True,
              startframe = None,
              stopframe = None,
              writevideo = True,
              showvideo = False,
              videosuffix = "_V",
              fps=25,
              trajlength=50,
              resize = 1,
              smoothresize = False,
              canvasdims = None,
              logo=None,
              logooffsets=(10,10),
              drawwalls=True,
              drawwallborder=True,
              drawroi=True,
              drawmask=True,
              drawmaskborder=True,
              drawtrajs=True,
              drawtrajsbehind=False,
              partrajopacity=0.5,
              partrajcol = namedcols("yellow"),
              parmaskptcol = namedcols("pink"),
              parwallptcol = namedcols("pink"),
              parcomcol = namedcols("white"),
              pararrowcol = namedcols("white"),
              parmaskptsize = 6,
              parwallptsize = 6,
              parwallsborderthick = 3,
              parwallscol = namedcols("purple"),
              drawwallsborder = True,
              parwallsbordercol = namedcols("black"),
              drawobjects = False,
              drawcontours = False,
              drawcentroid = True,
              drawID = True,
              draweyes = False,
              drawvision = False,
              draworientarrow = True,
              drawhead = False,
              drawtail = False,
              drawframenr = True,
              drawptonwalls = False,
              drawptonmask = False,
              drawskeleton = False):

    # Setup data to show
    startfr = min(data.frame) if startframe is None else startframe
    stopfr = max(data.frame) if stopframe is None else stopframe
    data = data.loc[(data.frame>=startfr) & (data.frame<=stopfr)].copy()
    framelist = None if framestep <= 1 else list(data.iloc[list(range(1,len(data),framestep))]["frame"])

    # Resize data
    if resize != 1:
        cols = ["cx","cy"]
        if "fx" in data:
            cols += ["fx","fy"]
        if "tx" in data:
            cols += ["tx","ty"]
        if "mx" in data:
            cols += ["mx","my"]
        if "wx" in data:
            cols += ["wx","wy"]
        data[cols] = data[cols]*resize
    if "com" not in data:
        if "cx" in data:
            data["com"] = [np.nan if np.isnan(a) else (int(a),int(b)) for a,b in zip(data.cx, data.cy)]
    if "head" not in data:
        if "fx" in data:
            data["head"] = [np.nan if np.isnan(a) else (int(a),int(b)) for a,b in zip(data.fx, data.fy)]
        else:
            data["head"] = np.nan
    if "tail" not in data:
        if "tx" in data:
            data["tail"] = [np.nan if np.isnan(a) else (int(a),int(b)) for a,b in zip(data.tx, data.ty)]
    if "mx" in data:
        data["maskpt"] = [np.nan if np.isnan(a) else (int(a),int(b)) for a,b in zip(data.mx, data.my)]
    if "wx" in data:
        data["wallpt"] = [np.nan if np.isnan(a) else (int(a),int(b)) for a,b in zip(data.wx, data.wy)]

    if "ID" not in data:
        data["ID"] = data.id

    # Load video and set frame to startframe
    cap = cv2.VideoCapture(videofile)
    cap.set(cv2.CAP_PROP_POS_FRAMES, startfr-1)

    # Get video dimensions
    if roi is not None and cropimg:
        vidw  = roi[1][0] - roi[0][0]
        vidh =  roi[1][1] - roi[0][1]
    else:
        _,img = cap.read()
        vidw, vidh = (img.shape[1],img.shape[0])

    wallconts2 = wallconts
    # Adjust wall coordinates for ROI crop
    if roi is not None and wallconts is not None:
        wallconts = [[[(coord[0][0]-roi[0][0],coord[0][1]-roi[0][1])] for coord in cont] for cont in wallconts]

    # Adjust zone coordinates for ROI crop
    if roi is not None and zone_coords is not None:
        xoff, yoff = roi[0]
        zone_coords = {k: [(x - xoff, y - yoff) for x, y in coords]
                       for k, coords in zone_coords.items()}

    # Mask stuff
    if cropimg and roi is not None and img_mask is not None:
        img_mask = crop(img_mask, roi[0], roi[1])
    if resize != 1 and img_mask is not None:
        img_mask = imgresize(img_mask, resize)

    # Set up for video writing
    if writevideo:
        _outfile = outfile if outfile is not None else os.path.splitext(videofile)[0]+videosuffix+".mp4"
        viddims = (vidw,vidh) if smoothresize or resize==1 else (int(vidw*resize),int(vidh*resize))
        vidoutdims = canvasdims if canvasdims is not None else viddims
        vidout = videowriter(_outfile, vidoutdims[0], vidoutdims[1], fps)

    # Set up for video display
    if showvideo:
        cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Video", vidw, vidh)


    # Start the frame loop
    print("starting frameloop..", end=" ")
    t1 = time.time()
    frame_nr = startfr
    while cap.isOpened():
        frameOK, img = cap.read()
        stop, skip, frame_nr = framechecks(cap, frameOK, framelist, stopfr, displaystep)
        if stop:
            break
        if skip:
            continue
        # Get data for drawing
        framedat = data.loc[data["frame"]==frame_nr].copy()
        framedat = framedat.to_dict('list')
        framedat["trajdat"] = []
        if len(framedat["frame"])==0:
            framedat["frame"] = [frame_nr]
        else:
            tframes = list(range(frame_nr-trajlength+1,frame_nr+1))
            for i,_ in enumerate(framedat["frame"]):
                id = framedat["ID"][i]
                framedat["trajdat"].append(list(data.query('ID==@id & frame in @tframes')["com"]))

        # Draw everything on the image
        img_draw = draw_frame(img, framedat, img_bg, img_mask, img_thresh=img_thresh,
            roi=roi,
            resizeimg=resize,
            resizetosmooth=smoothresize,
            cropimg=cropimg,
            wallconts=wallconts, wallconts2=wallconts2,
            zone_coords=zone_coords,
            logo=logo,
            logooffsets=logooffsets,
            drawwalls=drawwalls,
            drawwallsborder=drawwallsborder,
            drawroi=drawroi,
            drawmask=drawmask,
            drawmaskborder=drawmaskborder,
            drawtrajs=drawtrajs,
            drawtrajsbehind=drawtrajsbehind,
            partrajopacity=partrajopacity,
            parwallsbordercol=parwallsbordercol,
            partrajcol=partrajcol,
            drawobjects=drawobjects,
            drawcontours=drawcontours,
            drawcentroid=drawcentroid,
            drawID=drawID,
            draworientarrow=draworientarrow,
            drawhead=drawhead,
            drawtail=drawtail,
            drawframenr=drawframenr,
            drawptonwalls=drawptonwalls,
            drawskeleton=drawskeleton,
            draweyes=draweyes,
            drawvision=drawvision,
            drawptonmask=drawptonmask,
            canvasdims=canvasdims,
            parmaskptcol=parmaskptcol,
            parmaskptsize=parmaskptsize,
            parwallptsize=parwallptsize,
            parwallsborderthick=parwallsborderthick,
            parwallscol=parwallscol,
            parwallptcol=parwallptcol,
            parcomcol=parcomcol,
            pararrowcol=pararrowcol)

        # Show video
        if showvideo:
            cv2.imshow("Video", img_draw)
            key = cv2.waitKey(1) & 0xff
            if key == 27:
                break

        # Write image to video
        if writevideo:
            vidout.write(img_draw)

    # Close everything that is open
    if showvideo:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    if writevideo:
        vidout.release()

    # Final output
    timediff = time.time()-t1
    speed = str(round((frame_nr - startfr)/float(timediff), 1))
    lineprint("Completed in "+"%.2f" % timediff+" s at "+speed+" fps")


def draw_frame(img, framedat, img_bg = None, img_mask = None, img_thresh = None,
               roi = None, wallconts=None, wallconts2=None, zone_coords=None,
               cropimg = True,
               resizeimg = 1,
               resizetosmooth = True,
               drawobjects = False,
               drawtrajs = True,
               drawtrajsbehind = False,
               drawcontours = True,
               drawskeleton = True,
               drawcentroid = True,
               drawID = True,
               drawhead = True,
               drawtail = True,
               draweyes = False,
               draworientarrow = True,
               drawvision = False,
               drawroi = False,
               drawsegments = False,
               drawmask = True,
               drawmaskborder = True,
               drawptonmask = True,
               drawwalls = False,
               drawwallsborder = True,
               drawptonwalls = True,
               drawframenr = True,
               drawinfobox = False,
               canvasdims = None,
               logo = None,
               logooffsets = (10,10),
               partrajcol = namedcols("yellow"),
               parconcol = namedcols("blue"),
               parskelcol = namedcols("orange"),
               parcomcol = namedcols("white"),
               paridcol = namedcols("black"),
               parheadcol = namedcols("lightgreen"),
               partailcol = namedcols("red"),
               pararrowcol = namedcols("white"),
               parroimaskcol = namedcols("black"),
               parwallscol = namedcols("purple"),
               parwallsbordercol = namedcols("mediumpurple"),
               parcanvascol = namedcols("black"),
               parmaskptcol = namedcols("pink"),
               parwallptcol = namedcols("pink"),
               partrajopacity = 0.5,
               partrajminthick = 6.4,
               partrajmaxthick = 9,
               parconthick = 1,
               parcomdotsize = 12,
               paridsize = 0.3,
               parheaddotsize = 6,
               partaildotsize = 6,
               pararrowlen = 13,
               pararrowtiplen = 0.4,
               parframesize = 0.8,
               parwallsborderthick = 3,
               parmaskptsize = 6,
               parwallptsize = 6,
               parroimaskopacity = 0.6,
               parmaskopacity = 0.6,
               parwallsopacity = 0.6):

        # 1) Crop image
        #--------------------
        if cropimg and roi is not None:
            img = crop(img, roi[0], roi[1])

        if resizetosmooth:
            dims = (img.shape[1], img.shape[0])

        # 2) Resize image
        #--------------------
        if resizeimg != 1:
            img = imgresize(img, resizeimg)
            img_bg = imgresize(img_bg, resizeimg)

        # Copy image for drawing
        img_draw = img.copy()

        # 3) Draw background
        #--------------------
        if type(img_bg)==str:
            if img_bg == "white":
                img_draw = np.zeros(img_draw.shape, dtype="uint8") + 255
        elif img_bg is not None:
            img_nobg = cv2.absdiff(img_draw, img_bg)
            if img_mask is not None:
                img_nobg = cv2.bitwise_and(img_nobg, img_mask)
                img_nobg = 255-cv2.bitwise_not(img_nobg)
            img_draw = 255-img_nobg

        # 4) Draw trajectories
        #--------------------
        if drawtrajs:
            cols = uniqcols(len(np.unique(framedat["ID"])))
            for i,_ in enumerate(framedat["trajdat"]):
                if partrajcol is not None:
                    trajcol = partrajcol 
                else:
                    trajcol = namedcols(framedat["ID"][i]) if (type(framedat["ID"][i])==str and framedat["ID"][i][0]!="F") else cols[i]
                draw_traj(img_draw, framedat["trajdat"][i], trajcol, partrajminthick,
                          partrajmaxthick, partrajopacity)

        # 5) Draw black shapes
        #--------------------
        if drawobjects and "contours" in framedat:
            cv2.drawContours(img_draw, framedat["contours"], -1, 0, -1)

        # 6) Draw trajectories behind objects
        #--------------------
        if drawtrajsbehind and not drawobjects and img_thresh is not None:
            img_draw[img_thresh == 255] = img[img_thresh == 255]

        # 7) Draw contours
        #--------------------
        if drawcontours and "contours" in framedat:
            cv2.drawContours(img_draw, framedat["contours"], -1, parconcol, int(parconthick*resizeimg))

        # Draw furter individual object data
        for i,_ in enumerate(framedat["ID"]):
            col = True if len(framedat["ID"])>3 else False

            # 8) Draw skeleton
            #--------------------
            if drawskeleton and "skeleton" in framedat:
                img_draw = draw_coordlist(img_draw, framedat["skeleton"][i], parskelcol)

            # 9) Draw centroid
            #--------------------
            if drawcentroid:
                comcol = namedcols(framedat["ID"][i]) if (type(framedat["ID"][i])==str and framedat["ID"][i][0]!="F") else parcomcol
                if framedat["com"][i]==framedat["com"][i]:
                    cv2.circle(img_draw, framedat["com"][i], 0, comcol, int(parcomdotsize*resizeimg))

            # 10) Draw ID
            #--------------------
            if drawID:
                if framedat["com"][i]==framedat["com"][i]:
                    if i<9:
                        textloc = (framedat["com"][i][0]-int(4*resizeimg),framedat["com"][i][1]-int(4*resizeimg))
                    else:
                        textloc = (framedat["com"][i][0]-int(6*resizeimg),framedat["com"][i][1]-int(4*resizeimg))
                    idcol = "white" if type(framedat["ID"][i])==str else paridcol
                    draw_text(img_draw, str(i+1), textloc, paridsize*resizeimg, idcol, 0, 1)

            # 11) Draw head
            #--------------------
            if drawhead and "head" in framedat:
                if framedat["head"][i]==framedat["head"][i]:
                    cv2.circle(img_draw, framedat["head"][i], 0, parheadcol, int(parheaddotsize*resizeimg))

            # 12) Draw tail
            #--------------------
            if drawtail and "tail" in framedat:
                if framedat["tail"][i]==framedat["tail"][i]:
                    cv2.circle(img_draw, framedat["tail"][i], 0, partailcol, int(partaildotsize*resizeimg))

            # 13) Draw eyes
            #--------------------
            # TO ADD

            # 14) Draw orientation arrow
            #--------------------
            if draworientarrow:
                tip = None 
                if "orient" in framedat:
                    if framedat["orient"][i]==framedat["orient"][i]:
                        tip = get_coord(framedat["cx"][i], framedat["cy"][i], framedat["orient"][i], int(pararrowlen*resizeimg), True)
                else:
                    if framedat["heading"][i]==framedat["heading"][i]:
                        tip = get_coord(framedat["cx"][i], framedat["cy"][i], framedat["heading"][i], int(pararrowlen*resizeimg), True)
                if tip is not None:
                    cv2.arrowedLine(img_draw, (int(framedat["cx"][i]), int(framedat["cy"][i])), tip, pararrowcol, 1, tipLength = pararrowtiplen*resizeimg)

            # 15) Draw vision
            #--------------------
            # TO ADD

        # 16) Draw ROI box
        #--------------------
        if not cropimg and drawroi and roi is not None:
            stencil = np.zeros(img.shape).astype(img.dtype)
            stencil[:] = parroimaskcol
            tl, br = [tuple(int(i*resizeimg) for i in pt) for pt in roi]
            stencil[tl[1]:br[1], tl[0]:br[0]] = crop(img_draw, tl, br)
            cv2.addWeighted(stencil, parroimaskopacity, img_draw, 1-parroimaskopacity, 0, img_draw)

        # 17) Draw zones
        #--------------------
        if zone_coords is not None:
            overlay = img_draw.copy()
            for zone_idx, coords in zone_coords.items():
                scaled = [(int(x * resizeimg), int(y * resizeimg)) for x, y in coords]
                contour = np.array([[[x, y]] for x, y in scaled], dtype=np.int32)
                col = _ZONE_COLORS[(zone_idx - 1) % len(_ZONE_COLORS)]
                cv2.drawContours(overlay, [contour], -1, col, -1)
            cv2.addWeighted(overlay, 0.18, img_draw, 0.82, 0, img_draw)
            font = cv2.FONT_HERSHEY_SIMPLEX
            for zone_idx, coords in zone_coords.items():
                scaled = [(int(x * resizeimg), int(y * resizeimg)) for x, y in coords]
                contour = np.array([[[x, y]] for x, y in scaled], dtype=np.int32)
                col = _ZONE_COLORS[(zone_idx - 1) % len(_ZONE_COLORS)]
                cv2.drawContours(img_draw, [contour], -1, col, 2)
                cx = int(np.mean([x for x, y in scaled]))
                cy = int(np.mean([y for x, y in scaled]))
                cv2.putText(img_draw, f"Z{zone_idx}", (cx - 9, cy + 5), font, 0.4,
                            (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(img_draw, f"Z{zone_idx}", (cx - 9, cy + 5), font, 0.4,
                            (255, 255, 255), 1, cv2.LINE_AA)

        # 18) Draw mask
        #--------------------
        if drawmask and img_mask is not None:
            img_masked = cv2.bitwise_and(img_draw, img_mask)
            cv2.addWeighted(img_masked, parmaskopacity, img_draw, 1-parmaskopacity, 0, img_draw)
            if drawptonmask and "maskpt" in framedat:
                for i,_ in enumerate(framedat["trajdat"]):
                    if framedat["maskpt"][i]==framedat["maskpt"][i]:
                        cv2.circle(img_draw, framedat["maskpt"][i], 0, parmaskptcol, int(parmaskptsize*resizeimg))

        # 19) Draw walls
        #--------------------
        if (drawwalls or drawwallsborder) and wallconts is not None:
            #wallconts = [tuple(int(i*resizeimg) for i in pt) for pt in wallconts]
            wallconts = [[[list(np.round(i*resizeimg).astype(int))] for pt in cont for i in pt] for cont in wallconts]
            img_masked = img_draw.copy()
            if drawwallsborder:
                #wallconts2 = [tuple(int(i*resizeimg) for i in pt) for pt in wallconts2]
                thick = int(parwallsborderthick*resizeimg)
                #cv2.polylines(img_masked, wallconts, 0, parwallsbordercol, thick)
                cv2.drawContours(img_masked, wallconts2, contourIdx=-1, color=parwallsbordercol,thickness=thick)
            #cv2.fillPoly(img_masked, wallconts, parwallscol)
            if drawwalls:
                cv2.drawContours(img_masked, wallconts2, contourIdx=-1, color=parwallscol,thickness=-1)
            cv2.addWeighted(img_masked, parwallsopacity, img_draw, 1-parwallsopacity, 0, img_draw)
            if drawptonwalls and "wallpt" in framedat:
                for i,_ in enumerate(framedat["trajdat"]):
                    if framedat["wallpt"][i]==framedat["wallpt"][i]:
                        cv2.circle(img_draw, framedat["wallpt"][i], 0, parwallptcol, int(parwallptsize*resizeimg))

        # 20) Draw framenr
        #--------------------
        if drawframenr and not drawinfobox:
            draw_text(img_draw, str(list(framedat["frame"])[0]), (0,0), parframesize*resizeimg, margin=5, bgcol="white")

        # 21) Draw infobox
        #--------------------
        if drawinfobox:
            _info_lines = [f"frame {list(framedat['frame'])[0]}"]
            for i, id in enumerate(framedat.get("ID", [])):
                line = f"ID{id}"
                if "area" in framedat and i < len(framedat["area"]) and framedat["area"][i] == framedat["area"][i]:
                    line += f" area={int(framedat['area'][i])}"
                if "aspect_ratio" in framedat and i < len(framedat["aspect_ratio"]) and framedat["aspect_ratio"][i] == framedat["aspect_ratio"][i]:
                    line += f" ar={framedat['aspect_ratio'][i]:.2f}"
                _info_lines.append(line)
            _font = cv2.FONT_HERSHEY_SIMPLEX
            _fsize, _pad = 0.38, 4
            _dims = [cv2.getTextSize(l, _font, _fsize, 1)[0] for l in _info_lines]
            _box_w = max(w for w, h in _dims) + 2 * _pad
            _box_h = sum(h + _pad for w, h in _dims) + _pad
            cv2.rectangle(img_draw, (0, 0), (_box_w, _box_h), (255, 255, 255), -1)
            y = _pad
            for line, (_, th) in zip(_info_lines, _dims):
                cv2.putText(img_draw, line, (_pad, y + th), _font, _fsize, (0, 0, 0), 1, cv2.LINE_AA)
                y += th + _pad

        # Return to normal size when resizing for smoothing
        if resizetosmooth and resizeimg != 1:
            img_draw = imgresize(img_draw, dims = dims, back = True)

        # 22) Add image to canvas
        #--------------------
        if canvasdims is not None:
            img_draw = addcanvas(img_draw, canvasdims, parcanvascol)

        # 23) Draw logo
        #--------------------
        if logo is not None:
            img_draw = add_transimg(img_draw, logo, logooffsets)

        return img_draw
