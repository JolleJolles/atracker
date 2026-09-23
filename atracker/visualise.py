#! /usr/bin/env python

import os
import cv2
import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from pythutils.sysutils import lineprint
from pythutils.mediautils import crop, videowriter, add_transimg, imgresize
from pythutils.drawutils import namedcols, draw_text, draw_traj, uniqcols
from pythutils.mathutils import points_to_angle

from .helpers.media import framechecks
from .helpers.geometry import get_coord
from .helpers.contours import draw_coordlist


def addcanvas(img, dims, color):
    bgcanvas = np.zeros((dims[1], dims[0], 3), dtype="uint8") + color
    wratio = dims[0] / float(img.shape[1])
    hratio = dims[1] / float(img.shape[0])
    resize = min(wratio, hratio)
    img = imgresize(img, resize)
    if wratio == hratio:
        bgcanvas = img
    elif wratio < hratio:
        extra = int((bgcanvas.shape[0] - img.shape[0]) / 2)
        bgcanvas[extra:extra + img.shape[0], :] = img
    else:
        extra = int((bgcanvas.shape[1] - img.shape[1]) / 2)
        bgcanvas[:, extra:extra + img.shape[1]] = img
    return bgcanvas.astype(np.uint8)


_ZONE_COLORS = [
    (0, 210, 255),   # orange
    (255, 255, 0),   # cyan
    (0, 220, 80),    # lime green
    (200, 0, 230),   # pink/magenta
    (255, 180, 0),   # teal/azure
    (50, 50, 255),   # red
]


class Visualiser:
    """
    Unified visualisation for tracking and post-processing video output.

    Instantiate once per video with scene geometry and visual settings, then:
    - During live tracking: call draw_thresh_pass(), draw_orient_link(),
      draw_scene_overlays(), and draw_info_overlay() per frame.
    - For post-processing from a CSV: call render().
    - For single-frame drawing: call draw_frame() directly.
    """

    def __init__(self, objects=1, thresh_types=None, threshcolors=False,
                 orientfrombw=False, mask_contours=None, wall_contours=None,
                 zone_coords=None, config=None):
        self.objects       = max(1, int(objects))
        self.thresh_types  = thresh_types or ["bw"]
        self.threshcolors  = threshcolors
        self.orientfrombw  = orientfrombw
        self.mask_contours = mask_contours
        self.wall_contours = wall_contours
        self.zone_coords   = zone_coords

        cfg = config.vis if (config is not None and hasattr(config, "vis")) else None

        def _cv(key, default):
            v = getattr(cfg, key, None) if cfg is not None else None
            return default if v is None else v

        # Core colours (config stores them as eval-able strings)
        self.col_contour     = eval(str(_cv("contour_col", str(namedcols("blue")))))
        self.col_com         = eval(str(_cv("centre_col",  str(namedcols("white")))))
        self.col_orient      = eval(str(_cv("orient_col",  str(namedcols("white")))))
        self.col_traj        = eval(str(_cv("traj_col",    str(namedcols("yellow")))))
        self.col_red         = namedcols("red")
        self.col_lightgreen  = namedcols("lightgreen")
        self.col_orange      = namedcols("orange")
        self.col_head        = namedcols("lightgreen")
        self.col_tail        = namedcols("red")
        self.col_skel        = namedcols("orange")
        self.col_mask_pt     = namedcols("pink")
        self.col_wall_pt     = namedcols("pink")
        self.col_wall_fill   = namedcols("purple")
        self.col_wall_border = namedcols("mediumpurple")

        # Numeric style settings
        self.idcol         = _cv("idcol",         True)
        self.traj_minthick = float(_cv("traj_minthick", 6.4))
        self.traj_maxthick = float(_cv("traj_maxthick", 9.0))
        self.traj_opacity  = float(_cv("traj_opacity",  0.5))
        self.centre_lwidth = int(  _cv("centre_lwidth", 13))
        self.orient_lwidth = int(  _cv("orient_lwidth", 2))
        self.orient_length = int(  _cv("orient_length", 13))
        self.mask_opacity  = float(_cv("mask_opacity",  0.6))
        self.box_opacity   = float(_cv("box_opacity",   0.7))
        self.trajs_below   = bool( _cv("trajs_below",   False))
        self.wall_opacity  = 0.6

        self.cols        = uniqcols(self.objects)
        self.thresh_cols = {t: namedcols(t) for t in self.thresh_types
                            if not t.startswith("bw")}

    # ------------------------------------------------------------------
    # Scene overlays  (walls, zones, mask border)
    # ------------------------------------------------------------------
    def draw_scene_overlays(self, img_draw,
                            wall_contours=None, zone_coords=None,
                            mask_contours=None):
        """Draw wall fill/border, zone fills with labels, and mask border."""
        wc = wall_contours  if wall_contours  is not None else self.wall_contours
        zc = zone_coords    if zone_coords    is not None else self.zone_coords
        mc = mask_contours  if mask_contours  is not None else self.mask_contours

        if wc is not None:
            overlay = img_draw.copy()
            cv2.drawContours(overlay, wc, -1, (160, 50, 160), -1)
            cv2.addWeighted(overlay, 0.35, img_draw, 0.65, 0, img_draw)
            cv2.drawContours(img_draw, wc, -1, (110, 30, 110), 2)

        if zc is not None:
            overlay = img_draw.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            for zone_idx, coords in zc.items():
                contour = np.array([[[x, y]] for x, y in coords], dtype=np.int32)
                col = _ZONE_COLORS[(zone_idx - 1) % len(_ZONE_COLORS)]
                cv2.drawContours(overlay, [contour], -1, col, -1)
            cv2.addWeighted(overlay, 0.18, img_draw, 0.82, 0, img_draw)
            for zone_idx, coords in zc.items():
                contour = np.array([[[x, y]] for x, y in coords], dtype=np.int32)
                col = _ZONE_COLORS[(zone_idx - 1) % len(_ZONE_COLORS)]
                cv2.drawContours(img_draw, [contour], -1, col, 2)
                cx = int(np.mean([x for x, _ in coords]))
                cy = int(np.mean([y for _, y in coords]))
                cv2.putText(img_draw, f"Z{zone_idx}", (cx - 9, cy + 5),
                            font, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(img_draw, f"Z{zone_idx}", (cx - 9, cy + 5),
                            font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        if mc is not None:
            cv2.drawContours(img_draw, mc, -1, (180, 180, 180), 1)

    # ------------------------------------------------------------------
    # Info overlay  (mask darkening + semi-transparent box)
    # ------------------------------------------------------------------
    def _line_colour(self, line):
        if line.startswith("frame "):
            return (50, 50, 50)
        if line.startswith("ID"):
            id_str = line[2:line.index(":")] if ":" in line else line[2:]
            try:
                base = self.cols[(int(id_str) - 1) % len(self.cols)]
                return tuple(max(0, int(v * 0.55)) for v in base)
            except ValueError:
                try:
                    base = namedcols(id_str)
                    return tuple(max(0, int(v * 0.55)) for v in base)
                except Exception:
                    pass
        return (80, 80, 80)

    def draw_info_overlay(self, img_draw, img_mask, frame_nr, frame_info):
        """Apply mask darkening and draw a semi-transparent info box."""
        if img_mask is not None:
            img_masked = cv2.bitwise_and(img_draw, img_draw, mask=img_mask)
            cv2.addWeighted(img_masked, self.mask_opacity,
                            img_draw, 1 - self.mask_opacity, 0, img_draw)

        lines = [f"frame {frame_nr}"] + list(frame_info)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fsize, pad = 0.38, 5
        dims = [cv2.getTextSize(ln, font, fsize, 1)[0] for ln in lines]
        box_w = max(w for w, _ in dims) + 2 * pad
        box_h = sum(h + pad for _, h in dims) + pad

        overlay = img_draw.copy()
        cv2.rectangle(overlay, (0, 0), (box_w, box_h), (255, 255, 255), -1)
        cv2.addWeighted(overlay, self.box_opacity, img_draw,
                        1 - self.box_opacity, 0, img_draw)
        cv2.rectangle(img_draw, (0, 0), (box_w - 1, box_h - 1), (140, 140, 140), 1)

        y = pad
        for ln, (_, th) in zip(lines, dims):
            cv2.putText(img_draw, ln, (pad, y + th), font, fsize,
                        self._line_colour(ln), 1, cv2.LINE_AA)
            y += th + pad

    # ------------------------------------------------------------------
    # Unified per-frame drawing
    # ------------------------------------------------------------------
    def draw_frame(self, img, frame_data,
                   img_bg=None, img_mask=None,
                   roi=None, cropimg=True, resizeimg=1, resizetosmooth=False,
                   # draw toggles
                   draw_trajs=True, draw_trajs_behind=False,
                   draw_contours=True, draw_all_contours=False,
                   draw_skeleton=False,
                   draw_centroid=True, draw_id=True,
                   draw_head=True, draw_tail=True, draw_orient=True,
                   draw_mask=True, draw_mask_pt=False,
                   draw_walls_pt=False,
                   draw_scene=True,
                   draw_frame_nr=True, draw_info_box=False,
                   draw_roi=False,
                   # optional scene override (for ROI-adjusted coords in render())
                   wall_contours=None, zone_coords=None,
                   canvasdims=None, logo=None, logooffsets=(10, 10)):
        """
        Draw frame_data onto img and return the annotated frame.

        frame_data keys
        ---------------
        ID       : list of object identifiers
        com      : list of (cx, cy) tuples or np.nan
        head     : list of (fx, fy) or np.nan          (optional)
        tail     : list of (tx, ty) or np.nan          (optional)
        orient   : list of angle floats or np.nan      (optional)
        heading  : list of angle floats or np.nan      (fallback for orient)
        trajdat  : list of [(cx,cy),...] per object
        maskpt   : list of (mx, my) or np.nan          (optional)
        wallpt   : list of (wx, wy) or np.nan          (optional)
        contours : list of cv2 contour arrays          (optional, tracking)
        allcons  : list of all detected contours        (optional, tracking)
        skeleton : list of skeleton coord lists         (optional)
        frame    : frame number (int or list with one int)
        area     : list of float areas                  (optional, for info box)
        aspect_ratio : list of floats                  (optional, for info box)
        """
        # 1) Crop
        if cropimg and roi is not None:
            if isinstance(img_bg, np.ndarray) and img_bg.shape[:2] == img.shape[:2]:
                img_bg = crop(img_bg, roi[0], roi[1])
            img = crop(img, roi[0], roi[1])
        if resizetosmooth:
            orig_dims = (img.shape[1], img.shape[0])

        # 2) Resize
        if resizeimg != 1:
            img    = imgresize(img, resizeimg)
            if img_bg is not None and not isinstance(img_bg, str):
                img_bg = imgresize(img_bg, resizeimg)

        img_draw = img.copy()

        # 3) Background
        if isinstance(img_bg, str) and img_bg == "white":
            img_draw = np.zeros(img_draw.shape, dtype="uint8") + 255
        elif img_bg is not None:
            img_nobg = cv2.absdiff(img_draw, img_bg)
            if img_mask is not None:
                img_nobg = cv2.bitwise_and(img_nobg, img_mask)
                img_nobg = 255 - cv2.bitwise_not(img_nobg)
            img_draw = np.asarray(255 - img_nobg)

        # 4) Trajectories
        ids     = frame_data.get("ID", [])
        trajdat = frame_data.get("trajdat", [])
        if draw_trajs and trajdat:
            for i, traj in enumerate(trajdat):
                if len(traj) < 2:
                    continue
                id_ = ids[i] if i < len(ids) else i + 1
                if self.idcol and isinstance(id_, (int, np.integer)):
                    col = self.cols[(int(id_) - 1) % len(self.cols)]
                elif isinstance(id_, str) and not id_.startswith("F"):
                    col = self.thresh_cols.get(id_, self.col_traj)
                else:
                    col = self.col_traj
                draw_traj(img_draw, traj, col,
                          self.traj_minthick, self.traj_maxthick, self.traj_opacity)

        # 5) All detected contours in red (tracking only)
        if draw_all_contours and frame_data.get("allcons"):
            cv2.drawContours(img_draw, frame_data["allcons"], -1, self.col_red, 1)

        # 6) Trajectories behind objects
        if draw_trajs_behind and frame_data.get("img_thresh") is not None:
            img_draw[frame_data["img_thresh"] == 255] = img[frame_data["img_thresh"] == 255]

        # 7) ID'd contours
        if draw_contours and frame_data.get("contours"):
            cv2.drawContours(img_draw, frame_data["contours"], -1, self.col_contour, 1)

        # Per-object details
        coms      = frame_data.get("com",          [])
        heads     = frame_data.get("head",         [])
        tails     = frame_data.get("tail",         [])
        orients   = frame_data.get("orient",       [])
        headings  = frame_data.get("heading",      [])
        skeletons = frame_data.get("skeleton",     [])
        mask_pts  = frame_data.get("maskpt",       [])
        wall_pts  = frame_data.get("wallpt",       [])

        for i, id_ in enumerate(ids):
            com = coms[i] if i < len(coms) else np.nan
            com_valid = com == com  # NaN check

            # 8) Skeleton
            if draw_skeleton and i < len(skeletons):
                sk = skeletons[i]
                if sk == sk:
                    img_draw = draw_coordlist(img_draw, sk, self.col_skel)

            # 9) Centroid
            if draw_centroid and com_valid:
                comcol = (self.thresh_cols.get(id_, self.col_com)
                          if isinstance(id_, str) and not id_.startswith("F")
                          else self.col_com)
                cv2.circle(img_draw, com, 0, comcol, int(12 * resizeimg))

            # 10) ID text
            if draw_id and com_valid:
                off = int(4 * resizeimg)
                draw_text(img_draw, str(id_),
                          (com[0] - off, com[1] - off),
                          0.3 * resizeimg, "black", 0, 1)

            # 11) Head
            if draw_head and i < len(heads):
                h = heads[i]
                if h == h:
                    cv2.circle(img_draw, h, 0, self.col_head, int(6 * resizeimg))

            # 12) Tail
            if draw_tail and i < len(tails):
                t = tails[i]
                if t == t:
                    cv2.circle(img_draw, t, 0, self.col_tail, int(6 * resizeimg))

            # 14) Orientation arrow
            if draw_orient and com_valid:
                angle = None
                if i < len(orients) and orients[i] == orients[i]:
                    angle = orients[i]
                elif i < len(headings) and headings[i] == headings[i]:
                    angle = headings[i]
                if angle is not None:
                    tip = get_coord(com[0], com[1], angle,
                                    int(self.orient_length * resizeimg), True)
                    cv2.arrowedLine(img_draw, com, tip, self.col_orient,
                                    self.orient_lwidth, tipLength=0.4 * resizeimg)

        # 16) ROI outline
        if draw_roi and not cropimg and roi is not None:
            stencil = np.zeros(img.shape).astype(img.dtype)
            stencil[:] = namedcols("black")
            tl, br = [tuple(int(c * resizeimg) for c in pt) for pt in roi]
            stencil[tl[1]:br[1], tl[0]:br[0]] = crop(img_draw, tl, br)
            cv2.addWeighted(stencil, 0.6, img_draw, 0.4, 0, img_draw)

        # 17) Scene overlays (walls, zones, mask border)
        if draw_scene:
            self.draw_scene_overlays(img_draw,
                                     wall_contours=wall_contours,
                                     zone_coords=zone_coords)

        # 18) Mask darkening + nearest-mask dot
        if draw_mask and img_mask is not None:
            img_masked = cv2.bitwise_and(img_draw, img_mask)
            cv2.addWeighted(img_masked, self.mask_opacity,
                            img_draw, 1 - self.mask_opacity, 0, img_draw)
            if draw_mask_pt:
                for mpt in mask_pts:
                    if mpt == mpt:
                        cv2.circle(img_draw, mpt, 0, self.col_mask_pt,
                                   int(6 * resizeimg))

        # 19) Nearest-wall dot
        if draw_walls_pt:
            for wpt in wall_pts:
                if wpt == wpt:
                    cv2.circle(img_draw, wpt, 0, self.col_wall_pt, int(6 * resizeimg))

        # 20) Frame number
        fn = frame_data.get("frame")
        if isinstance(fn, list):
            fn = fn[0] if fn else None
        if draw_frame_nr and not draw_info_box and fn is not None:
            draw_text(img_draw, str(fn), (0, 0), 0.8 * resizeimg,
                      margin=5, bgcol="white")

        # 21) Info box
        if draw_info_box and fn is not None:
            info_lines = [f"frame {fn}"]
            areas = frame_data.get("area", [])
            ars   = frame_data.get("aspect_ratio", [])
            for i, id_ in enumerate(ids):
                line = f"ID{id_}"
                if i < len(areas) and areas[i] == areas[i]:
                    line += f" area={int(areas[i])}"
                if i < len(ars) and ars[i] == ars[i]:
                    line += f" ar={ars[i]:.2f}"
                info_lines.append(line)
            font  = cv2.FONT_HERSHEY_SIMPLEX
            fsize = 0.38
            pad   = 4
            tdims = [cv2.getTextSize(l, font, fsize, 1)[0] for l in info_lines]
            if tdims:
                box_w = max(w for w, _ in tdims) + 2 * pad
                box_h = sum(h + pad for _, h in tdims) + pad
                cv2.rectangle(img_draw, (0, 0), (box_w, box_h), (255, 255, 255), -1)
                y = pad
                for line, (_, th) in zip(info_lines, tdims):
                    cv2.putText(img_draw, line, (pad, y + th), font, fsize,
                                self._line_colour(line), 1, cv2.LINE_AA)
                    y += th + pad

        # Return to original size (smooth-resize mode)
        if resizetosmooth and resizeimg != 1:
            img_draw = imgresize(img_draw, dims=orig_dims, back=True)

        # Canvas + logo
        if canvasdims is not None:
            img_draw = addcanvas(img_draw, canvasdims, namedcols("black"))
        if logo is not None:
            img_draw = add_transimg(img_draw, logo, logooffsets)

        return img_draw

    # ------------------------------------------------------------------
    # Batch post-processing
    # ------------------------------------------------------------------
    def render(self, data, videofile=None, outfile=None,
               img_bg=None, img_mask=None, roi=None,
               startframe=None, stopframe=None,
               fps=25, resize=1, trajlength=50,
               writevideo=True, showvideo=False,
               videosuffix="_V", canvasdims=None,
               logo=None, logooffsets=(10, 10),
               framestep=1, displaystep=25, cropimg=True,
               **draw_kwargs):
        """
        Create a visualisation video from CSV data.

        Parameters
        ----------
        data      : DataFrame or path to CSV.
        videofile : path to original video; None → draw on a blank canvas.
        outfile   : output path; default is videofile stem + videosuffix + .mp4.
        **draw_kwargs : forwarded to draw_frame() for every frame.
        """
        if isinstance(data, str):
            data = pd.read_csv(data)

        startfr = int(data.frame.min()) if startframe is None else int(startframe)
        stopfr  = int(data.frame.max()) if stopframe  is None else int(stopframe)
        data = data.loc[(data.frame >= startfr) & (data.frame <= stopfr)].copy()
        framelist = (None if framestep <= 1
                     else list(data.iloc[list(range(1, len(data), framestep))]["frame"]))

        # Rescale coordinate columns
        if resize != 1:
            for col in ["cx", "cy", "fx", "fy", "tx", "ty", "mx", "my", "wx", "wy"]:
                if col in data:
                    data[col] = data[col] * resize

        # Build tuple columns from x/y pairs
        def _ptcol(xc, yc):
            return [np.nan if not np.isfinite(float(a)) else (int(a), int(b))
                    for a, b in zip(data[xc].values, data[yc].values)]

        if "com" not in data and "cx" in data:
            data["com"] = _ptcol("cx", "cy")
        if "head" not in data and "fx" in data:
            data["head"] = _ptcol("fx", "fy")
        if "tail" not in data and "tx" in data:
            data["tail"] = _ptcol("tx", "ty")
        if "maskpt" not in data and "mx" in data:
            data["maskpt"] = _ptcol("mx", "my")
        if "wallpt" not in data and "wx" in data:
            data["wallpt"] = _ptcol("wx", "wy")
        if "ID" not in data and "id" in data:
            data["ID"] = data["id"]

        # ROI-adjust scene overlays (they're in full-image coords)
        _wall_conts  = self.wall_contours
        _zone_coords = self.zone_coords
        xo, yo = roi[0] if roi is not None and cropimg else (0, 0)
        if _wall_conts is not None:
            _wall_conts = [np.rint((np.asarray(cont) - (xo, yo)) * resize).astype(np.int32)
                           for cont in _wall_conts]
        if _zone_coords is not None:
            _zone_coords = {
                k: [(int(round((x - xo) * resize)), int(round((y - yo) * resize)))
                    for x, y in coords]
                for k, coords in _zone_coords.items()
            }

        # Crop and resize mask
        _img_mask = img_mask
        if cropimg and roi is not None and _img_mask is not None:
            _img_mask = crop(_img_mask, roi[0], roi[1])
        if resize != 1 and _img_mask is not None:
            _img_mask = imgresize(_img_mask, resize)

        # Video / canvas dimensions
        if videofile is not None:
            cap = cv2.VideoCapture(videofile)
            cap.set(cv2.CAP_PROP_POS_FRAMES, startfr - 1)
            if roi is not None and cropimg:
                vidw = roi[1][0] - roi[0][0]
                vidh = roi[1][1] - roi[0][1]
            else:
                ok, _f = cap.read()
                vidw, vidh = (_f.shape[1], _f.shape[0]) if ok else (640, 480)
                cap.set(cv2.CAP_PROP_POS_FRAMES, startfr - 1)
        else:
            cap = None
            if _img_mask is not None:
                vidh, vidw = _img_mask.shape[:2]
            elif img_bg is not None and not isinstance(img_bg, str):
                vidh, vidw = img_bg.shape[:2]
            else:
                vidw, vidh = 640, 480

        if resize != 1:
            vidw, vidh = int(vidw * resize), int(vidh * resize)

        if writevideo:
            if outfile is None:
                stem = os.path.splitext(videofile)[0] if videofile else "output"
                outfile = stem + videosuffix + ".mp4"
            frame_dims = tuple(int(v) for v in (canvasdims if canvasdims is not None else (vidw, vidh)))
            # MP4 encoders commonly require even frame dimensions. Pad the
            # right/bottom edge rather than cropping pixels or moving points.
            vidoutdims = tuple(v + v % 2 for v in frame_dims)
            try:
                vidout = _open_video_writer(outfile, vidoutdims, fps)
            except Exception:
                if cap is not None:
                    cap.release()
                raise
            lineprint(f"Writing video to: {os.path.abspath(outfile)}")
        else:
            lineprint("Video saving disabled (writevideo=False); no output file will be created.")

        if showvideo:
            cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Video", vidw, vidh)

        print("starting frameloop..", end=" ")
        t1 = time.time()
        frame_nr = startfr
        frames_written = 0

        try:
            if cap is not None:
                # Video-driven frame loop
                while cap.isOpened():
                    frameOK, img = cap.read()
                    stop, skip, frame_nr = framechecks(
                        cap, frameOK, framelist, stopfr, displaystep)
                    if stop:
                        break
                    if skip:
                        continue

                    frame_data = _build_frame_data(data, frame_nr, trajlength)
                    img_draw = self.draw_frame(
                        img, frame_data,
                        img_bg=img_bg, img_mask=_img_mask,
                        roi=roi, cropimg=cropimg, resizeimg=resize,
                        wall_contours=_wall_conts, zone_coords=_zone_coords,
                        canvasdims=canvasdims, logo=logo, logooffsets=logooffsets,
                        **draw_kwargs)

                    if showvideo:
                        cv2.imshow("Video", img_draw)
                        if cv2.waitKey(1) & 0xff == 27:
                            break
                    if writevideo:
                        vidout.write(_prepare_video_frame(img_draw, frame_dims))
                        frames_written += 1
            else:
                # No video: iterate over frames that have data
                for frame_nr in sorted(data["frame"].unique()):
                    frame_nr = int(frame_nr)
                    if frame_nr < startfr or frame_nr > stopfr:
                        continue
                    img = np.zeros((vidh, vidw, 3), dtype="uint8")
                    frame_data = _build_frame_data(data, frame_nr, trajlength)
                    img_draw = self.draw_frame(
                        img, frame_data,
                        img_bg=img_bg, img_mask=_img_mask,
                        roi=None, cropimg=False, resizeimg=1,
                        wall_contours=_wall_conts, zone_coords=_zone_coords,
                        canvasdims=canvasdims, logo=logo, logooffsets=logooffsets,
                        **draw_kwargs)

                    if showvideo:
                        cv2.imshow("Video", img_draw)
                        if cv2.waitKey(1) & 0xff == 27:
                            break
                    if writevideo:
                        vidout.write(_prepare_video_frame(img_draw, frame_dims))
                        frames_written += 1

        finally:
            if cap is not None:
                cap.release()
            if showvideo:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            if writevideo:
                vidout.release()
        if writevideo:
            if frames_written == 0 or not os.path.isfile(outfile) or os.path.getsize(outfile) == 0:
                raise RuntimeError(f"No video output was written: {os.path.abspath(outfile)}")
            lineprint(f"Saved video ({frames_written} frames submitted): {os.path.abspath(outfile)}")

        timediff = time.time() - t1
        speed = round((frame_nr - startfr) / max(timediff, 0.001), 1)
        lineprint(f"Completed in {timediff:.2f}s at {speed}fps")

    # ------------------------------------------------------------------
    # Tracking-specific methods
    # ------------------------------------------------------------------
    def draw_thresh_pass(self, img_draw, img, img_thresh, conlist, allcons,
                         ids, filtered_coms, traj_history, tracked_ids, thresh_type):
        """Draw one threshold-type pass onto img_draw during live tracking."""
        # Trajectories
        draw_ids = sorted(tracked_ids) if thresh_type.startswith("bw") else [thresh_type]
        for id_ in draw_ids:
            trajdat = list(reversed(traj_history.get(id_, [])))
            if len(trajdat) >= 2:
                if not thresh_type.startswith("bw"):
                    col = self.thresh_cols.get(thresh_type, self.col_traj)
                elif self.idcol and isinstance(id_, (int, np.integer)):
                    col = self.cols[(int(id_) - 1) % len(self.cols)]
                else:
                    col = self.col_traj
                draw_traj(img_draw, trajdat, col,
                          self.traj_minthick, self.traj_maxthick, self.traj_opacity)

        if self.trajs_below:
            img_draw[img_thresh == 255] = img[img_thresh == 255]

        # All detected contours (thin red)
        cv2.drawContours(img_draw, allcons, -1, self.col_red, 1)

        # ID'd contours
        is_merged = (self.objects > 1 and thresh_type.startswith("bw")
                     and conlist.get("consmerged"))
        contour_col = (128 if not thresh_type.startswith("bw")
                       else self.col_contour if not is_merged else 128)
        cv2.drawContours(img_draw, conlist["contour"], -1, contour_col, 1)

        # Per-object details
        for i, id_ in enumerate(conlist["id"]):
            sk = conlist["skeleton"][i]
            if sk == sk:
                img_draw = draw_coordlist(img_draw, sk, self.col_orange)
            t = conlist["tail"][i]
            if t == t:
                cv2.circle(img_draw, t, 0, self.col_red, 6)
            h = conlist["head"][i]
            if h == h:
                arrowtip = get_coord(h[0], h[1], conlist["angle"][i], 13, True)
                cv2.arrowedLine(img_draw, h, arrowtip,
                                self.col_orient, 1, tipLength=0.4)
                cv2.circle(img_draw, h, 0, self.col_lightgreen, 6)
            c = conlist["com"][i]
            if c == c:
                idcol = (self.thresh_cols.get(thresh_type, self.col_com)
                         if not thresh_type.startswith("bw") else self.col_com)
                cv2.circle(img_draw, c, 0, idcol, self.centre_lwidth)
                if thresh_type.startswith("bw"):
                    draw_text(img_draw, str(id_),
                              (c[0] - 4, c[1] - 4), 0.3, "black", 0, 1)

    def draw_orient_link(self, img_draw, fulldat, frame_nr):
        """Draw orientation arrows linking colour contours to bw contours."""
        currframe_inds = [i for i, f in enumerate(fulldat["frame"]) if f == frame_nr]
        ids = [id_ for i, id_ in enumerate(fulldat["id"]) if i in currframe_inds]

        if "angle" not in fulldat:
            fulldat["angle"] = [None] * len(fulldat["frame"])

        if not ids or len(ids) != 2 * len([i for i in ids if type(i) == str]):
            return

        coms = [(fulldat["cx"][i], fulldat["cy"][i]) for i in currframe_inds]
        colids, colcoms = zip(*[(ids[i], com) for i, com in enumerate(coms)
                                if type(ids[i]) == str])
        _, bwcoms = zip(*[(ids[i], com) for i, com in enumerate(coms)
                          if type(ids[i]) != str])
        _, bw_inds = linear_sum_assignment(cdist(colcoms, bwcoms))

        for i, id_ in enumerate(colids):
            angle = int(points_to_angle(colcoms[i], bwcoms[bw_inds[i]], flip=True))
            target_index = currframe_inds[ids.index(id_)]
            if target_index >= len(fulldat["angle"]):
                fulldat["angle"].extend(
                    [None] * (target_index - len(fulldat["angle"]) + 1))
            fulldat["angle"][target_index] = angle
            arrowtip = get_coord(colcoms[i][0], colcoms[i][1], angle, 5, True)
            col = (255, 255, 255) if id_ in ["blue", "black"] else (0, 0, 0)
            cv2.arrowedLine(img_draw, colcoms[i], arrowtip, col,
                            self.orient_lwidth, tipLength=0.4)


# ------------------------------------------------------------------
# Module-level helper
# ------------------------------------------------------------------
class _FFmpegVideoWriter:
    """Adapt imageio's FFmpeg writer to the OpenCV write/release interface."""

    def __init__(self, outfile, fps):
        import imageio.v2 as imageio
        self.outfile = os.path.abspath(outfile)
        self.writer = imageio.get_writer(
            self.outfile, format="FFMPEG", fps=float(fps), codec="libx264",
            pixelformat="yuv420p", macro_block_size=1,
        )

    def write(self, frame):
        try:
            self.writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        except Exception as exc:
            raise RuntimeError(f"FFmpeg could not write video output {self.outfile}: {exc}") from exc

    def release(self):
        self.writer.close()


def _open_video_writer(outfile, dimensions, fps):
    """Use OpenCV when available, otherwise fall back to bundled FFmpeg."""
    try:
        writer = videowriter(outfile, dimensions[0], dimensions[1], fps)
        if writer is not None and writer.isOpened():
            return writer
        if writer is not None:
            writer.release()
    except (cv2.error, OSError) as exc:
        lineprint(f"OpenCV writer error: {exc}")
    lineprint("OpenCV MP4 writer unavailable; trying imageio/FFmpeg (H.264).")
    try:
        return _FFmpegVideoWriter(outfile, fps)
    except Exception as exc:
        raise RuntimeError(
            f"Could not open video output: {os.path.abspath(outfile)} "
            f"(width={dimensions[0]}, height={dimensions[1]}, fps={fps}, "
            f"OpenCV={cv2.__version__}). FFmpeg fallback failed: {exc}"
        ) from exc


def _prepare_video_frame(frame, frame_dims):
    """Validate rendered dimensions and pad odd MP4 edges without rescaling."""
    width, height = frame_dims
    if frame.shape[:2] != (height, width):
        raise ValueError(
            f"Rendered frame is {frame.shape[1]}x{frame.shape[0]}, "
            f"but video output expects {width}x{height}."
        )
    if width % 2 or height % 2:
        return cv2.copyMakeBorder(frame, 0, height % 2, 0, width % 2,
                                  cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return frame


def _build_frame_data(data, frame_nr, trajlength):
    """Build the frame_data dict for a single frame from the full DataFrame."""
    fd = data.loc[data["frame"] == frame_nr].to_dict("list")
    fd["trajdat"] = []
    if not fd.get("frame"):
        fd["frame"] = [frame_nr]
    else:
        tframes = list(range(frame_nr - trajlength + 1, frame_nr + 1))
        for id_ in fd.get("ID", []):
            fd["trajdat"].append(
                list(data.query("ID == @id_ & frame in @tframes")["com"]))
    return fd
