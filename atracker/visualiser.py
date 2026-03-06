#! /usr/bin/env python

import os
import cv2
import time

from pythutils.sysutils import lineprint
from pythutils.mediautils import crop, videowriter, add_transimg, imgresize
from pythutils.drawutils import namedcols, draw_text, draw_traj, uniqcols

from atracker.utils import *


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



def visualise(data, videofile, img_bg = None, img_mask = None, img_thresh=None,
              wallconts = None, roi = None,
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
   # Resize wall coordinates
    if roi is not None and wallconts is not None:
        #wallconts = [(pt[0]-roi[0][0],pt[1]-roi[0][1]) for pt in wallconts]
        wallconts = [[[(coord[0][0]-roi[0][0],coord[0][1]-roi[0][1])] for coord in cont] for cont in wallconts]

    # Mask stuff
    if cropimg and roi is not None and img_mask is not None:
        img_mask = crop(img_mask, roi[0], roi[1])
    if resize != 1 and img_mask is not None:
        img_mask = imgresize(img_mask, resize)

    # Set up for video writing
    if writevideo:
        outfile = os.path.splitext(videofile)[0]+videosuffix+".mp4"
        viddims = (vidw,vidh) if smoothresize or resize==1 else (int(vidw*resize),int(vidh*resize))
        vidoutdims = canvasdims if canvasdims is not None else viddims
        vidout = videowriter(outfile, vidoutdims[0], vidoutdims[1], fps)

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
               roi = None, wallconts=None, wallconts2=None,
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

        # 17) Draw segments
        #--------------------
        # TO ADD

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
        if drawframenr:
            draw_text(img_draw, str(list(framedat["frame"])[0]), (0,0), parframesize*resizeimg, margin=5, bgcol="white")

        # 21) Draw infobox
        #--------------------
        # TO ADD

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
