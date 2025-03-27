#! /usr/bin/env python

import os
import cv2
import numpy as np
from pythutils.mathutils import seqcount

def bg_extract(vidfile, start=None, stop=None, framenr=25):

    """Extracts a background image of a video"""

    if not os.path.splitext(vidfile)[1] == ".mp4":
        print("Video needs to be .mp4")
        return
    cap = cv2.VideoCapture(vidfile)
    if not cap.isOpened():
        print("Video source failed to open..")
        return
    flag, frame = cap.read()
    if not flag:
        print("Video source opened but failed to read any images..")
        return

    start = 1 if start is None else start
    stop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if stop is None else stop
    framelist = seqcount(start, stop, framenr)
    frames = []

    print("Extracting bg image..", end=" ")
    print(start, stop, framenr, end=" ")
    for frameloc in framelist:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameloc)
        flag, frame = cap.read()
        if flag:
            frames.append(frame)

    if len([f for f in frames if f is not None]) < framenr:
        frnr = str(len(frames))
        print("Lost frames encountered, bgfile created from "+frnr+" files..")
    else:
        print("Done")

    img_bg = np.median(frames, axis=0).astype(dtype=np.uint8)

    return img_bg
