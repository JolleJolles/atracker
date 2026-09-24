#! /usr/bin/env python

import os
import sys
import glob
import shutil
import subprocess

import cv2
import imageio
import numpy as np
import pandas as pd

from pythutils.mathutils import seqcount


def make_even(x):
    return x if x % 2 == 0 else x + 1


def videowriter(filein, w, h, fps, codec="libx264", preset="ultrafast", crf=23):
    """Create an imageio FFmpeg writer with configurable encoder settings."""
    fileout = filein if filein.endswith(".mp4") else filein + ".mp4"
    try:
        return imageio.get_writer(
            fileout, fps=fps, codec=str(codec), macro_block_size=None,
            quality=None, ffmpeg_params=[
                "-crf", str(crf), "-preset", str(preset)])
    except Exception as exc:
        print(f"[ERROR] Could not create video writer for {fileout}: {exc}")
        return None


def convert_h264_to_mp4(indir, outdir=None, fps=24, overwrite=False):
    """
    Convert all .h264 files in a folder to .mp4.
    Supports fps as int, list (parallel with file list), or dict {basename: fps}.
    Tries fast system ffmpeg remuxing first; falls back to portable imageio if needed.
    """
    outdir = outdir or indir
    os.makedirs(outdir, exist_ok=True)
    files = glob.glob(os.path.join(indir, "*.h264"))

    if isinstance(fps, (list, pd.Series)):
        fps_dict = {os.path.splitext(os.path.basename(f))[0]: fval for f, fval in zip(files, fps)}
    elif isinstance(fps, dict):
        fps_dict = fps
    else:
        fps_dict = {}

    for filein in files:
        basename = os.path.splitext(os.path.basename(filein))[0]
        outfile = os.path.join(outdir, basename + ".mp4")

        if not overwrite and os.path.exists(outfile):
            print(f"[SKIP] Already exists: {basename}.mp4")
            continue

        this_fps = fps_dict.get(basename, fps)

        if shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-r", str(this_fps), "-i", filein, "-vcodec", "copy",
                   outfile, "-y", "-nostats", "-loglevel", "0"]
            try:
                subprocess.run(cmd, check=True)
                print(f"Converted: {basename}.h264 to .mp4 at {this_fps} fps")
                continue
            except subprocess.CalledProcessError:
                print(f"Failed to convert {basename}. Falling back to imageio...")

        try:
            reader = imageio.get_reader(filein, format='ffmpeg', fps=this_fps)
            writer = imageio.get_writer(outfile, format='ffmpeg', fps=this_fps,
                                        codec='libx264', macro_block_size=None,
                                        ffmpeg_params=['-crf', '23', '-preset', 'fast'])
            for frame in reader:
                writer.append_data(frame)
            reader.close()
            writer.close()
            print(f"[IMAGEIO] Converted: {basename}.h264 to .mp4 at {this_fps} fps")
        except Exception as e:
            print(f"[ERROR] Could not convert {basename}.h264: {e}")


def bg_extract(vidfile, start=None, stop=None, framenr=25):
    """Extracts a background image of a video."""
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
        print("Lost frames encountered, bgfile created from " + frnr + " files..")
    else:
        print("Done")

    img_bg = np.median(frames, axis=0).astype(dtype=np.uint8)
    return img_bg


def get_media_type(source):
    ext = os.path.splitext(str(source))[1].lower()
    if ext in [".mov", ".mp4", ".avi"]:
        return "vid"
    if ext in [".jpg", ".png", ".jpeg", ".bmp"]:
        return "img"
    if isinstance(source, int):
        return "stream"
    return None


def showvideo(videofile, wait=False):
    cap = cv2.VideoCapture(videofile)
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    waitkey = 0 if wait else 1
    while cap.isOpened():
        frameOK, img = cap.read()
        frame_nr = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.imshow("Video", img)
        key = cv2.waitKey(waitkey) & 0xff
        if key == 27:
            break
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def framechecks(cap, frameOK, framelist=None, stopframe=999999, displaystep=100, identifier=""):
    frame_nr = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if frame_nr % displaystep == 0:
        if identifier == "":
            print(frame_nr, end=" ")
        else:
            print("[" + str(identifier) + ":" + str(frame_nr) + "]", end="")
        sys.stdout.flush()
    skip = False if frameOK else True
    if framelist is not None:
        if frame_nr not in framelist:
            skip = True
    stop = False
    if frame_nr > stopframe or (not frameOK and frame_nr > stopframe - 10):
        print(" ")
        stop = True
    return stop, skip, frame_nr


def find_max_working_pyframe(cap):
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_number in range(total_frames - 1, -1, -1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            return frame_number
    print("No valid frames found")
    return 0
