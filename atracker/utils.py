#! /usr/bin/env python

import re
import os 
import cv2
import sys
import glob
import math
import shutil
import colorsys

import signal
import imageio

import subprocess
from screeninfo import get_monitors

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype
from ast import literal_eval
from collections import defaultdict
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter
from shapely.geometry import Point, Polygon

from PyQt5.QtGui import QImage, QColor
from PyQt5.QtCore import QPoint

from pythutils.datutils import contour_to_tuple
from pythutils.drawutils import namedcols
from pythutils.fileutils import listfiles
from pythutils.mathutils import points_to_angle, angle_to_vec, get_weights, ptsToDist, seqcount

import threading
from typing import Callable


def get_filepart(dir, sep="_", part=2, remove_ext=True, ext=None):
    """Extract a specific part of filenames in a directory"""
    files = listfiles(dir)
    out = []

    for f in files:
        if ext is not None:
            if isinstance(ext, str):
                if not f.endswith(ext):
                    continue
            else:
                if not f.endswith(tuple(ext)):
                    continue

        name = os.path.splitext(f)[0] if remove_ext else f
        parts = name.split(sep)
        if len(parts) >= part:
            out.append(parts[part - 1])  # 1-based index to 0-based
        else:
            out.append(None)

    return out


def qimg_to_grayscale(qimg):
    """Convert a QImage to grayscale using OpenCV."""
    qimg = qimg.convertToFormat(QImage.Format_RGBA8888)
    width = qimg.width()
    height = qimg.height()
    ptr = qimg.bits()
    ptr.setsize(qimg.byteCount())
    arr = np.array(ptr).reshape(height, width, 4)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bgr2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    rgba2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGBA)
    return QImage(rgba2.data, width, height, qimg.bytesPerLine(), QImage.Format_RGBA8888).copy()


def dist_to_target(xs, ys, target, method="point", **kwargs):
    if method == "point":
        return dist_to_point(xs, ys, *target)
    elif method == "rect":
        return dist_to_rect(xs, ys, *target)
    elif method == "poly":
        return dist_to_poly(xs, ys, target)
    elif method == "mask":
        return dist_to_mask(xs, ys, target, kwargs.get("conv", 1.0))
    else:
        raise ValueError("Unknown method")
    
def dist_to_point(xs, ys, cx, cy):
    return np.hypot(xs - cx, ys - cy)

def calc_borderdist(pt, roi):
    x, y = pt
    xmin, ymin = roi[0]
    xmax, ymax = roi[1]
    return dist_to_rect(np.array([x]), np.array([y]), xmin, xmax, ymin, ymax)[0]

def calc_borderdistdf(df, roi):
    cx = df['cx'].to_numpy()
    cy = df['cy'].to_numpy()
    xmin, ymin = roi[0]
    xmax, ymax = roi[1]
    return dist_to_rect(cx, cy, xmin, xmax, ymin, ymax)

def dist_to_rect(xs, ys, xmin, xmax, ymin, ymax):
    # positive outside, negative inside
    dx = np.maximum(np.maximum(xmin - xs, 0), xs - xmax)
    dy = np.maximum(np.maximum(ymin - ys, 0), ys - ymax)

    outside_dist = np.hypot(dx, dy)

    inside = (xs >= xmin) & (xs <= xmax) & (ys >= ymin) & (ys <= ymax)

    if np.any(inside):
        min_dist_inside = np.minimum.reduce([
            xs[inside] - xmin,
            xmax - xs[inside],
            ys[inside] - ymin,
            ymax - ys[inside]
        ])
        outside_dist[inside] = -min_dist_inside

    return outside_dist

def dist_to_poly(xs, ys, poly_coords):
    poly = Polygon(poly_coords)
    pts = [Point(x, y) for x, y in zip(xs, ys)]
    dists = np.array([poly.exterior.distance(pt) for pt in pts])
    inside = np.array([poly.contains(pt) for pt in pts])
    dists[inside] = -dists[inside]
    return dists

def dist_to_mask(xs, ys, mask, conv=1.0):
    """
    Signed distance to mask: negative if inside, positive if outside, 0 on edge.
    xs, ys: coordinates in mask pixel space!
    mask: binary (uint8) mask
    conv: pixel to mm conversion
    """
    h, w = mask.shape
    xs_ = np.round(xs).astype(int)
    ys_ = np.round(ys).astype(int)
    valid = (xs_ >= 0) & (xs_ < w) & (ys_ >= 0) & (ys_ < h)
    dists = np.full(xs.shape, np.nan)
    inside = np.zeros(xs.shape, dtype=bool)
    inside[valid] = mask[ys_[valid], xs_[valid]] > 0

    # 1. Find mask edge points using OpenCV
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours and len(contours[0]) > 0:
        edge_pts = np.vstack(contours).squeeze()  # shape (N, 2)
        # edge_pts in (x, y) order
        tree = KDTree(edge_pts)
    else:
        edge_pts = None
        tree = None

    # 2. Outside: distance to nearest mask pixel
    outside_idx = np.where(~inside & valid)[0]
    if len(outside_idx):
        ys_mask, xs_mask = np.where(mask > 0)
        if len(xs_mask):
            tree_mask = KDTree(np.column_stack([xs_mask, ys_mask]))
            for idx in outside_idx:
                pt = [xs_[idx], ys_[idx]]
                dists[idx] = tree_mask.query(pt)[0] * conv

    # 3. Inside: negative distance to nearest edge point
    inside_idx = np.where(inside & valid)[0]
    if len(inside_idx) and edge_pts is not None:
        for idx in inside_idx:
            pt = [xs_[idx], ys_[idx]]
            dists[idx] = -tree.query(pt)[0] * conv
    elif len(inside_idx):
        dists[inside_idx] = 0  # fallback if no edge found

    return dists

def is_axis_aligned_rectangle(coords):
    coords = np.asarray(coords)
    if coords.shape[0] != 4:
        return False
    xs, ys = coords[:,0], coords[:,1]
    # Check for two unique x's and two unique y's (axis-aligned)
    return (len(np.unique(xs)) == 2 and len(np.unique(ys)) == 2)

def valid_img_path(fileinfo, key, originals_dir):
    """
    Returns full image path if column exists, is not nan/empty, and file exists, else None.
    """
    # column exists, value is not nan, not empty string, not whitespace
    val = getattr(fileinfo, key, None)
    if val is not None and isinstance(val, str) and val.strip() and val.strip().lower() != "nan":
        fpath = os.path.join(originals_dir, val)
        if os.path.isfile(fpath):
            return fpath
    return None

def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
        colors.append(QColor(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)))
    return colors

def load_and_convert_tracking_dataframe(data_file, firstframe=None, lastframe=None):
    """
    Loads a tracking data CSV file, converts old formats to new, filters frames,
    and returns a standardized DataFrame with columns:
    frame, id, IDstr, cx, cy, hx, hy, tx, ty, angle

    id: integer (starting from 1)
    IDstr: original ID as string
    """
    import pandas as pd
    import numpy as np
    from ast import literal_eval

    df = pd.read_csv(data_file)
    df.columns = [c.lower() for c in df.columns]

    # --- Convert to new format if needed
    new_cols = {'frame', 'id', 'cx', 'cy', 'hx', 'hy', 'tx', 'ty', 'angle'}
    if not new_cols.issubset(df.columns):
        if {'com', 'angle'}.issubset(df.columns):
            # --- Old format (com + angle)
            def parse_point(s):
                try:
                    return literal_eval(str(s))
                except Exception:
                    return (np.nan, np.nan)
            new_df = pd.DataFrame()
            new_df['frame'] = df['frame']
            new_df['IDstr'] = df['id'].astype(str)
            new_df[['cx', 'cy']] = df['com'].apply(parse_point).apply(pd.Series)
            new_df['hx'] = np.nan
            new_df['hy'] = np.nan
            new_df['tx'] = np.nan
            new_df['ty'] = np.nan
            new_df['angle'] = pd.to_numeric(df['angle'], errors='coerce')
            df = new_df

        elif {'cx', 'cy'}.issubset(df.columns):
            # --- Simple format (cx, cy only)
            new_df = pd.DataFrame()
            new_df['frame'] = df['frame']
            new_df['IDstr'] = df['id'].astype(str)
            new_df['cx'] = df['cx']
            new_df['cy'] = df['cy']
            new_df['hx'] = np.nan
            new_df['hy'] = np.nan
            new_df['tx'] = np.nan
            new_df['ty'] = np.nan
            new_df['angle'] = np.nan
            df = new_df

        else:
            raise ValueError(f"Unrecognized tracking file format: {data_file}")
    else:
        # --- Already new format
        df['IDstr'] = df['id'].astype(str)

    # --- Filter frames if requested
    if firstframe is not None:
        df = df[df['frame'] >= firstframe]
    if lastframe is not None:
        df = df[df['frame'] <= lastframe]

    # Assign numeric IDs
    unique_ids = sorted(df['IDstr'].unique(), key=str)
    id_map = {name: i+1 for i, name in enumerate(unique_ids)}
    df['ID'] = df['IDstr'].map(id_map)

    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    return df.reset_index(drop=True)
    

def build_points_by_frame(df):
    """
    Converts standardized tracking DataFrame into points_by_frame dict:
    {id: {'c': {frame: QPoint}, 'h': {...}, ...}}
    Always uses df['ID'] as the ID key (int, 0-based).
    """
    frame_data = defaultdict(lambda: {'c': {}, 'h': {}, 't': {}, 'a': {}})
    for _, row in df.iterrows():
        id_val = int(row['ID'])
        frame = int(row['frame'])
        if pd.notnull(row.get('cx')) and pd.notnull(row.get('cy')):
            frame_data[id_val]['c'][frame] = QPoint(int(row['cx']), int(row['cy']))
        if pd.notnull(row.get('hx')) and pd.notnull(row.get('hy')):
            frame_data[id_val]['h'][frame] = QPoint(int(row['hx']), int(row['hy']))
        if pd.notnull(row.get('tx')) and pd.notnull(row.get('ty')):
            frame_data[id_val]['t'][frame] = QPoint(int(row['tx']), int(row['ty']))
        if pd.notnull(row.get('angle')):
            frame_data[id_val]['a'][frame] = float(row['angle'])
    return frame_data


def draw_arrow_head(painter, tip, size=10, angle_rad=0.0, angle_offset=math.pi / 8):
    # Draw two lines from 'tip' at ±angle_offset from 'angle_rad'
    for offset in [+angle_offset, -angle_offset]:
        x = tip.x() - int(size * math.cos(angle_rad + offset))
        y = tip.y() - int(size * math.sin(angle_rad + offset))
        painter.drawLine(tip, QPoint(x, y))


def get_media_type(source):
    ext = os.path.splitext(str(source))[1].lower()
    if ext in [".mov",".mp4",".avi"]:
        return "vid"
    if ext in [".jpg", ".png", ".jpeg", ".bmp"]:
        return "img"
    if isinstance(source, int):
        return "stream"
    return None

def ensure_columns(df, columns_with_defaults):
    for col, default in columns_with_defaults.items():
        if col not in df.columns:
            df[col] = default
    return df

def make_even(x): return x if x % 2 == 0 else x + 1

def videowriter(filein, w, h, fps):
    """Compact and safe video writer using imageio + ffmpeg."""
    fileout = filein if filein.endswith(".mp4") else filein + ".mp4"
    w, h = make_even(w), make_even(h)

    try:
        return imageio.get_writer(
            fileout,
            fps=fps,
            codec='libx264',
            macro_block_size=None,
            quality=8,
            ffmpeg_params=['-crf', '23', '-preset', 'fast']
        )
    except Exception as e:
        print(f"[ERROR] Could not create video writer for {fileout}: {e}")
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
    
    # If fps is a list or Series, convert to dict using basenames
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

        # Get fps for this file, default to input fps if not specified
        this_fps = fps_dict.get(basename, fps)

        # Try system ffmpeg
        if shutil.which("ffmpeg"):
            cmd = [
                "ffmpeg",
                "-r", str(this_fps),
                "-i", filein,
                "-vcodec", "copy",
                outfile,
                "-y",
                "-nostats",
                "-loglevel", "0"
            ]
            try:
                subprocess.run(cmd, check=True)
                print(f"Converted: {basename}.h264 to .mp4 at {this_fps} fps")
                continue
            except subprocess.CalledProcessError:
                print(f"Failed to convert {basename}. Falling back to imageio...")

        # Fallback: imageio re-encode
        try:
            reader = imageio.get_reader(filein, format='ffmpeg', fps=this_fps)
            writer = imageio.get_writer(
                outfile,
                format='ffmpeg',
                fps=this_fps,
                codec='libx264',
                macro_block_size=None,
                ffmpeg_params=['-crf', '23', '-preset', 'fast']
            )

            for frame in reader:
                writer.append_data(frame)

            reader.close()
            writer.close()
            print(f"[IMAGEIO] Converted: {basename}.h264 to .mp4 at {this_fps} fps")

        except Exception as e:
            print(f"[ERROR] Could not convert {basename}.h264: {e}")


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


def cvMatToQImage(frame):
    """Convert a BGR or grayscale OpenCV image to QImage (RGBA)."""
    if len(frame.shape) == 2:  # Grayscale
        h, w = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGBA)
    elif len(frame.shape) == 3:
        h, w, ch = frame.shape
        if ch == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        elif ch == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
    else:
        raise ValueError("Unsupported frame format for QImage conversion.")
    return QImage(frame.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()

def qimage_to_numpy(qimg, format=QImage.Format_RGBA8888, to_bgr=True):
    """Convert a QImage to a NumPy array. Supports grayscale and RGBA."""
    qimg = qimg.convertToFormat(format)
    width = qimg.width()
    height = qimg.height()
    ptr = qimg.bits()
    ptr.setsize(qimg.byteCount())

    if format == QImage.Format_Grayscale8:
        # Grayscale image: one channel
        arr = np.array(ptr).reshape(height, width)
        return arr
    else:
        # 4 channels (RGBA)
        arr = np.array(ptr).reshape(height, width, 4)
        if to_bgr:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        return arr

def numpy_to_qimage(arr, force_grayscale=False):
    """Convert a NumPy array (grayscale or BGR) to QImage."""
    if force_grayscale:
        if len(arr.shape) == 3 and arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        if len(arr.shape) != 2:
            raise ValueError("Expected a 2D array for grayscale.")
        h, w = arr.shape
        return QImage(arr.data, w, h, w, QImage.Format_Grayscale8).copy()
    
    if len(arr.shape) == 2:  # Grayscale
        h, w = arr.shape
        return QImage(arr.data, w, h, w, QImage.Format_Grayscale8).copy()
    elif len(arr.shape) == 3 and arr.shape[2] == 3:  # BGR
        arr_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        h, w, ch = arr_rgb.shape
        return QImage(arr_rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    else:
        raise ValueError("Unsupported array shape for QImage conversion.")


def start_powermate_listener(
    vendor_id: int = 1917,
    product_id: int = 1040,
    update_key_function: Callable[[int], None] = None,
    running_flag: threading.Event = None,
    leftkey: str = "e",
    rightkey: str = "r",
    pressedleftkey: str = "w",
    pressedrightkey: str = "t"
) -> threading.Thread:
    """
    Starts a thread to listen for PowerMate input and updates the key value.
    """
    import hid
    import time

    def powermate_loop():
        try:
            # Open the PowerMate device
            device = hid.Device(vendor_id, product_id)
            print("PowerMate connected!")
            pressed = False

            while running_flag.is_set():
                try:
                    # Non-blocking read
                    data = device.read(64)  # Read input
                    if data:
                        press_noticed = data[0] == 0x01
                        rotation = data[1]
                        if press_noticed:
                            pressed = not pressed # change state
                        if pressed and rotation == 0xff:
                            key = pressedleftkey
                        if pressed and rotation == 0x01:
                            key = pressedrightkey
                        if not pressed and rotation == 0xff:  
                            key = leftkey 
                        if not pressed and rotation == 0x01: 
                            key = rightkey
                        update_key_function(ord(key))

                except OSError:
                    print(f"Read error: {e}")
                    if not running_flag.is_set():
                        break
                time.sleep(0.01)  # Prevent tight loop
        except Exception as e:
            print(f"Error in PowerMate loop: {e}")
        finally:
            if 'device' in locals():
                device.close()
                print("PowerMate disconnected.")

    listener_thread = threading.Thread(target=powermate_loop, daemon=True)
    listener_thread.start()
    return listener_thread


def stop_powermate_listener(listener_thread: threading.Thread, running_flag: threading.Event):
    """
    Stops the PowerMate listener thread.

    Parameters:
    - listener_thread: The thread returned by `start_powermate_listener`.
    - running_flag: The threading.Event controlling the loop.
    """
    print("Stopping PowerMate listener...")
    running_flag.clear()  # Signal the thread to stop
    if listener_thread and listener_thread.is_alive():
        listener_thread.join()  # Wait for the thread to finish
    print("PowerMate listener stopped.")


def get_screen_dimensions():
    """Fallback method to get screen dimensions using system_profiler."""
    try:
        output = subprocess.check_output(["system_profiler", "SPDisplaysDataType"])
        output = output.decode("utf-8")
        for line in output.split("\n"):
            if "Resolution" in line:
                # Use regex to extract dimensions (e.g., 5120 x 2880)
                match = re.search(r"(\d+)\s*x\s*(\d+)", line)
                if match:
                    width, height = map(int, match.groups())
                    return (width, height)
        # If no resolution line is found
        print("No resolution line found in system_profiler output.")
        return (None, None)
    except Exception as e:
        print(f"Error using fallback method: {e}")
        return (None, None)


def get_screen_resolution():
    """Try screeninfo first, fallback to system_profiler if necessary."""
    try:
        monitor = get_monitors()[0]
        return (monitor.width, monitor.height)
    except (ImportError, IndexError) as e: #removed ScreenInfoError
        return get_screen_dimensions()


def create_hue_gradient(height=10):
    # Create an array where hue values vary from 0 to 179 across the width
    hue_gradient = np.zeros((height, 180, 3), dtype=np.uint8)

    # Set hue values from 0 to 179, saturation and value to maximum
    hue_gradient[..., 0] = np.arange(0, 180)  # Hue from 0 to 179
    hue_gradient[..., 1] = 255  # Full saturation
    hue_gradient[..., 2] = 255  # Full value (brightness)

    # Convert the HSV image to BGR for display
    hue_gradient_bgr = cv2.cvtColor(hue_gradient, cv2.COLOR_HSV2BGR)

    return hue_gradient_bgr


def find_max_working_pyframe(cap):
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_number in range(total_frames - 1, -1, -1):  # start from the last frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # Seek to the frame
        ret, frame = cap.read()  # Try reading the frame
        if ret:  # If frame is read successfully
            return frame_number  # Return the frame number of the last working frame
    print("No valid frames found")
    return 0  # In case no valid frames are found, return 0


def notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def subdic(dic, inds):
    return {k:[j for i,j in enumerate(dic[k]) if i in inds] for k, v in dic.items()}


def duplicate_row(df, index, number):

    row_to_duplicate = pd.DataFrame([df.loc[index]] * (number-1), columns=df.columns)
    row_to_duplicate['region'] = range(2, number + 1)
    df = pd.concat([df, row_to_duplicate], ignore_index=True)    

    return df


def nextvec(prevpoint, currpoint, delay):
    displ = ptsToDist(prevpoint, currpoint) / delay
    angle = points_to_angle(prevpoint, currpoint)
    return get_coord(0, 0, angle, displ, astuple=True) if displ>0.5 else (0,0)


def concom(contour):
    M = cv2.moments(contour)
    cX = int(M['m10']/M['m00'])
    cY = int(M['m01']/M['m00'])
    return (cX,cY)


def filledconcom(contour):
    x,y = zip(*contoarray(contour))
    cX = intproper(np.mean(x))
    cY = intproper(np.mean(y))
    return (cX,cY)


def consplit(prevarrays, mergedarray):

    dists, ids = KDTree(prevarrays[0]).query(mergedarray)
    dists2, ids2 = KDTree(prevarrays[1]).query(mergedarray)

    distlist = [[i, dists[i], ids[i], dists2[i], ids2[i]] for i,_ in enumerate(dists)]
    mindis = [abs(distlist[i][1]-distlist[i][3]) for i,_ in enumerate(distlist)]
    distlist = [row + [mindis[i]] for i,row in enumerate(distlist)]
    distlist.sort(key=lambda x: int(x[5]), reverse=True)
    conids = []
    prevratio = len(prevarrays[0])/(len(prevarrays[0])+len(prevarrays[1]))
    for i,_ in enumerate(distlist):
        connrs = [1,2] if distlist[i][1] < distlist[i][3] else [2,1]
        if connrs[0]==2:
            prevratio=1-prevratio
        lenthresh = int(prevratio*len(mergedarray))
        conids += [connrs[0] if conids.count(connrs[0])<=lenthresh else connrs[1]]

    newcon1 = [[list(mergedarray[distlist[i][0]]),distlist[i][0]] for i,j in enumerate(conids) if j==1]
    newcon1.sort(key=lambda x: int(x[1]))
    newcon1 = arraytocon([i[0] for i in newcon1])
    newcon1 = concoords(newcon1, True, False)
    newcon1 = arraytocon([list(pt[0]) for pt in newcon1 if tuple(pt[0]) in mergedarray])
    newcon1 = concoords(newcon1, False, False)

    newcon2 = [[list(mergedarray[distlist[i][0]]),distlist[i][0]] for i,j in enumerate(conids) if j==2]
    newcon2.sort(key=lambda x: int(x[1]))
    newcon2 = arraytocon([i[0] for i in newcon2])
    newcon2 = concoords(newcon2, True, False)
    newcon2 = arraytocon([list(pt[0]) for pt in newcon2 if tuple(pt[0]) in mergedarray])
    newcon2 = concoords(newcon2, False, False)

    newcons = [newcon1] if len(newcon2)==0 else (newcon1,newcon2)

    return np.array(newcons)


def roifromcon(contour, pan=5):
    x = [i[0][0] for i in contour]
    y = [i[0][1] for i in contour]
    tl = (max(1,min(x)-pan),max(1,min(y)-pan))
    br = (max(x)+pan,max(y)+pan)
    w = br[0]-tl[0]
    h = br[1]-tl[1]
    return tl,br,w,h


def getavgvel(coordlist, forward=True):
    x,y = zip(*coordlist)
    vellist = list(calcudiff(pd.Series(x), pd.Series(y), period=1))
    if not forward:
        vellist = vellist[::-1]
    avgvel = weightedavg(vellist,3) if len(vellist)>3  else np.nan
    return avgvel


def concoords(contour, filled=True, array=True):
    tl,br,w,h = roifromcon(contour)
    canvas = np.zeros((h,w), np.uint8)
    cv2.polylines(canvas, [contour-tl], 0, 255, 1)
    canvas = cv2.blur(canvas,(3,3))
    canvas = cv2.erode(canvas, np.ones((3,3),np.uint8))
    con,_ = cv2.findContours(canvas, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
    if not filled:
        out = contoarray(con[0]+tl) if array else con[0]+tl
    else:
        canvas = np.zeros((h,w), np.uint8)
        cv2.fillPoly(canvas, [con[0]], 255)
        out = [(x+tl[0],y+tl[1]) for y,row in enumerate(canvas) for x,col in enumerate(row) if col==255]
        if not array:
            out = arraytocon(out)
    return out


def contoarray(contour):
    return np.array([list(pt[0]) for pt in contour])


def arraytocon(array):
    return np.array([[list([pt][0])] for pt in array])


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


def getouterpts(img):
    horpts = [pt for x in range(1,img.shape[1]) for pt in [(x,1),(x,img.shape[0])]]
    verpts = [pt for y in range(1,img.shape[0]) for pt in [(1,y),(img.shape[1],y)]]
    return sorted(horpts+verpts, key=lambda k: [k[0], k[1]])


def loadmask(maskfile):
    try:
        img_mask = cv2.imread(maskfile, 0)
        kernel = np.ones((5,5),np.uint8)
        img_mask = cv2.erode(img_mask, kernel)
        img_mask = cv2.dilate(img_mask, kernel)
    except:
        img_mask = None
    return img_mask


def coordsfrommask(maskfile, epsilon=4.0):
    """
    Given a mask file (filename or numpy array), return:
      - maskconts: the contours found,
      - maskcoords: a list of (x, y) coordinates from simplified contours.
    epsilon: how much to simplify (in pixels) — increase for simpler shapes.
    """
    if isinstance(maskfile, str):
        img_mask = cv2.imread(maskfile, 0)
    elif isinstance(maskfile, np.ndarray):
        img_mask = maskfile
        if len(img_mask.shape) == 3:
            img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
    else:
        return None, None

    try:
        img_maskinv = cv2.erode(img_mask, np.ones((5,5), np.uint8))
        img_maskinv = cv2.threshold(img_maskinv, 10, 255, cv2.THRESH_BINARY_INV)[1]
        maskconts, _ = cv2.findContours(img_maskinv, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # Simplify contours:
        maskcoords = []
        for cnt in maskconts:
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            maskcoords.extend([tuple(pt[0]) for pt in approx])
    except Exception as e:
        print("Mask error:", e)
        maskconts = None
        maskcoords = None
    return maskconts, maskcoords


def coordsfromzones(imgfile, palette_hues=None, tol=20, min_sat=200, min_val=200, epsilon=2.0):
    """
    For each zone (color) in the image, return a simplified polygon as a list of (x, y) tuples.
    """
    if palette_hues is None:
        palette_hues = list(range(0, 360, 36))
    img = cv2.imread(imgfile)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    zone_coords = {}
    for i, h in enumerate(palette_hues):
        lower = np.array([(h - tol)//2, min_sat, min_val])
        upper = np.array([(h + tol)//2, 255, 255])
        mask = cv2.inRange(img_hsv, lower, upper)
        # Find contours for this mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            # Use the largest contour (in case there are small artifacts)
            cnt = max(contours, key=cv2.contourArea)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            coords = [tuple(pt[0]) for pt in approx]
            # Remove duplicate endpoint if present
            if len(coords) > 2 and coords[0] == coords[-1]:
                coords = coords[:-1]
            if len(coords) >= 3:  # Only polygons with at least 3 points
                zone_coords[i+1] = coords
    return zone_coords


def framechecks(cap, frameOK, framelist=None, stopframe=999999, displaystep=100, identifier=""):
    frame_nr = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if frame_nr % displaystep == 0:
        if identifier == "":
            print(frame_nr, end=" ")
        else:
            print("["+str(identifier)+":"+str(frame_nr)+"]", end="")
        sys.stdout.flush()
    skip = False if frameOK else True
    if framelist is not None:
        if frame_nr not in framelist:
            skip = True
    stop = False
    if frame_nr > stopframe or (not frameOK and frame_nr > stopframe-10):
        print(" ")
        stop = True

    return stop, skip, frame_nr


def lit_converter(val):
    if pd.isnull(val):
        return (np.nan, np.nan)
    if isinstance(val, (tuple, list)):
        return val
    try:
        return literal_eval(val)
    except Exception:
        return (np.nan, np.nan)


def safe_literal_eval(val):
    try:
        # Only evaluate if it's a valid tuple-like string and not NaN
        if isinstance(val, str) and val.startswith("("):
            return literal_eval(val)
        elif pd.isna(val):
            return (np.nan, np.nan)  # Handle NaN values
        else:
            return val  # If it's not a tuple-like string, return it as is
    except (ValueError, SyntaxError):
        return (np.nan, np.nan)  # Return NaN tuple on error


def con_lathom(skeleton,contour):
    distmat = cdist(skeleton, [list(i[0]) for i in contour])
    minmat = [min(row) for row in distmat]
    lathom = round((np.max(minmat)-np.min(minmat))/len(minmat),3)
    return lathom


def draw_coordlist(img, coordlist, col="orange"):
    cols,rows = list(zip(*coordlist))
    img[rows,cols] = list(namedcols(col, BRG=False))
    return img


def geom_tocoord(geom):
    if geom.geom_type in ["GeometryCollection","MultiLineString","MultiPoint"]:
        if len(list(geom.geoms))==0:
            return np.nan
        geom = geom.geoms[0]
    if geom.geom_type == "Polygon":
        coord = geom.exterior.coords.xy
    else:
        if len(np.array(geom.coords))==0:
            return np.nan
        coord = geom.coords.xy
    return int(coord[0][0]),int(coord[1][0])


def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]


def dic_exclnan(dic, var1, var2=None):
    if var2 == None:
        var2 = var1
    inds = [i for i,j in enumerate(dic[var2]) if j==j]
    vals = [dic[var1][i] for i,j in enumerate(dic[var2]) if j==j]
    return inds, vals


def fix_roi(roi,resolution):
    try:
        pt1 = max(0,roi[0][0]),max(0,roi[0][1])
        pt2 = min(resolution[0],roi[1][0]),min(resolution[1],roi[1][1])
    except:
        pt1,pt2 = ((0,0),(resolution))
    return ((pt1,pt2))


def adjpt(pt, xypad, add=True):
    if add:
        pt = (pt[0]+xypad[0],pt[1]+xypad[1])
    else:
        pt = (pt[0]-xypad[0],pt[1]-xypad[1])
    return pt


def check_threshtypes(thresh_types):
    if isinstance(thresh_types, (np.floating, float)):
        thresh_types = ["bw"]
    if thresh_types[0] == "(":
        thresh_types = thresh_types[1:len(thresh_types)-1]
    if "," in thresh_types:
        thresh_types = thresh_types.split(',')
    if not isinstance(thresh_types, list):
        thresh_types = [thresh_types]
    return thresh_types


def eval_func_tuple(f_args):
    """Takes a tuple of a function and args, evaluates and returns result"""
    return f_args[0](*f_args[1:])


def minangle(angle, angle2):
    angle = angle2 - angle
    angle = abs((angle + 180) % 360 - 180)
    return angle


def intproper(x):
    i, f = divmod(x, 1)
    return int(i + ((f >= 0.5) if (x > 0) else (f > 0.5)))


def get_coord(x, y, angle, length, astuple = False):
    x2 = np.array(x + np.sin(np.radians(angle)) * length)
    y2 = np.array(y - np.cos(np.radians(angle)) * length)
    if astuple:
        coord = [np.nan if np.isnan(a) else (int(a),int(b)) for a,b in zip([x2],[y2])][0]
        return coord
    else:
        return x2,y2


def convert(x, y, conv, height, flip=True, roi=None, already_relative=False, decimals=3):
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    conv = float(conv)

    if roi is not None:
        x0, y0 = roi[0]
        x1, y1 = roi[1]
        if not already_relative:
            x = x - float(x0)
            y = y - float(y0)
        if flip:
            y = (float(y1) - float(y0)) - y
    else:
        if flip:
            y = float(height) - y

    x = np.round(x * conv, decimals)
    y = np.round(y * conv, decimals)
    return x, y


def hflipangle(angle):
    vx,vy = angle_to_vec(angle)
    angle = points_to_angle((vx,vy*-1))
    return angle


def hvflipangle(angle):
    if angle < 0:
        angle += 180.0
    else:
        angle -= 180.0
    return angle


def weightedavg(datlist, minlen=10):
    weights = get_weights(length=len(datlist))
    datlist = [val for i,val in enumerate(datlist) if val is not None and val==val]
    weights = [weights[i] for i,val in enumerate(datlist) if val is not None and val==val]
    weightedavg = round(np.average(datlist, weights = weights), 1) if len(datlist)>=minlen else np.nan
    return weightedavg


def get_head(x, y, angle, length, contour):
    fbx, fby = get_coord(x, y, angle, length * 0.5)
    contuple = contour_to_tuple(contour)
    _, pixid = KDTree(contuple).query((fbx, fby))
    fx, fy = contuple[pixid]
    return fx,fy


def compare_angle(angle, angle2):
    vect1 = (np.sin(np.radians(angle)), np.cos(np.radians(angle)))
    vect2 = (np.sin(np.radians(angle2)), np.cos(np.radians(angle2)))
    angle = hvflipangle(angle) if np.inner(vect1, vect2) < 0 else angle
    return angle


def series_to_point_tuple(df, cols):
    """
    Returns a tuple of NumPy arrays (x, y) from a DataFrame's specified columns.
    
    Parameters:
    - df: pandas DataFrame
    - cols: list or tuple of two strings, e.g., ["cx", "cy"]

    Returns:
    - (x_array, y_array): tuple of NumPy arrays
    """
    if len(cols) != 2:
        raise ValueError("cols must be a list or tuple of exactly two column names")
    x = pd.to_numeric(df[cols[0]], errors='coerce').to_numpy()
    y = pd.to_numeric(df[cols[1]], errors='coerce').to_numpy()
    return (x, y)


def differentiate(val, period=1):
    val = np.asarray(val, dtype=np.float64)
    diff = np.full_like(val, np.nan)
    if len(val) > period:
        diff[period:] = val[period:] - val[:-period]
    return diff

def calcudiff(x, y, period=1, angle=False):
    x = pd.to_numeric(x, errors='coerce').to_numpy()
    y = pd.to_numeric(y, errors='coerce').to_numpy()

    xdiff = differentiate(x, period)
    ydiff = differentiate(y, period)

    if angle:
        return np.arctan2(xdiff, ydiff) * 180 / np.pi
    else:
        return np.sqrt(xdiff**2 + ydiff**2)


def get_anglediff(angle):
    if type(angle) is not pd.core.series.Series:
        angle = pd.Series(angle)
    anglediff = angle-angle.shift(periods=1) if len(angle)>1 else angle
    anglediff.loc[anglediff < -180] = 360-abs(anglediff.loc[anglediff < -180])
    anglediff.loc[anglediff > 180] = (360-anglediff.loc[anglediff > 180])*-1
    return list(anglediff)


def anglediff(angle1, angle2):
    return (angle1 - angle2 + 180) % 360 - 180


def orientchecker(curr_frame, start_frame, prev_frame, delay, centre_coord, con_angle,
                  extdists, convexity, prev_angle, avg_vel, vel_thresh, prev_coord,
                  min_convex = 0.7, min_extdist_ratio = 0.1, max_anglechange = 30):

    # The object will have gotten an angle based on an ellipse of the
    # contour. If the contour is very convex, such as a fish in startle pose
    # the ellipse angle will be increasingly wrong so should be disregarded, but
    # in most cases it comes very close to the actual orientation of the object.
    # The angle is however arbitrary + or -, so should be confirmed. This can be
    # based on the previous angle, the movement angle, and the tip-to-tip angle.
    # We can be certain the movement angle is right (in terms of +/-) if the
    # object is moving at some speed that is not possible sideways or backwards.
    # We an be certain the tip-to-tip angle is correct when the ratio between
    # the two distances is large enough. A threshold of 10% seems good.

    # First, if the object is very convex, i.e. when the centroid lies towards
    # the edge or even outside an object, calculating the orientation on its
    # shape will give a false orientation and more complicated computations are
    # necessary
    convex_ok = convexity > min_convex

    # Second, if the ratio in distances to extreme points is very different from
    # 1, then we can be sure what is the the front, and use that coordinate.
    extdist_ratio = 1-(min(extdists)/max(extdists))
    distratio_ok = extdist_ratio >= min_extdist_ratio

    # Third, if much time has passed
    delay_ok = (curr_frame-prev_frame) <= delay

    # Fourth, the object is moving at significant speed, we can get a reliable
    # measure of the movement angle which should be close to being the same as
    # the orientation angle. But there can't have passed too much time, and
    # there should be enough data to reliable calculate the average speed
    if None in prev_coord:
        vel_ok = False
    else:
        if not delay_ok:
            vel_ok = False
        else:
            if avg_vel is None:
                vel_ok = False
            else:
                vel = ptsToDist(centre_coord,prev_coord)/(curr_frame-prev_frame)
                vel_ok = avg_vel >= vel_thresh and vel >= vel_thresh
                move_angle = points_to_angle(prev_coord, centre_coord, flip=True)

    # Scenario 1: shape is convex and front-back ratio is distinct
    # >>> We can be sure what is front based on centroid to closest extreme pt,
    # irrespective of objects speed or time delay
    if convex_ok and distratio_ok:
        if extdists[0]<extdists[1]:
            new_angle = con_angle
        else:
            new_angle = hvflipangle(con_angle)

    # Scenario 2: there is no distinct front-back ratio but the object passed
    # the speed threshold (convexity is irrelevant here)
    # >>> contour angle can be flipped if needed based on movement angle
    elif not distratio_ok and vel_ok:
        move_angle = points_to_angle(prev_coord, centre_coord, flip=True)
        new_angle = compare_angle(con_angle, move_angle)
        # Be careful for sudden flips in angle
        if minangle(prev_angle, new_angle) > max_anglechange:
            new_angle = hvflipangle(new_angle)

    # Scenario 3: the shape is convex, but there is no distinct front-back ratio,
    # the object did not pass the speed threshold, and we can use the previous angle
    # >>> Now we can use the previous stored angle if not too far back in time
    # and not nan to adjust the contour angle
    elif convex_ok and not distratio_ok and not vel_ok and delay_ok and not np.isnan(prev_angle):
        new_angle = compare_angle(con_angle, prev_angle)
        # Be careful for sudden flips in angle
        if minangle(prev_angle, new_angle) > max_anglechange:
            new_angle = hvflipangle(new_angle)

    # Scenario 4: the shape is convex, but there is no distinct front-back ratio,
    # and the object did not pass the speed threshold, and we cannot use the previous angle
    # >>> We cannot be certain enough of the angle, so it will be stored as nan
    # elif convex_ok and not distratio_ok and not vel_ok and delay_ok and np.isnan(prev_angle):
    else:
        new_angle = np.nan

    #print("convexity",convexity,convex_ok)
    #print("distratio",extdist_ratio,distratio_ok)
    #print("velocity",avg_vel,vel,vel_ok)
    return new_angle


def fixheadtail(series, areamultiplier = 1.75, distmultiplier = 10,
                seqlenthreshold = 40, thresharea=None):

    data = series.copy()
    swappedinds = []
    if thresharea is None:
        thresharea = np.nanmedian(data.area)

    # Calculate additional variables
    data["headdiff"] = round(calcudiff(data.fx, data.fy, period=1))
    data["taildiff"] = round(calcudiff(data.tx, data.ty, period=1))
    data["headangle"] = round(calcudiff(data.fx, data.fy, period=1, angle=True))
    data["tailangle"] = round(calcudiff(data.tx, data.ty, period=1, angle=True))
    tx,ty,fx,fy = (data["tx"].copy(),data["ty"].copy(),data["fx"].copy(),data["fy"].copy())
    data["fxsw"], data["fysw"] = (data.fx, data.fy)
    inds = np.arange(0,len(data.fx),2)
    data.loc[inds,"fxsw"],data.loc[inds,"fysw"] = (data.loc[inds,"tx"],data.loc[inds,"ty"])
    data["swapdiff"] = round(calcudiff(data.fxsw, data.fysw, period=1))
    data["anglediff"] = get_anglediff(data.orient)

    # We are interested in head position, so create list of all
    # head indices where time step is more than threshold
    indstocheck = data.index[(data["headdiff"] > data["swapdiff"]) | (data["taildiff"] > data["swapdiff"])]
    # There are potentially wrong head positions
    if len(indstocheck)>0:
        # Go through all indices
        for i,ind in enumerate(indstocheck):
            # Determine if not already used
            if ind in swappedinds:
                continue

            # If index is last of indices to check and near end of data
            # make NA, if not at end, make nearby heads and tails NA
            if i == len(indstocheck)-1:
                if len(indstocheck)==1:
                    # print("b", ind - min(data.index))
                    pass
#                    if ind - min(data.index) < 10:
#                        data.loc[range(min(data.index),ind+1),["head","fx","fy","tail","tx","ty"]] = np.nan
                elif (max(data.index) - ind) < seqlenthreshold:
                    data.loc[range(ind,max(data.index)+1),["head","fx","fy","tail","tx","ty"]] = np.nan
                else:
                    data.loc[range(max(ind-10,0),ind+10),["head","fx","fy","tail","tx","ty"]] = np.nan
                continue

            # Tail stays in the same position so set head to NA
            if data.loc[ind,"taildiff"] < (np.nanmedian(data["taildiff"])*distmultiplier):
                data.loc[ind,["head","fx","fy"]] = np.nan
                # But orientation angle does not change much so continue with next ind
                if data.loc[ind,"anglediff"]<20:
                    continue
            # Head stays in the same position so set tail to NA
            if data.loc[ind,"headdiff"] < (np.nanmedian(data["headdiff"])*distmultiplier):
                data.loc[ind,["tail","fx","fy"]] = np.nan
                # But orientation angle does not change much so continue with next ind
                if data.loc[ind,"anglediff"]<20:
                    continue

            # Both head and tail moved far because the object is moving fast
            # so no need for swapping
            if abs(anglediff(data.loc[ind,"headangle"],data.loc[ind,"tailangle"]))<90:
                continue

            # Sequence is abnormally long, therefore likely next point missing
            # therefore make frame and net 10 frames NA
            seqlen = indstocheck[i+1]-indstocheck[i]
            if(seqlen) > seqlenthreshold:
                data.loc[range(ind,ind+10),["head","fx","fy","tail","tx","ty"]] = np.nan
                continue

            # Check if the index and/or next one will be excluded because near
            # the edge
            if data.loc[ind,"orienttoexcl"] == 1:
                while True:
                     ind = ind+1
                     if ind>(len(data)-1) or data.loc[ind,"orienttoexcl"] != 1:
                         break

            toswap = range(ind+1,indstocheck[i+1]+1)
            if data.loc[indstocheck[i+1],"orienttoexcl"] == 1:
                data.loc[toswap,["head","fx","fy","tail","tx","ty"]] = np.nan

            data.loc[toswap,"fx"] = tx.loc[toswap]
            data.loc[toswap,"fy"] = ty.loc[toswap]
            data.loc[toswap,"tx"] = fx.loc[toswap]
            data.loc[toswap,"ty"] = fy.loc[toswap]
            swappedinds += toswap

    # # Head and tail positions are not reliable when the object area
    # # is considerably different from normal
    # if not np.all(data.area!=data.area):
    #     print("area1",len(data.loc[data.area<(thresharea/areamultiplier),["head","fx","fy","tail","tx","ty"]]))
    #     print("area2",len(data.loc[data.area>(thresharea*areamultiplier),["head","fx","fy","tail","tx","ty"]]))
    #     data.loc[data.area<(thresharea/areamultiplier),["head","fx","fy","tail","tx","ty"]] = np.nan
    #     data.loc[data.area>(thresharea*areamultiplier),["head","fx","fy","tail","tx","ty"]] = np.nan
    return data, swappedinds


def getindsections(fullinds, win = 0, gap = 1):

    """Get list of tracking sections with starting index, stopping index, and length"""

    inds = [fullinds[0]]+[fullinds[i] for i in list(range(1,len(fullinds))) if (fullinds[i]-fullinds[i-1]> gap)]
    relinds = [list(fullinds).index(i) for i in inds]+[len(fullinds)]
    fullinds2 = [min(fullinds)-1]+fullinds
    lengths = [fullinds2[relinds[i+1]]-fullinds[relinds[i]]+1 for i,val in enumerate(relinds[1:])]
    indsecs = [[ind, ind+lengths[i]-1, lengths[i]] for i,ind in enumerate(inds)]
    if win > 0:
        indsecs = [[indsec[0]-win,indsec[1]+win, indsec[2]+win+win] for indsec in indsecs]

    for i,_ in enumerate(indsecs):
        if indsecs[i][0] < 0:
            indsecs[i][0] = 0

    return indsecs


def getalones(dataset, var, win):

    """Get lists of tracked sections shorter than specified window"""

    temp = dataset.dropna(subset=[var]).copy()
    if len(temp)==0:
        alones = []
    elif len(temp)<=win:
        alones = temp.index.values
    else:
        inds = getindsections(list(temp.index.values))
        alinds = []
        for i,ind in enumerate(inds):
            if len(inds)==1:
                if ind[2]<win:
                    alinds = alinds+[ind]
            else:
                if ind[2]<win:
                    if i==0:
                        if (inds[i+1][0]-ind[1])>win:
                            alinds = alinds+[ind]
                    else:
                        if i<(len(inds)-1):
                            if ((ind[0]-inds[i-1][1]))>win and (inds[i+1][0]-ind[1])>win:
                                alinds = alinds+[ind]
                        else:
                            if (ind[0]-inds[i-1][1])>win:
                                alinds = alinds+[ind]
        alones = [item for i in alinds for item in list(range(i[0],i[0]+i[2]))]

    return alones


def findmissinginds(series, col, checkzero = False):

    """Output list of all indices with rows that contain missing values"""

    var = series[pd.isnull(list(series[col]))]
    missinds = [val for k,val in enumerate(var.index) if val > k]
    if checkzero:
        missinds = [val for k,val in enumerate(var.index) if val == k] + missinds if 0 in var.index else missinds
    return missinds


def checknearmask(series, coordcols, mask, indsecs, masksecs, win=5, nearmaskdis=50):
    nearlist = []
    awaylist = []

    for i, indsec in enumerate(indsecs):
        if (min(series.index.values) + 1) >= indsec[0]:
            beflen = win
        else:
            subset = series.loc[masksecs[i][0]:indsecs[i][0]-1, coordcols].dropna()
            array = np.stack((
                pd.to_numeric(subset[coordcols[0]], errors='coerce'),
                pd.to_numeric(subset[coordcols[1]], errors='coerce')
            ), axis=1)
            array = array[~np.isinf(array).any(axis=1)]
            befdis, _ = KDTree(mask).query(array)
            beflen = len([n for n in befdis if n < nearmaskdis or n == np.inf])

        if max(series.index.values) == indsec[1]:
            aftlen = win
        else:
            subset = series.loc[indsecs[i][1]+1:masksecs[i][1], coordcols].dropna()
            array = np.stack((
                pd.to_numeric(subset[coordcols[0]], errors='coerce'),
                pd.to_numeric(subset[coordcols[1]], errors='coerce')
            ), axis=1)
            array = array[~np.isinf(array).any(axis=1)]
            aftdis, _ = KDTree(mask).query(array)
            aftlen = len([n for n in aftdis if n < nearmaskdis or n == np.inf])

        if beflen > 0 and aftlen > 0:
            nearlist.append(i)
        else:
            awaylist.append(i)

    return nearlist, awaylist


def fillmissing(series, colpair, mask, roi, win=5, nearmaskdis=25, edgedis=10, lenthresh=100):
    """
    Fill-in missing data for specific numeric columns in a dataset

    First, missing data is ignored when at the start or end of the data series
    as cannot be filled. Second, sections with missing data near a mask are
    ignored as presumed object is missing because it is (partly) behind the
    mask. Third, missing data is ignored when caused by objects disappearing
    from view. Finally, missing values are calculated by differentiating.
    """

    # Fix new roi
    if roi is not None: 
        roi = ((1, 1), (roi[1][0] - roi[0][0], roi[1][1] - roi[0][1]))

    # Get missing indices
    missinds = findmissinginds(series, colpair[0], checkzero=False)
    missing = len(missinds)
    if missing > 0:
        indsecs = getindsections(missinds)

        # Remove starts and stops
        indsecs = [ind for ind in indsecs if ind[0] >= (min(series.index.values)+1) and ind[1] != max(series.index.values)]
        missinds = [item for i in indsecs for item in list(range(i[0], i[0] + i[2]))]
        missing = len(missinds)

        if missing > 0:
            diffsecs = getindsections(missinds, win=1)

            # Subset ind sections to only those not caused by blobs near a mask
            if mask not in (None, []):
                masksecs = getindsections(missinds, win=win)
                nearlist, awaylist = checknearmask(series, colpair, mask, indsecs, masksecs, nearmaskdis=nearmaskdis)
                indsecs = [indsec for i, indsec in enumerate(indsecs) if (indsec[2] < 5 and i in nearlist) or i in awaylist]
                diffsecs = [diffsec for i, diffsec in enumerate(diffsecs) if (diffsec[2] < (5 + 2) and i in nearlist) or i in awaylist]
                missing = sum([item[2] for item in indsecs])

        # Label "inroi" as 0 when fish is near the border both before and after
        if missing > 0 and "cx" in colpair and roi is not None:
            dellist = []
            for i, indsec in enumerate(indsecs):
                befxy = (series.at[indsec[0] - 1, colpair[0]], series.at[indsec[0] - 1, colpair[1]])
                aftxy = (series.at[indsec[1] + 1, colpair[0]], series.at[indsec[1] + 1, colpair[1]])
                befout = calc_borderdist(befxy, roi)
                aftout = calc_borderdist(aftxy, roi)
                if befout < edgedis and aftout < edgedis:
                    series.loc[indsec[0]:indsec[1], "inroi"] = 0
                    dellist.append(i)
            indsecs = [ind for i, ind in enumerate(indsecs) if i not in dellist]
            diffsecs = [ind for i, ind in enumerate(diffsecs) if i not in dellist]
            missing = sum([item[2] for item in indsecs])

    # Now calculate missing values by interpolation
    if missing > 0:
        for i, indsec in enumerate(indsecs):
            if indsec[2] > lenthresh:
                missing -= indsec[2]
            else:
                for col in colpair:
                    diffsec = diffsecs[i]
                    newvals = np.around(np.linspace(series[col][diffsec[0]], series[col][diffsec[1]], diffsec[2]), 3)[1:diffsec[2]-1]
                    series.loc[indsec[0]:indsec[1], col] = newvals

    return series, missing

def process_trajectories(series, mask=None, cover=True, win=5,
                         trajgap=50, inmaskdis=10, mintrajlength=10,
                         erase_coords=True, interpolate=True, force_single_traj=False):
    """
    Process trajectories: detects mask-covered segments, removes data near mask,
    assigns trajectory IDs, interpolates gaps, and removes short trajectories.

    Parameters
    ----------
    series : pd.DataFrame
        Tracking data with at least 'cx', 'cy', and 'frame' columns.
    mask : list of (x, y)
        Polygon points of the mask (optional).
    cover : bool
        Whether to use mask to detect cover state.
    win : int
        Buffer window around missing data when testing mask proximity.
    trajgap : int
        Max frame gap to join points into the same trajectory.
    inmaskdis : int
        Distance threshold to count as "near mask".
    mintrajlength : int
        Minimum number of points to keep a trajectory.
    erase_coords : bool
        Whether to blank coordinates under the mask.
    interpolate : bool
        Whether to fill small gaps within trajectories.

    Returns
    -------
    series : pd.DataFrame
        Modified tracking data with 'traj' and 'inmask' columns.
    ntraj : int
        Number of valid trajectories.
    nremoved : int
        Number of rows removed due to being near/in the mask.
    """
    import numpy as np
    import pandas as pd
    from scipy.spatial import KDTree
    from shapely.geometry import Point, Polygon

    series["inmask"] = 0
    series["traj"] = np.nan

    # Ensure cx/cy are numeric
    series["cx"] = pd.to_numeric(series["cx"], errors="coerce")
    series["cy"] = pd.to_numeric(series["cy"], errors="coerce")

    ## 1. Detect which rows are "in mask" and optionally remove coordinates
    removed_mask = np.zeros(len(series), dtype=bool)
    if cover and mask and len(mask) > 0:
        polygon = Polygon(mask)
        valid = series[["cx", "cy"]].dropna().copy()
        points = list(valid.itertuples(index=True, name=None))
        locs = [(x[1], x[2]) for x in points]
        indices = [x[0] for x in points]

        distances, _ = KDTree(mask).query(locs)
        inside = [polygon.contains(Point(x, y)) for x, y in locs]
        removed_mask[indices] = (np.array(inside) | (distances < inmaskdis))
        series.loc[removed_mask, "inmask"] = 1

        if erase_coords:
            series.loc[removed_mask, ["cx", "cy"]] = np.nan

    ## 2. Determine "non-masked" points
    nonmiss = series.index[series["cx"].notna() & (series["inroi"] != 0)]

    ## 3. Group into trajectory segments
    def getindsections(idxs, gap=1):
        """Split sorted indices into continuous segments with max gap."""
        if not len(idxs): return []
        idxs = sorted(idxs)
        breaks = [0] + [i+1 for i in range(len(idxs)-1) if idxs[i+1] - idxs[i] > gap] + [len(idxs)]
        return [(idxs[start], idxs[end-1]) for start, end in zip(breaks[:-1], breaks[1:])]

    sections = getindsections(nonmiss, gap=trajgap)
    # --- FORCE SINGLE TRAJECTORY MODE ---
    if force_single_traj:
        if len(nonmiss) > 0:
            start, end = nonmiss[0], nonmiss[-1]
            sections = [(start, end)]
        else:
            sections = []
            
    for i, (start, end) in enumerate(sections, start=1):
        series.loc[start:end, "traj"] = i

        if interpolate:
            inds = series.loc[start:end].index
            for col in ["cx", "cy"]:
                y = series.loc[inds, col]
                if y.isna().any():
                    x = y.index
                    filled = y.interpolate(method='linear', limit_direction='both')
                    series.loc[x, col] = filled

    ## 4. Remove short trajectories
    nremoved = 0
    for trajid, group in series.groupby("traj"):
        if pd.isna(trajid):
            continue
        if len(group) < mintrajlength:
            series.loc[group.index, ["cx", "cy", "traj"]] = np.nan
            nremoved += len(group)

    ## 5. Reindex trajectories
    valid_trajids = [i for i in series.traj.unique() if not pd.isna(i)]
    for new_id, old_id in enumerate(valid_trajids, start=1):
        series.loc[series.traj == old_id, "traj"] = new_id

    ntraj = len(valid_trajids)
    return series, ntraj, int(removed_mask.sum())


def smooth(series, trajs=None, columns=("cx","cy"), smoothwin=5, polyorder=3):
    for col in columns:
        if col in series.columns and not is_float_dtype(series[col].dtype):
            series[col] = pd.to_numeric(series[col], errors="coerce").astype("float64")

    if trajs is None:
        if len(series) > 6:
            for column in columns:
                if column not in series.columns:
                    continue
                nonan = series[column].dropna()
                nonanlen = len(nonan)
                win = smoothwin if nonanlen > smoothwin else nonanlen
                if win > 6:
                    win = win - 1 if win % 2 == 0 else win  # make odd
                    # Also ensure win > polyorder
                    if win <= polyorder:
                        win = polyorder + 2 + ((polyorder + 2) % 2)
                        if win > nonanlen:
                            continue
                    # Filter only valid (non-NaN) positions, then write back
                    filt = savgol_filter(nonan.to_numpy(dtype=float), polyorder=polyorder, window_length=win)
                    series.loc[nonan.index, column] = filt
    else:
        for t in trajs:
            sub = series.loc[series.traj == t]
            if len(sub) > 6:
                for column in columns:
                    if column not in series.columns:
                        continue
                    nonan = sub[column].dropna()
                    nonanlen = len(nonan)
                    win = smoothwin if nonanlen > smoothwin else nonanlen
                    if win > 6:
                        win = win - 1 if win % 2 == 0 else win  # make odd
                        if win <= polyorder:
                            win = polyorder + 2 + ((polyorder + 2) % 2)
                            if win > nonanlen:
                                continue
                        filt = savgol_filter(nonan.to_numpy(dtype=float), polyorder=polyorder, window_length=win)
                        series.loc[nonan.index, column] = filt

    return series

def filter_tracking_jumps(ids, coms, frame_nr, last_valid, max_framedist=200, max_gap=10, pr_comm=""):
    """
    Accept point only if (1) within max_gap frames from last_valid and (2) within max_framedist * gap.
    After max_gap, only accept points within max_framedist (i.e., don't keep growing allowed jump forever).
    """
    filtered_coms = []
    for idx, id in enumerate(ids):
        c = coms[idx]
        if c is None or not isinstance(c, (tuple, list)) or any([ci != ci for ci in c]):
            filtered_coms.append((np.nan, np.nan))
            continue
        if id in last_valid:
            prev_x, prev_y, prev_frame = last_valid[id]
            gap = frame_nr - prev_frame
            if gap <= max_gap:
                allowed_jump = max_framedist * gap
            else:
                allowed_jump = max_framedist  # after max_gap, only allow within fixed distance
            dist = np.linalg.norm([c[0] - prev_x, c[1] - prev_y])
            #print(f"[{pr_comm}] Frame {frame_nr}, ID {id}: jump {dist:.1f} > {allowed_jump}")
            if dist > allowed_jump:
                filtered_coms.append((np.nan, np.nan))
                # Crucially: do NOT update last_valid here!
                continue
        filtered_coms.append((c[0], c[1]))
        last_valid[id] = (c[0], c[1], frame_nr)
    return filtered_coms, last_valid
