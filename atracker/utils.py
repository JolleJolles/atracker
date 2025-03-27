#! /usr/bin/env python

import re 
import cv2
import sys
import signal

import subprocess
from screeninfo import get_monitors

import numpy as np
import pandas as pd
from ast import literal_eval
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter

from pythutils.datutils import contour_to_tuple
from pythutils.drawutils import namedcols
from pythutils.mathutils import points_to_angle, angle_to_vec, get_weights, ptsToDist

import threading
from typing import Callable


def qimage_to_numpy(qimg):
    """Convert a QImage (Grayscale8) to a NumPy array."""
    qimg = qimg.convertToFormat(QImage.Format_Grayscale8)
    width = qimg.width()
    height = qimg.height()
    ptr = qimg.bits()
    ptr.setsize(qimg.byteCount())
    arr = np.array(ptr).reshape(height, width)
    return arr

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
        lentresh = int(prevratio*len(mergedarray))
        conids += [connrs[0] if conids.count(connrs[0])<=lentresh else connrs[1]]

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


def coordsfrommask(maskfile):
    """
    Given a mask file (filename or numpy array), process it and return:
      - maskconts: the contours found,
      - maskcoords: a list of (x, y) coordinates from the contours.
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
        maskcoords = [tuple(pt[0]) for cnt in maskconts for pt in cnt]
    except Exception as e:
        print("Mask error:", e)
        maskconts = None
        maskcoords = None
    return maskconts, maskcoords



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
    return (np.nan,np.nan) if val=="" else literal_eval(val)

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


def check_treshtypes(tresh_types):
    if isinstance(tresh_types, (np.floating, float)):
        tresh_types = ["bw"]
    if tresh_types[0] == "(":
        tresh_types = tresh_types[1:len(tresh_types)-1]
    if "," in tresh_types:
        tresh_types = tresh_types.split(',')
    if not isinstance(tresh_types, list):
        tresh_types = [tresh_types]
    return tresh_types


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


def convert(x, y, conv, height, flip=True, roi=None):
    x = x if roi is None else x-roi[0][0]
    x = round(x*conv,3)
    if flip:
        y = height-y if roi is None else roi[1][1]-y
    else:
        y = y if roi is None else y-roi[0][1]
    y = round(y*conv,3)
    return x,y


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


def calc_borderdist(pt, roi):
    wle = np.abs(pt[0])
    wri = np.abs((roi[1][0]-roi[0][0]) - pt[0])
    wto = np.abs(pt[1])
    wbo = np.abs((roi[1][1]-roi[0][1]) - pt[1])
    return min([wle, wri, wto, wbo])

import numpy as np
import pandas as pd

def calc_borderdistvec(df, roi):
    """
    Calculate border distances for all points in a DataFrame
    """
    # Extract cx and cy as numpy arrays
    cx = df['cx'].to_numpy()
    cy = df['cy'].to_numpy()

    # Calculate distances to each border
    wle = np.abs(cx)
    wri = np.abs((roi[1][0] - roi[0][0]) - cx)
    wto = np.abs(cy)
    wbo = np.abs((roi[1][1] - roi[0][1]) - cy)

    # Find the minimum distance for each point
    borderdist = np.minimum.reduce([wle, wri, wto, wbo])

    # Return boolean array where True means the distance is below the threshold
    return borderdist

def differentiate(val, period=1):
    val2 = val.shift(periods=period)
    newval = val-val2
    return newval


def calcudiff(x, y, period=1, angle=False):
    xdiff = differentiate(x, period)
    ydiff = differentiate(y, period)
    if angle:
        head = np.arctan2((xdiff),(ydiff)) * 180 / np.pi
        return head
    else:
        displ = np.sqrt(xdiff**2 + ydiff**2)
        return displ


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
                  extdists, convexity, prev_angle, avg_vel, vel_tresh, prev_coord,
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
    # the two distances is large enough. A treshold of 10% seems good.

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
                vel_ok = avg_vel >= vel_tresh and vel >= vel_tresh
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
    # the speed treshold (convexity is irrelevant here)
    # >>> contour angle can be flipped if needed based on movement angle
    elif not distratio_ok and vel_ok:
        move_angle = points_to_angle(prev_coord, centre_coord, flip=True)
        new_angle = compare_angle(con_angle, move_angle)
        # Be careful for sudden flips in angle
        if minangle(prev_angle, new_angle) > max_anglechange:
            new_angle = hvflipangle(new_angle)

    # Scenario 3: the shape is convex, but there is no distinct front-back ratio,
    # the object did not pass the speed treshold, and we can use the previous angle
    # >>> Now we can use the previous stored angle if not too far back in time
    # and not nan to adjust the contour angle
    elif convex_ok and not distratio_ok and not vel_ok and delay_ok and not np.isnan(prev_angle):
        new_angle = compare_angle(con_angle, prev_angle)
        # Be careful for sudden flips in angle
        if minangle(prev_angle, new_angle) > max_anglechange:
            new_angle = hvflipangle(new_angle)

    # Scenario 4: the shape is convex, but there is no distinct front-back ratio,
    # and the object did not pass the speed treshold, and we cannot use the previous angle
    # >>> We cannot be certain enough of the angle, so it will be stored as nan
    # elif convex_ok and not distratio_ok and not vel_ok and delay_ok and np.isnan(prev_angle):
    else:
        new_angle = np.nan

    #print("convexity",convexity,convex_ok)
    #print("distratio",extdist_ratio,distratio_ok)
    #print("velocity",avg_vel,vel,vel_ok)
    return new_angle


def fixheadtail(series, areamultiplier = 1.75, distmultiplier = 10,
                seqlentreshold = 40, tresharea=None):

    data = series.copy()
    swappedinds = []
    if tresharea is None:
        tresharea = np.nanmedian(data.area)

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
    # head indices where time step is more than treshold
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
                elif (max(data.index) - ind) < seqlentreshold:
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
            if(seqlen) > seqlentreshold:
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
    #     print("area1",len(data.loc[data.area<(tresharea/areamultiplier),["head","fx","fy","tail","tx","ty"]]))
    #     print("area2",len(data.loc[data.area>(tresharea*areamultiplier),["head","fx","fy","tail","tx","ty"]]))
    #     data.loc[data.area<(tresharea/areamultiplier),["head","fx","fy","tail","tx","ty"]] = np.nan
    #     data.loc[data.area>(tresharea*areamultiplier),["head","fx","fy","tail","tx","ty"]] = np.nan
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


def checknearmask(series, coordcols, mask, indsecs, masksecs, win = 5, nearmaskdis = 50):

    nearlist = []
    awaylist = []
    for i,indsec in enumerate(indsecs):
        if (min(series.index.values)+1) >= indsec[0]:
            beflen = win
        else:
            subset = series.loc[masksecs[i][0]:indsecs[i][0]-1,coordcols]
            valid_subset = subset.dropna()
            valid_subset = valid_subset[~np.isinf(valid_subset.to_numpy()).any(axis=1)]
            befdis,_ = KDTree(mask).query(valid_subset)
            beflen = len([n for n in befdis if n<nearmaskdis or n == np.Inf])
        if max(series.index.values) == indsec[1]:
            aftlen = win
        else:
            subset = series.loc[indsecs[i][1]+1:masksecs[i][1], coordcols]
            valid_subset = subset.dropna()
            valid_subset = valid_subset[~np.isinf(valid_subset.to_numpy()).any(axis=1)]
            aftdis,_ = KDTree(mask).query(valid_subset)
            aftlen = len([n for n in aftdis if n<nearmaskdis or n == np.Inf])
        nearlist = nearlist+[i] if (beflen>0 and aftlen>0) else nearlist
        awaylist = awaylist if (beflen>0 and aftlen>0) else awaylist+[i]
        #nearlist = nearlist+[i] if (beflen==win and aftlen==win) else nearlist
        #awaylist = awaylist if (beflen==win and aftlen==win) else awaylist+[i]

    return nearlist, awaylist


def fillmissing(series, colpair, mask, roi, win = 5, nearmaskdis = 25, edgedis = 10,
                lentresh = 100):

    """
    Fill-in missing data for specific numeric columns in a dataset

    First, missing data is ignored when at the start or end of the data series
    as cannot be filled. Second, sections with missing data near a mask are
    ignored as presumed object is missing because it is (partly) behind the
    mask. Third, missing data is ignored when caused by objects dissappearing
    from view. Finally, missing values are calculated by differentiating.
    """

    # Fix new roi
    if roi is not None: 
        roi = ((1,1),(roi[1][0]-roi[0][0],roi[1][1]-roi[0][1]))

    # Get missing indices
    missinds = findmissinginds(series, colpair[0], checkzero = False)
    missing = len(missinds)
    if missing > 0:
        indsecs = getindsections(missinds)

        # Remove starts and stops (with fix if first frame has data but beyond it doesn't)
        indsecs = [ind for ind in indsecs if ind[0] >= (min(series.index.values)+1) and ind[1] != max(series.index.values)]
        missinds = [item for i in indsecs for item in list(range(i[0],i[0]+i[2]))]
        missing = len(missinds)

        # Create diffsecs
        if missing>0:
            diffsecs = getindsections(missinds, win = 1)

            # Subset ind sections to only those not caused by blobs near a mask
            if missing > 0 and mask not in (None,[]):
                masksecs = getindsections(missinds, win = win)
                nearlist, awaylist = checknearmask(series, colpair, mask, indsecs, masksecs, nearmaskdis = nearmaskdis)
                indsecs = [indsec for i,indsec in enumerate(indsecs) if (indsec[2]<5 and i in nearlist) or i in awaylist]
                diffsecs = [diffsec for i,diffsec in enumerate(diffsecs) if (diffsec[2]<(5+2) and i in nearlist) or i in awaylist]
                missing = sum([item[2] for i,item in enumerate(indsecs)])

        # Check if nans are not caused by fish moving out of view and if so label accordingly
        if missing > 0 and "cx" in colpair and roi is not None:
            dellist = []
            for i,indsec in enumerate(indsecs):
                befout = calc_borderdist(series.icom[indsec[0]-1], roi)
                aftout = calc_borderdist(series.icom[indsec[1]+1], roi)
                if befout < edgedis and aftout < edgedis:
                    series.loc[indsec[0]:indsec[1],"inroi"] = 0
                    dellist += [i]
            indsecs = [ind for i,ind in enumerate(indsecs) if i not in dellist]
            diffsecs = [ind for i,ind in enumerate(diffsecs) if i not in dellist]
            missing = sum([item[2] for i,item in enumerate(indsecs)])

    # Now calculate missing values by differentiating for each column
    if missing > 0:
        for i,indsec in enumerate(indsecs):
            if indsec[2]>lentresh:
                missing = missing-indsec[2]
            else:
                for col in colpair:
                    diffsec = diffsecs[i]
                    newvals = np.around(np.linspace(series[col][diffsec[0]], series[col][diffsec[1]], diffsec[2]),3)[1:diffsec[2]-1]
                    series.loc[indsec[0]:indsec[1],col] = newvals

    # Reorder and reindex
    #series = series.sort_values(["frame"])
    #series.index = list(range(0,len(series)))

    return series, missing


def addtrajsnmaskstate(series, mask, cover, win = 5, nearmaskdis = 20, trajgap=50, framebased = False):

    """
    Add trajectories and mask state

    mask : tuple of coordinates
    win : int, default = 5
        The number of frames that should be considered as a buffer on each side
        of key frame indices.
    nearmaskdis : int, default = 50
        The distance from cover in pixels that should be considered to count as
        being near cover. For example, a distance of 50 would mean that any
        coordinate data after a sequence of missing coordinate data that is less
        than 50 pixels away from the nearest mask coordinate would then indicate
        the previous data sequence is in cover.
    """

    if cover:
        series["inmask"] = 0
    nonmissinds = series.index.values

    if framebased:
        frdiff = series.frame - series.frame.shift(periods=1)
        fullinds = [0]+list(frdiff.index[(frdiff>1)]) + [frdiff.index[-1]+1]
        series["traj"] = sum([[trajnr]*(fullinds[trajnr]-fullinds[trajnr-1]) for trajnr in list(range(1,len(fullinds)))],[])

    else:
        missinds = findmissinginds(series, "cx", checkzero = True)
        if len(missinds)>0:
            indsecs = getindsections(missinds)
            if cover and len(mask)>0:
                fullinmaskinds = []
                masksecs = getindsections(missinds, win = win)
                nearlist,awaylist = checknearmask(series, ["cx","cy"], mask, indsecs, masksecs, nearmaskdis = nearmaskdis)
                nearsecs = [indsec for i,indsec in enumerate(indsecs) if i in nearlist]
                for i,nearsec in enumerate(nearsecs):
                    fullinmaskinds += list(range(nearsec[0],nearsec[1]+1))
                series.loc[fullinmaskinds,"inmask"] = 1
                nonmissinds = [i for i in series.index.values if i not in fullinmaskinds]

        outdat = series.index.values[series.inroi==0]
        nonmissinds = [i for i in nonmissinds if i not in outdat]
        if len(nonmissinds)>0:
            trajinds = getindsections(nonmissinds, gap=trajgap)
            for i,trajind in enumerate(trajinds):
                series.loc[trajind[0]:trajind[1],"traj"] = int(i+1)
                missinds = findmissinginds(series.loc[trajind[0]:trajind[1]],"cx")
                if len(missinds)>0:
                    diffsecs = getindsections(missinds, win = 1)
                    for diffsec in diffsecs:
                        if diffsec[1]!= max(series.index.values)+1:
                            for col in ["cx","cy"]:
                                newvals = np.around(np.linspace(series[col][diffsec[0]], series[col][diffsec[1]], diffsec[2]),3)[1:diffsec[2]-1]
                                series.loc[diffsec[0]+1:diffsec[1]-1,col] = newvals

        trajs = np.unique(series.traj[~np.isnan(series.traj)])
        ntraj = max(trajs) if len(trajs)>0 else 0

    return ntraj


def removenearmask(series, mask, inmaskdis = 10, mintrajlength = 10):

    """Removes data where object is within certain distance of the mask and handles short trajectories."""

    # Step 1: Identify and remove rows where the object is within a certain distance of the mask
    locs = [(np.nan, np.nan) if np.isnan(a) else (int(a), int(b)) for a, b in zip(series.cx, series.cy)]
    distances, maskids = KDTree(mask).query(locs)

    # Identify columns to set to NaN when near the mask
    cols = [i for i in list(series.loc[:, "area":].columns.values) if i not in ["frame", "time", "inroi", "ID"]]
    
    # Remove data where object is near the mask
    series.loc[distances < inmaskdis, cols] = np.nan
    missing = len(distances[distances < inmaskdis])

    # Step 2: Fill missing trajectory IDs between the first and last frame of each trajectory
    trajids = [i for i in series.traj.unique() if not np.isnan(i)]
    for trajid in trajids:
        traj_inds = series.loc[series.traj == trajid].index
        if len(traj_inds) > 0:
            # Fill missing trajectory IDs between the first and last index
            series.loc[traj_inds[0]:traj_inds[-1], "traj"] = trajid
   
    # Step 3: Remove short trajectories
    for trajid in trajids:
        traj_length = len(series.loc[series.traj == trajid])
        if traj_length < mintrajlength:
            # Remove all coordinates and relevant data for this short trajectory
            series.loc[series.traj == trajid, cols] = np.nan
    
    # Step 4: Re-index trajectories to ensure continuous numbering
    valid_trajids = [i for i in series.traj.unique() if not np.isnan(i)]
    for new_traj_id, trajid in enumerate(valid_trajids, start=1):
        series.loc[series.traj == trajid, "traj"] = new_traj_id

    # Step 5: Add missing mask status for `inmask`
    series.loc[series.inmask.isnull(), "inmask"] = 1

    return distances, maskids, missing


def smooth(series, trajs = None, columns = ["cx","cy"], smoothwin = 5):
    
    if trajs is None: 
        if len(series)>6:
            for column in columns:
                nonanlen = series.loc[:,column].dropna().shape[0]
                win = smoothwin if nonanlen>smoothwin else nonanlen
                if win>6:
                    win = win-1 if win % 2 == 0 else win # make odd
                    startind = next(i for i,j in enumerate(series.loc[:,column]) if j==j)
                    inds = list(series.loc[:][startind:].index)
                    series.loc[inds,column] = savgol_filter(x = series.loc[inds,column], polyorder = 3, window_length = win)
    
    else:
        for t in trajs:
            if len(series[series.traj==t])>6:
                for column in columns:
                    nonanlen = series.loc[series.traj==t,column].dropna().shape[0]
                    win = smoothwin if nonanlen>smoothwin else nonanlen
                    if win>6:
                        win = win-1 if win % 2 == 0 else win # make odd
                        startind = next(i for i,j in enumerate(series.loc[series.traj==t,column]) if j==j)
                        inds = list(series.loc[series.traj==t][startind:].index)
                        series.loc[inds,column] = savgol_filter(x = series.loc[inds,column], polyorder = 3, window_length = win)
    
    return series

