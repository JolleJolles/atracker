#! /usr/bin/env python

import cv2
import os
import pandas as pd

from pythutils.fileutils import listfiles
from pythutils.drawutils import namedcols, draw_crosshair
from pythutils.mathutils import ptsToDist

class mouse_events:
    def __init__(self):
        self.pts = []
        self.pointer = ()
    def draw(self,event,x,y,flags,param):
        self.pointer = (x,y)
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.pts) < 2:
                self.pts.append((x,y))
            else:
                self.pts[1] = (x,y)


def getsize(imgdir = ".", startimg = 1, filetype = ".jpg", datfilename = "",
             default = None, realsize = 50, linewidth = 3):

    assert os.path.isdir(imgdir), "Image folder does not exist.."

    imgfiles = listfiles(imgdir, filetype, keepdir=True)
    if default is not None:
        assert os.path.isfile(deffile), "Default image ("+default+") could not be loaded.."
        imgfiles = [default]+imgfiles
    else:
        print("Taking 1st image to set conversion ratio..")
    imgfiles = [imgfiles[0]]+imgfiles[startimg:]

    data = pd.DataFrame(columns = ["ID", "realdist"])
    conv = None
    exit = "off"

    for i, imgfile in enumerate(imgfiles):

        if i == 0:
            print("Draw distance of "+str(realsize)+"mm..", end=" ")
        else:
            print("File",i,"of",len(imgfiles)-1,os.path.basename(imgfile))

        mouse = mouse_events()
        cv2.namedWindow("Image", cv2.WND_PROP_FULLSCREEN)
        cv2.setMouseCallback('Image', mouse.draw)
        img = cv2.imread(imgfile)

        while True:
            drawimg = img.copy()
            if len(mouse.pointer) > 0:
                draw_crosshair(drawimg, mouse.pointer, col = "black")
            if len(mouse.pts) == 2:
                cv2.line(drawimg, mouse.pts[0], mouse.pts[1], namedcols("red"), linewidth)
            for pt in mouse.pts:
                cv2.circle(drawimg, pt, 5, namedcols("red"), -1)

            cv2.imshow("Image", drawimg)
            k = cv2.waitKey(1) & 0xFF

            # Store points
            if k == ord("s"):
                if len(mouse.pts) == 2:
                    dist = ptsToDist(mouse.pts[0], mouse.pts[1])
                    if conv is None:
                        conv = realsize/dist
                        print("conversion ratio stored..")
                    else:
                        realdist = round(dist*conv,2)
                        ID = input("Distance = "+str(realdist)+".. ID:")
                        data.loc[len(data)] = [ID,realdist]
                    break
                else:
                    print("not enough points..")

            # Erase points
            if k == ord("e"):
                mouse.pts = []

            # Exit and go to next fish
            if k == "]":
                print("skip to next photo..")
                break

            # Exit completely without saving
            if k == 27:
                exit = "exit"
                print("user exited.. data not stored..")
                break

            # Exit completely with saving
            if k == "w":
                exit = "exitandwrite"
                print("user exited..")
                break

        if exit == "exit":
            break

    # Close windows
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    for i in range(100):
        cv2.waitKey(1)

    # Store all data
    if exit != "exit":
        data.to_csv("lengthdata_"+datfilename+".csv", index=False)
        print("Stored all data..")
