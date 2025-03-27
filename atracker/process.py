#! /usr/bin/env python

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import point, LineString, Polygon

from pythutils.mediautils import crop
from pythutils.mathutils import points_to_angle, ptsToDist

from .utils import *

class ImgProcessor:

    def __init__(self, img, img_bg, img_mask=None, blur=9, erode=1, treshold=50,
    min_area=100, max_area=10000, croppad=20, simple=True):

        self.img = img
        self.img_bg = img_bg
        self.img_mask = img_mask
        self.blur = blur
        self.erode = np.ones((erode, erode), np.uint8)
        self.treshold = treshold
        self.min_area = min_area
        self.max_area = max_area
        self.croppad = croppad
        self.simple = simple


    def process(self):

        """
        Processes an image and returns tresholded image and contourlist

        inertia: The straightness of the contour. The closer inertia is to 0, the
            more the contour is like a straight line, so contours with very high
            inertia are best ignored.
        convexity: The curvature of the contour. The closer convexity is to 0, the
            more it is convexed and thus curved, so contours with very low convexity
            are best ignored.
        """

        # Create the different images
        self.img = cv2.absdiff(self.img, self.img_bg)
        if self.img_mask is not None:
            self.img = cv2.bitwise_and(self.img, self.img, mask = self.img_mask)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        self.img_tresh = cv2.GaussianBlur(self.img, (self.blur, self.blur), 3)
        self.img_tresh = cv2.erode(self.img_tresh, self.erode, 3)
        self.img_tresh = cv2.threshold(self.img_tresh, self.treshold, 255, cv2.THRESH_BINARY)[1]

        # Extract contours
        allcons,_ = cv2.findContours(self.img_tresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # Sort contours
        #self.allcons.sort(key=lambda x:get_contour_precedence(x, self.img.shape[1]))
        self.allcons = sorted(allcons, key=lambda x: cv2.contourArea(x), reverse=True)

        # Create conlist
        varlist = ("id","objbox","area","length","inertia","convexity","curvature","com","icom",
        "head","tail","angle","lathom","fed","skeleton","contour")
        self.conlist = {var:[np.nan]*len(self.allcons) for var in varlist}

        # Go through the contours and extract contour information
        for ind,contour in enumerate(self.allcons):
            self.processcon(ind, contour)

        # Subset to only contours within dimensions
        inds,_ = dic_exclnan(self.conlist,"com")
        for key in self.conlist.keys():
            self.conlist[key] = [self.conlist[key][i] for i in inds]

        return self.img_tresh, self.allcons, self.conlist


    def processcon(self, ind, contour):

        # Get contour area
        self.conlist["contour"][ind] = contour
        area = int(cv2.contourArea(contour))
        self.conlist["area"][ind] = area

        # Continue only with contours of the right size
        if area > self.min_area and area < self.max_area:

            # Get object centroid
            self.conlist["com"][ind] = concom(contour)
            if self.simple:
                self.conlist["icom"][ind] = concom(contour)
            else:

                # Calculate minimum straight bounding rectangle (used for cropping)
                rec = cv2.boundingRect(contour)
                rec_tl = (max(rec[0]-self.croppad,1),max(rec[1]-self.croppad,1))
                rec_br = (min(rec[0]+rec[2]+self.croppad,self.img_tresh.shape[1]),
                          min(rec[1]+rec[3]+self.croppad,self.img_tresh.shape[0]))
                rec_pts = (rec_tl, rec_br)
                self.conlist["objbox"][ind] = rec_pts

                # Calculate minimum rotatated bounding rectangle (used to get box centroid
                # and finding head and tail
                rotrecb = cv2.minAreaRect(contour)
                rotrec = cv2.boxPoints(rotrecb)
                rotrec = np.intp(rotrec)

                # Get object inertia and convexity
                ## The closer inertia is to 0, the more the object is like a line
                self.conlist["inertia"][ind] = round(float(min(rotrecb[1])+0.0001) / max(rotrecb[1]),3)

                ## The closer convexity is to 1 the more the object is convex thus curved
                self.conlist["convexity"][ind] = round(area / float(cv2.contourArea(cv2.convexHull(contour)) + 0.0001),3)

                # Calculate rotrec centroid and corner coordinates
                rotrec_com = concom(rotrec)
                rotrec_pts = [tuple(pt) for pt in rotrec]

                # Create distanced image of just contour
                canvas = crop(self.img_tresh, rec_pts)
                canvas = np.zeros(canvas.shape, np.uint8)
                cv2.drawContours(canvas, [contour-rec_tl], 0, (255,255,255), -1)
                cv2.drawContours(canvas, [contour-rec_tl], 0, (255,255,255), 10)

                # Create distanced image to get furthest distance of contour edge
                imgc_dis = cv2.distanceTransform(canvas, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                imgc_dis = cv2.normalize(imgc_dis, imgc_dis, 0, 1, cv2.NORM_MINMAX)*255
                _,_,_,obj_fed = cv2.minMaxLoc(imgc_dis)
                obj_fed = adjpt(obj_fed, rec_tl)
                self.conlist["fed"][ind] = obj_fed

                # Get skeleton (for getting head and tail)
                imgc_skel = cv2.ximgproc.thinning(canvas)

                # Calculate length of object
                self.conlist["length"][ind] = len(imgc_skel[imgc_skel==255])

                # Get head and tail
                ## Get coordinates of skeleton line
                skelpts = np.argwhere(imgc_skel==255)
                skelpts = [adjpt((pt[1],pt[0]),rec_tl) for pt in skelpts]

                ## Get two extreme skeleton points
                distmat = cdist(skelpts, skelpts)
                ext_inds = np.unravel_index(distmat.argmax(), distmat.shape)
                ext_pts = [skelpts[ext_inds[0]], skelpts[ext_inds[1]]]

                ## Head is point closest to widest part of object
                if ptsToDist(obj_fed,ext_pts[0]) > ptsToDist(obj_fed,ext_pts[1]):
                    ext_pts.reverse()
                head, tail = ext_pts
                self.conlist["tail"][ind] = tail

                # Get ordered skeleton points
                sortedinds = [i[0] for i in cdist(skelpts, [ext_pts[1]])]
                skelpts = [skelpts[i] for i in np.argsort(sortedinds)]
                self.conlist["skeleton"][ind] = skelpts

                # Get point on skeleton line closest to box midpoint
                icom = skelpts[cdist(skelpts, [(rotrec_com)]).argmin()]
                self.conlist["icom"][ind] = icom

                # Get def con_lathom(skeleton,contour):
                self.conlist["lathom"][ind] = con_lathom(skelpts, contour)

                # Get angle of head
                angle = int(points_to_angle(self.conlist["icom"][ind], head, flip=True))
                self.conlist["angle"][ind] = angle

                # Improve position of head to be on contour
                headout = get_coord(head[0], head[1], angle, 20, astuple=True)
                p = Polygon([list(i[0]) for i in contour])
                l = LineString([[headout[0],headout[1]],[head[0],head[1]]])
                newhead  = geom_tocoord(p.buffer(0).intersection(l))
                if newhead != newhead:
                    contuple = contour_to_tuple(contour)
                    _, pixid = KDTree(contuple).query(headout)
                    newhead = contuple[pixid]
                self.conlist["head"][ind] = newhead

                # Calculate curvature
                headtotail = ptsToDist(self.conlist["head"][ind], self.conlist["tail"][ind])
                canvas = np.zeros(canvas.shape, np.uint8)
                head,tail,icom = [adjpt(pt,rec_tl,add=False) for pt in [head,tail,icom]]
                cv2.drawContours(canvas, np.array([[head,tail]]), 0, 255, -1)
                headtotail = len(np.argwhere(canvas==255))
                cv2.drawContours(canvas, np.array([[head,tail,icom]]), 0, 255, -1)
                headcomtotail = len(np.argwhere(canvas==255))
                self.conlist["curvature"][ind] = round(headtotail/headcomtotail,3)
