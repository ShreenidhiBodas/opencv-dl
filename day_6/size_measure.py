import cv2
import numpy
import imutils
import argparse
from imutils import contours


ap = argparse.ArgumentParser();
ap.add_argument('-i', ='--image', help="Path to image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"], cv2.IMREAD_UNCHANGED)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(blurred, 50, 100)
edged = cv2.erode(edged, None, iterations=1)
edged = cv2.dilate(edged, None, iterations=1)

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
(cnts, _) = contours.sort_contours(cnts)
