from four_point_transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2 as cv
import imutils

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="path to image")
args = vars(ap.parse_args())

image = cv.imread(args["image"], cv.IMREAD_UNCHANGED)
ratio = image.shape[0] / 500.0
original = image.copy()
image = imutils.resize(image, height=500)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(image, (5,5), 0)
edged = cv.Canny(image, 75, 200)

cnts = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:5]
for c in cnts:
	# approximate the contour
	peri = cv.arcLength(c, True)
	approx = cv.approxPolyDP(c, 0.02 * peri, True)
	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break
# show the contour (outline) of the piece of paper
print("STEP 2: Find contours of paper")
cv.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

warped = four_point_transform(original, screenCnt.reshape(4, 2) * ratio)
# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255
# show the original and scanned images
print("STEP 3: Apply perspective transform")
cv.imshow("Original", imutils.resize(original, height = 650))
cv.imshow("Scanned", imutils.resize(warped, height = 650))
cv.waitKey(0)