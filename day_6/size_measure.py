import cv2
import numpy
import imutils
import argparse
from imutils import contours
import numpy as np
from imutils import perspective
from scipy.spatial import distance

def midpoint(x, y):
    return ((x[0] + y[0])/2, (x[1] + y[1])/2)

    
ap = argparse.ArgumentParser();
ap.add_argument('-i', '--image', required=True, help="Path to image")
ap.add_argument('-w', '--width', required=True, type=float, help="width of leftmost obejct in inches")
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
pixelsPerMetric = None

for c in cnts:
    if cv2.contourArea(c) < 100:
        continue

    original = image.copy()
    box = cv2.minAreaRect(c)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype='uint8')
    perspective.order_points(box)
    cv2.drawContours(original, [box.astype('int')], -1, (0,255,0), 2)

    for(x, y) in box:
        cv2.circle(original, (int(x), int(y)), 5, (0,0,255), -1)

    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)
    cv2.circle(original, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(original, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(original, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(original, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    cv2.line(original, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
    cv2.line(original, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

    dA = distance.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = distance.euclidean((tlblX, tlblY), (trbrX, trbrY))

    if pixelsPerMetric is None:
        pixelsPerMetric = dB / args["width"]
    
    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric

    cv2.putText(original, "{:.1f}in".format(dimA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(original, "{:.1f}in".format(dimB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.imshow("Image", original)
    cv2.waitKey(0)

cv2.destroyAllWindows()
