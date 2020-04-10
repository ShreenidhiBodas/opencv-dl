import imutils
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('E:/Wallpapers/721152.jpg', cv2.IMREAD_UNCHANGED)
(h, w, d) = image.shape
print("width = {}, height = {}, dimensions = {}".format(h, w, d))

# downscale_percent = 15
# width = int(image.shape[1]*downscale_percent/100)
# height = int(image.shape[0]*downscale_percent/100)

# new_dim = (width, height)
image = imutils.resize(image, width=1000)

# center = (w // 2, h // 2)
# M = cv2.getRotationMatrix2D(center, -45, 1.0)
# rotated = cv2.warpAffine(image, M, (w, h))

rotated = imutils.rotate(image, -45)
cv2.imshow("imutils Rotation", rotated)
cv2.imshow('image', image)
rotate_bound = imutils.rotate_bound(image, -45)
cv2.imshow("OpenCV Rotation", rotate_bound)
cv2.waitKey(0)
cv2.destroyAllWindows()
