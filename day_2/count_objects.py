import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt

image = cv2.imread('tetris_blocks.png', cv2.IMREAD_UNCHANGED)
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray_img, 30, 150, L2gradient=False)
_, thresh = cv2.threshold(gray_img, 225, 255, cv2.THRESH_BINARY_INV)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
output = image.copy()

for c in cnts:
    cv2.drawContours(output, [c], -1, (240, 0, 159), 3)
	
text = "I found {} objects!".format(len(cnts))
cv2.putText(output, text, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 0, 159), 2)
cv2.imshow("Contours", output)
cv2.waitKey(0)
# cv2.imshow('gray', gray_img)
# cv2.imshow('threshold', thresh)
# cv2.imshow('edged', edged)
# cv2.waitKey(0)
cv2.destroyAllWindows()