import imutils
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('E:/Wallpapers/721152.jpg', cv2.IMREAD_UNCHANGED)
(h, w, d) = image.shape
print("width = {}, height = {}, dimensions = {}".format(h, w, d))

downscale_percent = 15
width = int(image.shape[1]*downscale_percent/100)
height = int(image.shape[0]*downscale_percent/100)

new_dim = (width, height)
image = cv2.resize(image, new_dim)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
