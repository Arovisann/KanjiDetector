import cv2
from matplotlib import image
import numpy as np
import os

image = os.path.dirname("C:\\Users\\alexa\\OneDrive\\Bureau\\opencv\\image.png")
img = cv2.imread("C:\\Users\\alexa\\OneDrive\\Bureau\\opencv\\image.png")

print(img.shape)



cv2.imshow("image", img)
cv2.waitKey(0)

#ROI (region of interest)
cv2.rectangle(img, (75,75), (150,150), (0, 255, 0), 3)

crop_image = img[0:757, 0:756]
cv2.imshow("cropped", crop_image)

cv2.waitKey(0)