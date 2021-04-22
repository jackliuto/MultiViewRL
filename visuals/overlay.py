# import the necessary packages
from __future__ import print_function
import numpy as np
import cv2
# load the image
img_bgr = cv2.imread("orge.jpg")
ovr = cv2.imread("ove.jpg")

img = img_bgr[...,::-1].copy()

scale_percent = 100 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
  
#Resize image
resized = cv2.resize(ovr, dim, interpolation = cv2.INTER_AREA)

#Enhance regions where the model focuses on
print(resized[0][0].shape)
for i in range(len(resized)):
    for j in range(len(resized[i])):
        for k in range(len(resized[i][j])):
            if resized[i][j][k]>32:
                resized[i][j][k]+=80

#Overlay map on original image
out = cv2.addWeighted(resized, 0.65, img_bgr, 1 - 0.65, 0)

#View image
cv2.imshow("out", out)
