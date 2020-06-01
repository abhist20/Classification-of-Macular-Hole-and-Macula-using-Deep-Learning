import cv2
import numpy as np
image = cv2.imread("out.jpg")
gray = cv2.GaussianBlur(image, (41, 41) , 0)
gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
ioi=int(maxLoc[1])
ioi1=int(maxLoc[0])
cv2.circle(image, minLoc, 0, (0, 0, 255), 3)
print minLoc
cv2.imshow('lol',image)
cv2.waitKey(0)

