import numpy as np
import os
import cv2
image=cv2.imread("clahe/image13.jpg")
row,col,ch= image.shape
mean = 10
var = 10
sigma = var**10
gauss = np.random.normal(mean,sigma,(row,col,ch))
gauss = gauss.reshape(row,col,ch)
noisy = image + gauss
cv2.imwrite('lol.jpg',noisy)
cv2.waitKey(0)

