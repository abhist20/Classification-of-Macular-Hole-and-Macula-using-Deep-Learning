import numpy as np
import cv2 as cv
import cv2
img = cv.imread('macular hole/image41.jpg')
img1 = cv.imread('macular hole/image41.jpg',0)
dim = (250,250)
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
Z = img.reshape((-1,3))
kernel = np.ones((13,13),np.uint8)
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
cv2.imshow('kmeans',res2)
cv2.waitKey(0)
cv2.imwrite("pol2.jpg",res2)
dilation = cv.morphologyEx(res2,cv.MORPH_OPEN,kernel)
dilation1 = cv.dilate(res2,kernel,iterations = 2)
dilation2 = cv.erode(res2,kernel,iterations = 1)
gradient = cv.morphologyEx(res2, cv.MORPH_GRADIENT, kernel)
cv.imshow('open',dilation)
cv.imshow('dialate',dilation1)
cv.imshow('erode',dilation2)
cv.imshow('gradient',gradient)
cv2.waitKey(0)
cv.imwrite('lo1.jpg',gradient)

res2= cv2.imread('lo1.jpg',0)
blur = cv2.GaussianBlur(res2,(11,11),0)
cv2.imshow('blur',blur)
edges = cv2.Canny(blur,1,0)
cimg = img
cv.imshow('edge',edges)
cv2.waitKey(0)
circles = cv2.HoughCircles(edges,cv.HOUGH_GRADIENT,1,20000,
                            param1=50,param2=29,minRadius=40,maxRadius=108)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(255,255,255),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(255,255,255),3)

cv2.imshow('detected circles',cimg)

cv.waitKey(0)
cv.destroyAllWindows()
