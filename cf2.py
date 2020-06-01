import cv2
import numpy as np
from matplotlib import pyplot as plt
#from numba import jit
count = 0
centers=[]
img = cv2.imread('houghlines3.jpg',0)
img = np.array(img)
#print(img.ndim)
edges = img
#plt.plot(1),plt.imshow(img,cmap = 'gray')
#plt.show()
edges = cv2.rotate(edges, cv2.ROTATE_90_CLOCKWISE)
edges = cv2.resize(edges, (500,500))
#edges = edges[320:500, 0:500]
#edges = cv2.resize(edges, (500,500))
cv2.imwrite("star.jpg",edges)
edges.astype(int)
x=np.empty(edges.shape[0])
y=np.empty(edges.shape[1])
x.astype(int)
y.astype(int)

img = cv2.imread('star.jpg')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(img_gray,127,255,0)
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
        #if cv2.contourArea(c)<100:
            #continue
       # elif cv2.contourArea(c)>2000:
            #continue
        cv2.drawContours(img, [c], -1, (0,255,0), 3)
	print("hi")
        M = cv2.moments(c)
        cX = int(M['m10'] /M['m00'])
        cY = int(M['m01'] /M['m00'])
        centers.append([cX,cY])

	if len(centers) >=2:
        	dx= centers[0][0] - centers[1][0]
        	dy = centers[0][1] - centers[1][1]
        	D = np.sqrt(dx*dx+dy*dy)
       		print("dx:") 
		print(dx)
		print("dy:") 
		print(dy)
if (abs(dx) >  abs(dy)) :
	print("macular hole")
else :
	print("macula")

print(len(contours))
#cnt = contours[0]
'''
hull = cv2.convexHull(cnt,returnPoints = False)
defects = cv2.convexityDefects(cnt,hull)

for i in range(defects.shape[0]):
    s,e,f,d = defects[i,0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv2.line(img,start,end,[0,255,0],2)
    cv2.circle(img,far,5,[0,0,255],-1)
'''
img = cv2.drawContours(img, contours, -1, (0,255,0), 3)
cv2.imshow('img',img)
cv2.imwrite("contour.jpg",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
for i in range(edges.shape[0]):
    for j in range(edges.shape[1]):
        if edges[i, j] > 254:
		x=np.append(x,i)
		y=np.append(y,j)             

maxElementx = np.amax(x)
maxElementy = np.amax(y)
for i in range(edges.shape[0]):
    for j in range(edges.shape[1]):
        if y[j] == maxElementy:
		count=count+1
		
print(count)
plt.plot(1),plt.imshow(edges,cmap = 'gray')
plt.show()
'''
