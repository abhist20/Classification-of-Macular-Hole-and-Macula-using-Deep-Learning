import cv2
import numpy as np
from matplotlib import pyplot as plt

def find_first(item, vec):
    """return the index of the first occurence of item in vec"""
    for i in range(len(vec)):
        if item == vec[i]:
            return i
    return -1

img = cv2.imread('houghlines3.jpg',0)
edges = img #cv2.Canny(img,100,200)
edges = cv2.rotate(edges, cv2.ROTATE_90_CLOCKWISE)
edges=cv2.resize(edges, (500,500))
edges = edges[310:500, 0:500]
bounds = [750, 1500]
# Now the points we want are the lowest-index 255 in each row
window = edges[bounds[1]:bounds[0]:-1].transpose()

xy = []
for i in range(len(window)):
    col = window[i]
    j = find_first(255, col)
    if j != -1:
        xy.extend((i, j))
# Reshape into [[x1, y1],...]
data = np.array(xy).reshape((-1, 2))
# Translate points back to original positions.
data[:, 1] = bounds[1] - data[:, 1]
plt.figure(1, figsize=(8, 16))
ax1 = plt.subplot(211)
ax1.imshow(edges,cmap = 'gray')
ax2 = plt.subplot(212)
ax2.axis([0, edges.shape[1], edges.shape[0], 0])
ax2.plot(data[:,1])
plt.show()

#plt.plot(1),plt.imshow(edges,cmap = 'gray')
#plt.title('WAVE')
#plt.show()
