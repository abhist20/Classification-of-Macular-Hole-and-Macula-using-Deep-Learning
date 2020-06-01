import cv2
import numpy as np
 
source = cv2.imread("pol.jpg",0)
img64_float = source.astype(np.float64)
 
Mvalue = np.sqrt(((img64_float.shape[0]/2)**2.0)+((img64_float.shape[1]/2)**2.0))

blank=np.zeros((img64_float.shape[0], img64_float.shape[1]))
blank1=(img64_float.shape[0], img64_float.shape[1]) 
#ploar_image = cv2.linearPolar(img64_float,blank,Mvalue,cv2.WARP_FILL_OUTLIERS)
p2=cv2.LogPolar(img64_float, blank, 500,cv2.WARP_FILL_OUTLIERS) 
cartisian_image = cv2.linearPolar(ploar_image,blank,Mvalue, cv2.WARP_INVERSE_MAP)
 
cartisian_image = cartisian_image/200
#ploar_image = ploar_image/255
p2=p2/255
p2=cv2.resize(p2, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("log-P",p2)
#cv2.imshow("log-polar1", ploar_image)
cv2.imshow("log-polar2", cartisian_image)
 
cv2.waitKey(0)
cv2.destroyAllWindows()
