import numpy as np
import cv2 as cv
import cv2
import glob
images = [cv2.imread(file) for file in glob.glob("/home/mac/Desktop/clahe/*.jpg")]
for i in range(0,15):
		img = images[i]
		Z = img.reshape((-1,3))
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
		cv.imshow('res2',res2)
		cv.imwrite('pol1.jpg',res2)


		dim=(250,250)
		source = cv2.imread("pol1.jpg",0)
		source=cv2.resize(source, dim, interpolation = cv2.INTER_AREA)
		img64_float = source.astype(np.float64)
		 
		Mvalue = np.sqrt(((img64_float.shape[0]/2)**2.0)+((img64_float.shape[1]/2)**2.0))
		 
		 
		ploar_image = cv2.linearPolar(img64_float,(img64_float.shape[0]/2, img64_float.shape[1]/2),Mvalue,cv2.WARP_FILL_OUTLIERS)
		#p2=cv2.LogPolar(img64_float, (img64_float.shape[0]/2, img64_float.shape[1]/2),(40,50) , Mvalue,cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS) 
		cartisian_image = cv2.linearPolar(ploar_image, (img64_float.shape[0]/2, img64_float.shape[1]/2),Mvalue, cv2.WARP_INVERSE_MAP)

		cartisian_image = cartisian_image/200
		ploar_image = ploar_image/255

		ploar_image=cv2.resize(ploar_image, dim, interpolation = cv2.INTER_AREA)
		cv2.imwrite("ploarimg.jpg", ploar_image)
		pol=cv2.imread("ploarimg.jpg")
		blur = cv2.GaussianBlur(pol,(51,51),0)
		#cv2.imshow('blur',blur)
		edges = cv2.Canny(blur,3,0)

		'''
		lines = cv2.HoughLines(edges,1,np.pi/90,1)
		for rho,theta in lines[0]:
		    a = np.cos(theta)
		    b = np.sin(theta)
		    x0 = a*rho
		    y0 = b*rho
		    x1 = int(x0 + 1000*(-b))
		    y1 = int(y0 + 1000*(a))
		    x2 = int(x0 - 1000*(-b))
		    y2 = int(y0 - 1000*(a))

		    cv2.line(source,(x1,y1),(x2,y2),(255,255,255),2)
		'''
		cv2.imwrite('houghlines3.jpg',edges)
		cv2.imshow("polar transform of macular hole", ploar_image)
		cv2.imshow("edge", edges)
		imag=cv2.imread("217.jpg")
		cv2.imshow("orignal image", imag)
		cv2.imshow("macular hole", cartisian_image)
		#cv2.imshow("log-P",p2) 
		cv2.waitKey(0)
		cv2.destroyAllWindows()


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
			#elif cv2.contourArea(c)>2000:
			    #continue
			cv2.drawContours(img, [c], -1, (0,255,0), 3)
			print("hi")
			M = cv2.moments(c)
			cX = int(M['m10'] /M['m00'])
			cY = int(M['m01'] /M['m00'])
			centers.append([cX,cY])
			print cX,cY
		if len(centers) >=0:
				dx= centers[0][0] - centers[1][0]
				dy = centers[0][1] - centers[1][1]
				D = np.sqrt(dx*dx+dy*dy)
		       		print("dx:") 
				print(dx)
				print("dy:") 
				print(dy)
		if (dx >  dy) :
			print("macular hole")
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(images[i], 'macular hole', (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
		else :
			print("macula")
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(images[i], 'macula', (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)

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
		cv2.imshow('img',images[i])
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
		
