from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import sys
import tensorflow as tf
import glob
#global i
sys.path.append("..")
from utils import visualization_utils as vis_util
from utils import label_map_util

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt





## Good quality / Bad quality classifier (INCEPTION V3)



import argparse

import numpy as np
import tensorflow as tf
import cv2
font = cv2.FONT_HERSHEY_SIMPLEX
try:
	d=0	
	images1 = [cv2.imread(file) for file in glob.glob("/home/mac/Desktop/images/*.jpg")]
	for i in range(0,714):
		def load_graph(model_file):
		  graph = tf.Graph()
		  graph_def = tf.GraphDef()

		  with open(model_file, "rb") as f:
		    graph_def.ParseFromString(f.read())
		  with graph.as_default():
		    tf.import_graph_def(graph_def)

		  return graph


		def read_tensor_from_image_file(file_name,
				                input_height=299,
				                input_width=299,
				                input_mean=0,
				                input_std=255):
		  input_name = "file_reader"
		  output_name = "normalized"
		  file_reader = tf.read_file(file_name, input_name)
		  if file_name.endswith(".png"):
		    image_reader = tf.image.decode_png(
			file_reader, channels=3, name="png_reader")
		  elif file_name.endswith(".gif"):
		    image_reader = tf.squeeze(
			tf.image.decode_gif(file_reader, name="gif_reader"))
		  elif file_name.endswith(".bmp"):
		    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
		  else:
		    image_reader = tf.image.decode_jpeg(
			file_reader, channels=3, name="jpeg_reader")
		  float_caster = tf.cast(image_reader, tf.float32)
		  dims_expander = tf.expand_dims(float_caster, 0)
		  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
		  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
		  sess = tf.Session()
		  result = sess.run(normalized)

		  return result


		def load_labels(label_file):
		  label = []
		  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
		  for l in proto_as_ascii_lines:
		    label.append(l.rstrip())
		  return label


		if __name__ == "__main__":
		  cv2.imwrite('/home/mac/Desktop/temp/image.jpg',images1[i])
		  file_name = "/home/mac/Desktop/temp/image.jpg"
		  model_file = \
		    "/home/mac/Desktop/good ,bad images classfier model/output_graph.pb"
		  label_file = "/home/mac/Desktop/good ,bad images classfier model/output_labels.txt"
		  input_height = 299
		  input_width = 299
		  input_mean = 0
		  input_std = 255
		  input_layer = "Placeholder" 
		  output_layer = "final_result" 

		  '''parser = argparse.ArgumentParser()
		  parser.add_argument("--image", help="image to be processed")
		  parser.add_argument("--graph", help="graph/model to be executed")
		  parser.add_argument("--labels", help="name of file containing labels")
		  parser.add_argument("--input_height", type=int, help="input height")
		  parser.add_argument("--input_width", type=int, help="input width")
		  parser.add_argument("--input_mean", type=int, help="input mean")
		  parser.add_argument("--input_std", type=int, help="input std")
		  parser.add_argument("--input_layer", help="name of input layer")
		  parser.add_argument("--output_layer", help="name of output layer")
		  args = parser.parse_args()'''
		 

	 
		  '''if args.graph:
		    model_file = args.graph
		  if args.image:
		    file_name = args.image
		  if args.labels:
		    label_file = args.labels
		  if args.input_height:
		    input_height = args.input_height
		  if args.input_width:
		    input_width = args.input_width
		  if args.input_mean:
		    input_mean = args.input_mean
		  if args.input_std:
		    input_std = args.input_std
		  if args.input_layer:
		    input_layer = args.input_layer
		  if args.output_layer:
		    output_layer = args.output_layer
	`         '''		
		  graph = load_graph(model_file)
		  t = read_tensor_from_image_file(
		      file_name,
		      input_height=input_height,
		      input_width=input_width,
		      input_mean=input_mean,
		      input_std=input_std)

		  input_name = "import/" + input_layer
		  output_name = "import/" + output_layer
		  input_operation = graph.get_operation_by_name(input_name)
		  output_operation = graph.get_operation_by_name(output_name)

		  with tf.Session(graph=graph) as sess:
		    results = sess.run(output_operation.outputs[0], {
			input_operation.outputs[0]: t
		    })
		  results = np.squeeze(results)
		  image = images1[i]
		  top_k = results.argsort()[-5:][::-1]
		  labels = load_labels(label_file)
		  #for i in top_k:
		    #print(labels[i], results[i])
		  if (results[0] > results[1]):
		    print (labels[0],results[0])
		    cv2.putText(image,labels[0],(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
		    dim = (400,400)
		    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
                    cv2.imshow('result',image)
		    #cv2.waitKey(0)
		  else:
		    print(labels[1],results[1])
		    cv2.imwrite('/home/mac/Desktop/good images/image%d.jpg'%d,image)
		    cv2.putText(image,labels[1],(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
		    dim = (400,400)
		    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
		    cv2.imshow('result',image)
	    	    #cv2.waitKey(0)
		  d=d+1	

except Exception:
    pass

'''

## OPTIC DISC DETECTION ROI (RCNN)

try:
	images = [cv2.imread(file) for file in glob.glob("/home/mac/Desktop/good images/*.jpg")]
	for i in range(0,200):
		# Name of the directory containing the object detection module we're using
		MODEL_NAME = 'ODT'
		IMAGE_NAME = 'NO MACULA TESTING/m3.jpg'
		image=images[i]
		dim = (500,500)
		image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)		
		cv2.imwrite("/home/mac/Desktop/temp/grab.jpg",image)
		imga=cv2.imread("/home/mac/Desktop/temp/grab.jpg")
		# Grab path to current working directory
		CWD_PATH = os.getcwd()

		# Path to frozen detection graph .pb file, which contains the model that is used
		# for object detection.
		PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'OD.pb')

		# Path to label map file
		PATH_TO_LABELS = os.path.join(CWD_PATH,'ODT','labelmap.pbtxt')

		# Path to image
		PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

		# Number of classes the object detector can identify
		NUM_CLASSES = 1

		# Load the label map.
		# Label maps map indices to category names, so that when our convolution
		# network predicts `5`, we know that this corresponds to `king`.
		# Here we use internal utility functions, but anything that returns a
		# dictionary mapping integers to appropriate string labels would be fine
		label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
		categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
		category_index = label_map_util.create_category_index(categories)

		# Load the Tensorflow model into memory.
		detection_graph = tf.Graph()
		with detection_graph.as_default():
		    od_graph_def = tf.GraphDef()
		    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')

		    sess = tf.Session(graph=detection_graph)

		# Define input and output tensors (i.e. data) for the object detection classifier

		# Input tensor is the image
		image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

		# Output tensors are the detection boxes, scores, and classes
		# Each box represents a part of the image where a particular object was detected
		detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

		# Each score represents level of confidence for each of the objects.
		# The score is shown on the result image, together with the class label.
		detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
		detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

		# Number of objects detected
		num_detections = detection_graph.get_tensor_by_name('num_detections:0')

		# Load image using OpenCV and
		# expand image dimensions to have shape: [1, None, None, 3]
		# i.e. a single-column array, where each item in the column has the pixel RGB value
		image = imga
		image_expanded = np.expand_dims(image, axis=0)

		# Perform the actual detection by running the model with the image as input
		(boxes, scores, classes, num) = sess.run(
		    [detection_boxes, detection_scores, detection_classes, num_detections],
		    feed_dict={image_tensor: image_expanded})

		# Draw the results of the detection (aka 'visulaize the results')

		vis_util.visualize_boxes_and_labels_on_image_array(
		    image,
		    np.squeeze(boxes),
		    np.squeeze(classes).astype(np.int32),
		    np.squeeze(scores),
		    category_index,
		    use_normalized_coordinates=True,
		    line_thickness=4,
		    min_score_thresh=0.80)

		# All the results have been drawn on image. Now display the image.
		dim = (1000,1000)
		image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
		cv2.imshow('Object detector', image)
		#cv2.imshow( 'pro',np.squeeze(boxes))
		#cv2.imwrite('.jpg',image)
		# Press any key to close the image
		cv2.waitKey(0)

		# Clean up
		cv2.destroyAllWindows()

except Exception:
	pass




'''






## MACULAR HOLE AND MACULA (INCEPTION V3)
###############################################################

# This is needed since the notebook is stored in the object_detection folder.
try:

	# Import utilites
	i=0
	images = [cv2.imread(file) for file in glob.glob("/home/mac/Desktop/good images/*.jpg")]
	for i in range(0,714):
		# Name of the directory containing the object detection module we're using
		MODEL_NAME = 'W'
		IMAGE_NAME = 'NO MACULA TESTING/m3.jpg'

		# Grab path to current working directory
		CWD_PATH = os.getcwd()

		# Path to frozen detection graph .pb file, which contains the model that is used
		# for object detection.
		PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'mh.pb')

		# Path to label map file
		PATH_TO_LABELS = os.path.join(CWD_PATH,'W','labelmap.pbtxt')

		# Path to image
		PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

		# Number of classes the object detector can identify
		NUM_CLASSES = 1

		# Load the label map.
		# Label maps map indices to category names, so that when our convolution
		# network predicts `5`, we know that this corresponds to `king`.
		# Here we use internal utility functions, but anything that returns a
		# dictionary mapping integers to appropriate string labels would be fine
		label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
		categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
		category_index = label_map_util.create_category_index(categories)

		# Load the Tensorflow model into memory.
		detection_graph = tf.Graph()
		with detection_graph.as_default():
		    od_graph_def = tf.GraphDef()
		    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')

		    sess = tf.Session(graph=detection_graph)

		# Define input and output tensors (i.e. data) for the object detection classifier

		# Input tensor is the image
		image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

		# Output tensors are the detection boxes, scores, and classes
		# Each box represents a part of the image where a particular object was detected
		detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

		# Each score represents level of confidence for each of the objects.
		# The score is shown on the result image, together with the class label.
		detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
		detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

		# Number of objects detected
		num_detections = detection_graph.get_tensor_by_name('num_detections:0')

		# Load image using OpenCV and
		# expand image dimensions to have shape: [1, None, None, 3]
		# i.e. a single-column array, where each item in the column has the pixel RGB value

		image = images[i]
		image_expanded = np.expand_dims(image, axis=0)

		# Perform the actual detection by running the model with the image as input
		(boxes, scores, classes, num) = sess.run(
		    [detection_boxes, detection_scores, detection_classes, num_detections],
		    feed_dict={image_tensor: image_expanded})

		# Draw the results of the detection (aka 'visulaize the results')

		vis_util.visualize_boxes_and_labels_on_image_array(
		    image,
		    np.squeeze(boxes),
		    np.squeeze(classes).astype(np.int32),
		    np.squeeze(scores),
		    category_index,
		    use_normalized_coordinates=True,
		    line_thickness=4,
		    min_score_thresh=0.80)

		# All the results have been drawn on image. Now display the image.
		dim = (500,500)
		image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
		cv2.imshow('Object detector', image)
		#cv2.imshow( 'pro',np.squeeze(boxes))
		cv2.imwrite('aa.jpg',image)
		# Press any key to close the image
		#cv2.waitKey(0)

		# Clean up
		cv2.destroyAllWindows()
		img = cv2.imread('out.jpg',0)
		hist,bins = np.histogram(img.flatten(),256,[0,256])

		cdf = hist.cumsum()
		cdf_normalized = cdf * hist.max()/ cdf.max()

		'''plt.plot(cdf_normalized, color = 'b')
		plt.hist(img.flatten(),256,[0,256], color = 'r')
		plt.xlim([0,256])
		plt.legend(('cdf','histogram'), loc = 'upper left')
		plt.show()'''
		cdf_m = np.ma.masked_equal(cdf,0)
		cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
		cdf = np.ma.filled(cdf_m,0).astype('uint8')
		img2 = cdf[img]
		cv2.imshow('lol',img2)
		cv2.imwrite('/home/mac/Desktop/enhanced image/image%i.jpg'%i,img2)
		#cv2.waitKey(0)

		img = cv2.imread('out.jpg',0)
		clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(2,2))
		cl1 = clahe.apply(img)

		cv2.imwrite('/home/mac/Desktop/clahe/image%i.jpg'%i,cl1)
except Exception:
	pass	
images = [cv2.imread(file) for file in glob.glob("/home/mac/Desktop/enhanced image/*.jpg")]
d=0
for i in range(0,714):	

	try:
		cv2.imwrite('/home/mac/Desktop/temp/image1.jpg',images[i])
		img=cv2.imread("/home/mac/Desktop/temp/image1.jpg",0)	
		img = cv2.medianBlur(img,7)
		cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
		c1img = cv2.imread('mh1.jpg')
		circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
						    param1=50,param2=30,minRadius=45,maxRadius=75)

		circles = np.uint16(np.around(circles))
		for i in circles[0,:]:
			    # draw the outer circle
			    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
			    # draw the center of the circle
			    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
			    cv2.circle(c1img ,(i[0]+900,i[1]+980),2,(0,0,255),3)
			    cv2.circle(c1img,(i[0]+900,i[1]+980),i[2],(0,255,0),2)	

		cv2.imshow('detected circles',cimg)
		cv2.imwrite('/home/mac/Desktop/MH/image%d.jpg'%d,cimg)
		dim = (1000,1000)
		#c1img = cv2.resize(c1img, dim, interpolation = cv2.INTER_AREA)
		#cv2.imshow('lol',c1img)

		#cv2.waitKey(0)
		cv2.destroyAllWindows()
		d=d+1
	except Exception:
		print("image")
		print(i)
		print("Macular hole not present")
		pass	


