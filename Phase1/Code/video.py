import cv2
import numpy as np
import glob
import os 


def convert():
	if not os.path.exists('../Results'):
	    os.makedirs('../Results')


	img_array = []
	for filename in glob.glob('../Frames/*.png'):
	    img = cv2.imread(filename)
	    height, width, layers = img.shape
	    size = (width,height)
	    img_array.append(img)


	out = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
	 
	for i in range(len(img_array)):
	    out.write(img_array[i])
	out.release()