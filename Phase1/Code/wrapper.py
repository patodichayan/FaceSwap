#!/usr/bin/evn python

"""
CMSC733 Spring 2020: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project2: : FaceSwap.

Author(s): 
Chayan Kumar Patodi (ckp1804@terpmail.umd.edu)
University of Maryland, College Park

Saket Seshadri Gudimetla Hanumath (saketsgh@umd.edu)
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
import cv2
from glob import glob
import argparse
import os
import matplotlib.pyplot as plt
import random
import copy
import dlib
from facial_landmark_det import get_facial_landmarks
from warp_spline import*
from pb import*

# Add any python libraries here

#Helping Functions.


def main():

	Parser = argparse.ArgumentParser()
	Parser.add_argument('--video', default='../../Data/Video1.mp4', help='Provide Video Name with path here')
	Parser.add_argument('--target',default = '../../Data/Video5.mp4',help='Provide Image to be swapped in the image.')
	Parser.add_argument('--mode',default = 'tps',help='Provide Method of image transformation: delaunay(deln), TPS(tps)')

	Args = Parser.parse_args()
	Video = Args.video
	Target = Args.target
	Mode = Args.mode

	cap = cv2.VideoCapture(Video)

	print("Number of Frames in the Video:" , int(cap.get(cv2.cv2.CAP_PROP_FRAME_COUNT)))
	
	#As of now working for single target image and 1 source video , thus outside loop.

	_, img = (cv2.VideoCapture(Target)).read()
	#img = cv2.imread("../../Data/stark.jpg")
	#img = automatic_brightness_and_contrast(img)
	image, features1 , faces1 = get_facial_landmarks(img)


	count = 0
	while (cap.isOpened()):
		ret, frame = cap.read()
		if ret == False:
			break

		
		frame, features2, faces2 = get_facial_landmarks(frame)
		#frame = automatic_brightness_and_contrast(frame)
		count = count+1		
		
		if faces2 !=1:
			print("Not enough keypoints to determine the face")
			continue

		print("frame{}".format(count))

		parametersx = estimate_parameters(features1,features2,"x")
		parametersy = estimate_parameters(features1,features2,"y")

		result = tps(img,frame,features1,features2,parametersx,parametersy)
		cv2.imwrite("Frame{}.png".format(count),result)
		
		# cv2.imshow("",result)
		# cv2.waitKey(50)
		
	cap.release()
	cv2.destroyAllWindows()
		
		
if __name__ == '__main__':
  main()
 
