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
# Add any python libraries here

import numpy as np
import cv2
from glob import glob
import argparse
import os
import copy
import dlib
from facial_landmark_det import getFacialLandmarks
from warp_triangulation import SwapFaceWithImg, SwapTwoFacesInVid, showImage
from warp_spline import*
from video import*

def main():

	Parser = argparse.ArgumentParser()
	Parser.add_argument('--video', default='../../Data/Video9.mp4', help='Provide Video Name with path here')
	Parser.add_argument('--target',default = '../../Data/stark.jpg',help='Provide Image to be swapped in the image.')
	Parser.add_argument('--method',default = 'tps',help='Provide Method of image transformation: delaunay(deln), TPS(tps) or PRNet(PR)')
	Parser.add_argument('--mode',default = 1,help='If swapping 1 image in a video, use 1, If swapping 2 faces in a single video, use 2')

	Args = Parser.parse_args()
	Video = Args.video
	Target = Args.target
	method = Args.method
	mode = Args.mode

	cap = cv2.VideoCapture(Video)
	print("Number of Frames in the Video:" , int(cap.get(cv2.cv2.CAP_PROP_FRAME_COUNT)))


	if not os.path.exists('../Frames'):
		os.makedirs('../Frames')
	
	#As of now working for single target image and 1 source video , thus outside loop.
	if mode == 1:

		img = cv2.imread(Target)
		_, features1 = getFacialLandmarks(img)

		if len(features1) != 1:
			print("Not enough keypoints to determine the face. Try another face.")
			exit()

		features1 = features1[0]

		count = 0
		while (cap.isOpened()):
			ret, frame = cap.read()
			if ret == False:
				break

			_, features2 = getFacialLandmarks(frame)
			count += 1 		
		
			if len(features2) !=1:
				print("Not enough keypoints to determine the face")
				continue
			
			features2 = features2[0]
			print("frame{}".format(count))

			if method == 'tps':
	
				result = tps(img,frame,features1,features2)

			elif method == 'deln':

				result, _ = SwapFaceWithImg(frame,img)

			else:
				""

			cv2.imwrite("../Frames/Frame{}.png".format(count),result)

			# cv2.imshow("",result)
			# cv2.waitKey(50)
			# cv2.destroyAllWindows()

	else:

		count = 0
		while (cap.isOpened()):
			ret, frame = cap.read()

			if ret == False:
				break

			frame_copy = copy.deepcopy(frame)
			_, features = getFacialLandmarks(frame)
			count += 1

			if len(features) !=2:
				print("Not enough keypoints to determine the face")
				continue

			features1 = features[0]
			features2 = features[1]
			print("frame{}".format(count))

			if method == 'tps':
	
				output = tps(frame,frame,features1,features2)
				result = tps(frame_copy,output,features2,features1)

			elif method == 'deln':

				result, flag = SwapTwoFacesInVid(frame)

			else:
				""

			cv2.imwrite("../Frames/Frame{}.png".format(count),result)
			# cv2.imshow("",result)
			# cv2.waitKey(50)
			# cv2.destroyAllWindows()
	convert(method)
	# files = glob.glob('../Frames/*.png', recursive=True)
	# for f in files:
	# 	os.remove(f)
	cap.release()
	cv2.destroyAllWindows()
		
		
if __name__ == '__main__':
  main()
 
