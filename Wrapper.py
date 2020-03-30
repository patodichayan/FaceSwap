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

import sys
sys.path.append("Code/")


from facial_landmark_det import*
from warp_triangulation import*
from warp_spline import*
from video import*
from PRnetSwap import*


def main():
	# if you do not want to see tensorflow warnings uncomment the next line 
	# tensorflow_shutup()
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--video', default='Data/Data1.mp4', help='Provide Video Name with path here')
	Parser.add_argument('--target',default = 'Data/stinson.jpg', help='Provide Image to be swapped in the image.')
	Parser.add_argument('--method',default = 'tps', help='Provide Method of image transformation: delaunay(deln), Thin Plate Spline(tps), Position Map Regression Network(prnet)')
	Parser.add_argument('--mode',default = 1, help='If swapping 1 image in a video, use 1, If swapping 2 faces in a single video, use 2')
	Parser.add_argument('--isDlib',default = "True", help='True if dlib should be used prediction of facial landmarks. False for using PrNet for the same.')

	Args = Parser.parse_args()
	Video = Args.video
	Target = Args.target
	method = Args.method
	mode = Args.mode
	isDlib = Args.isDlib

	cap = cv2.VideoCapture(Video)
	print("Number of Frames in the Video:" , int(cap.get(cv2.cv2.CAP_PROP_FRAME_COUNT)))


	if not os.path.exists('Frames'):
		os.makedirs('Frames')

	if mode == 1:

		numFaces = 1

		img = cv2.imread(Target)
		landmarksImgTarget, features1 = getFacialLandmarks(isDlib, img, numFaces)

		if len(features1) != 1:
			print("Not enough keypoints to determine the face. Try another face.")
			exit()

		features1 = features1[0]
		count = 0

		while (cap.isOpened()):

			ret, frame = cap.read()
			if ret == False:
				break

			if method == 'tps':

				landmarksImgFrame, features2 = getFacialLandmarks(isDlib, frame, numFaces)
				count += 1

				if len(features2) !=1:
					print("Not enough keypoints to determine the face")
					continue

				features2 = features2[0]
				print("frame{}".format(count))
				result = tps(img,frame,features1,features2, isDlib)

			elif method == 'deln':

				landmarksImgFrame, features2 = getFacialLandmarks(isDlib, frame, numFaces)
				count += 1

				if len(features2) !=1:
					print("Not enough keypoints to determine the face")
					continue

				features2 = features2[0]
				print("frame{}".format(count))
				result = SwapFaceWithImg(frame, features2, img, features1)

			else:
				pos, result = prnetSwap(frame, img, numFaces)
				count += 1
				if(pos is None):
					continue
				print("frame{}".format(count))

			# showImage(result, "swapped")
			cv2.imwrite("Frames/Frame{}.png".format(count),result)

	else:

		count = 0
		numFaces = 2
		while (cap.isOpened()):

			ret, frame = cap.read()
			if ret == False:
				break

			if method == 'tps':

				frame_copy = copy.deepcopy(frame)
				facialLandmarksImg, features = getFacialLandmarks(isDlib, frame, numFaces)
				count += 1
				if len(features) !=2:
					print("Not enough keypoints to determine the face")
					continue

				features1 = features[0]
				features2 = features[1]
				print("frame{}".format(count))
				output = tps(frame,frame,features1,features2, isDlib)
				result = tps(frame_copy,output,features2,features1, isDlib)

			elif method == 'deln':
				count += 1
				frame_copy = copy.deepcopy(frame)
				facialLandmarksImg, features = getFacialLandmarks(isDlib, frame, numFaces)
				# showImage(facialLandmarksImg, "landmarks")
				if len(features) !=2:
					print("Not enough keypoints to determine the face")
					continue

				features1 = features[0]
				features2 = features[1]
				print("frame{}".format(count))
				result = SwapTwoFacesInVid(frame, features)

			else:
				count += 1
				pos, result = prnetSwap(frame, frame, numFaces)
				if(pos is None):
					continue
				print("frame{}".format(count))
			# showImage(result, "swapped")
			cv2.imwrite("Frames/Frame{}.png".format(count), result)

	# convert the images to video
	convert(method)
	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
