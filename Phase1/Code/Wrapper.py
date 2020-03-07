#!/usr/bin/evn python

"""
CMSC733 Spring 2020: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Starter Code

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
# Add any python libraries here

#Helping Functions.




def main():
	cap = cv2.VideoCapture('../../Data/Video2.mp4')
	
	while (cap.isOpened()):
		ret, frame = cap.read()
		if ret == False:
			break
		
		frame, features = get_facial_landmarks(frame)
		print(features)		
		cv2.imshow("",frame)
		cv2.waitKey(50)
	cap.release()
	cv2.destroyAllWindows()
		
		
if __name__ == '__main__':
  main()
 
