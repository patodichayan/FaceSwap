import cv2
import numpy as np
import copy
import random

from warp_triangulation import draw_triangles, triangulation
from facial_landmark_det import get_facial_landmarks

def main():

	file_path = '../../Data/'
	cap = cv2.VideoCapture(file_path+'together5.mp4')
	# while(True):
	ret, img = cap.read()
	img_copy = copy.deepcopy(img)

	# load the image and calculate facial landmarks using dlib
	face_ldmrk_img, landmark_coord_all = get_facial_landmarks(img)

	# display the facial landmarks
	cv2.imshow("OP_face_ldmrk", face_ldmrk_img)

	# perform triangulation on 1st face and gather their locations in the landmark_coord array
	triang_fc1_img, triangleIdList = triangulation(img_copy, landmark_coord_all[0])

	# show the output image with the delaunay triangulation on face 1
	cv2.imshow("OP_delaunay_fc1", triang_fc1_img)

	# retrieve the locations of the triangles in the second face
	triangleList = []
	for t in triangleIdList:
	    p1_id, p2_id, p3_id = t[0], t[1], t[2]
	    pt1 = landmark_coord_all[1][p1_id][0]
	    pt2 = landmark_coord_all[1][p2_id][0]
	    pt3 = landmark_coord_all[1][p3_id][0]

	    triangleList.append([pt1[0], pt1[1], pt2[0], pt2[1], pt3[0], pt3[1]])

	# draw triangles on face2
	triang_fc2_img = draw_triangles(triang_fc1_img, triangleList, face2=True, landmark_coord=None)

	# show the output image with the delaunay triangulation
	cv2.imshow("OP_delaunay_fc12", triang_fc2_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()