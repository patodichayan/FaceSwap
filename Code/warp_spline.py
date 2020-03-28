import numpy as np
import cv2
import math
import sys
import copy

#References Used : http://profs.etsmtl.ca/hlombaert/thinplates/


def U(r):
	return (r**2)*(math.log(r**2))


def generate_mask(img2,points,r):

	hull = []
	Index = cv2.convexHull(np.array(points), returnPoints = False)


	for i in xrange(0, len(Index)):
		hull.append(points[int(Index[i])])


	#Creating a mask.

	mask = np.zeros((r[3], r[2], 3), dtype = np.float32)

	points_t = []

	#Setting points with reference.
	for i in xrange(len(hull)):
		points_t.append(((hull[i][0]-r[0]),(hull[i][1]-r[1])))


	cv2.fillConvexPoly(mask, np.int32(points_t), (1.0, 1.0, 1.0), 16, 0)

	# cv2.imshow("mask",mask)
	# cv2.waitKey()
	# cv2.destroyAllWindows()

	return mask, hull


def estimate_parameters(points1,points2,axis, isDlib):

	if axis == "x":
		points_axis = points1[:,0]
	if axis == "y":
		points_axis = points1[:,1]

	p = len(points2)
	K = np.zeros((p,p),np.float32)
	P = np.zeros((p,3),np.float32)
	Z = np.zeros([3,3])

	for i in xrange(p):
		for j in xrange(p):
			a = points2[i,:]
			b = points2[j,:]
			K[i,j] = U(np.linalg.norm((a-b),ord =2)+sys.float_info.epsilon)

	P = np.hstack((points2,np.ones((p,1))))
	Mat = np.vstack((np.hstack((K,P)),np.hstack((P.transpose(),Z))))

	# The regularization parameter value works differently for PRnet and Dlib
	if(isDlib=="False"):
		lambda_ = 220
	else:
		lambda_ = 10 ** -8

	T = np.linalg.inv(Mat + lambda_*np.identity(p+3))
	target = np.concatenate((points_axis,[0,0,0]))
	parameters = np.matmul(T,target)

	return parameters


def blend(img1,img2,hull):

    # Calculate Mask

    mask = np.zeros(img2.shape, dtype = img2.dtype)
    cv2.fillConvexPoly(mask, np.int32(hull), (255, 255, 255))
    r = cv2.boundingRect(np.float32([hull]))
    center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))

    # Clone seamlessly.

    output = cv2.seamlessClone(np.uint8(img1), img2, mask, center, cv2.NORMAL_CLONE)

    return output


def tps(img1,img2,points1,points2,isDlib):

	parametersx = estimate_parameters(points1,points2,"x", isDlib)
	parametersy = estimate_parameters(points1,points2,"y", isDlib)

	img2_copy = copy.deepcopy(img2)
	points1 = np.round(points1).astype(np.int32)
	points2 = np.round(points2).astype(np.int32)
	p = len(points1)
	r = cv2.boundingRect(np.float32([points2]))
	mask, hull = generate_mask(img2,points2,r)

	a1_x = parametersx[p+2]
	a2_x = parametersx[p+1]
	a3_x = parametersx[p]

	a1_y = parametersy[p+2]
	a2_y = parametersy[p+1]
	a3_y = parametersy[p]


	#Creating the image for face that has to be swapped.
	new_img = np.copy(mask)

	for i in xrange(new_img.shape[1]):
		for j in xrange(new_img.shape[0]):
			t = 0
			l = 0
			n = i+ r[0]
			m = j+ r[1]
			b = [n,m]
			for k in xrange(p):

				# epsilon is used to avoid NAN situations.

				a = points2[k,:]
				t = t+parametersx[k]*(U(np.linalg.norm((a-b),ord =2)+sys.float_info.epsilon))
				l = l+parametersy[k]*(U(np.linalg.norm((a-b),ord =2)+sys.float_info.epsilon))

			x = int(a1_x + a3_x*n + a2_x*m + t)
			y = int(a1_y + a3_y*n + a2_y*m + l)

			x = min(max(x, 0), img1.shape[1] - 1)
			y = min(max(y, 0), img1.shape[0] - 1)

			new_img[j,i] = img1[y,x,:]

	new_img = new_img * mask

	#Pasting Face from one image to the other.
	img2[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img2[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( (1.0, 1.0, 1.0) - mask )
	img2[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img2[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] + new_img

	#Blending.

	output = blend(img2,img2_copy,hull)

	return output
