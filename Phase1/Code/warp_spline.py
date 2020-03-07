import numpy as np
import cv2


def U(r):


	sq = np.square(r)
	U = sq * np.log(sq)
	
	if np.isnan(U) == True:
		return 0
	else:	
		return U

 
	
