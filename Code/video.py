import cv2
import numpy as np
import glob
import os


def convert(method):
	if not os.path.exists('Results/'):
	    os.makedirs('Results/')


	img_array = []
	images_ = glob.glob('Frames/*.png')
	images_.sort(key=lambda f: int(filter(str.isdigit, f)))

	for filename in images_:
	    img = cv2.imread(filename)
	    height, width, layers = img.shape
	    size = (width,height)
	    img_array.append(img)


	out = cv2.VideoWriter('Results/Output{}.mp4'.format(method),cv2.VideoWriter_fourcc(*'mp4v'), 24, (size))

	for i in range(len(img_array)):
	    out.write(img_array[i])
	out.release()
