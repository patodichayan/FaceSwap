import cv2
import dlib
import numpy as np
import copy
import os
import sys

sys.path.append("prnet/")
from api import PRN
from utils.cv_plot import plot_kpt




def drawFacialLandmarks(img, landmarkCoord):
    
    jaw = landmarkCoord[0:17]
    left_ebrow = landmarkCoord[17:22]
    right_ebrow = landmarkCoord[22:27]
    nose = landmarkCoord[27:36]
    eye_left = landmarkCoord[36:42]
    eye_right = landmarkCoord[42:48]
    lips = landmarkCoord[48:]

    cv2.polylines(img, [jaw], False, (0, 255, 0), 1)
    cv2.polylines(img, [left_ebrow], False, (0, 255, 0), 1)
    cv2.polylines(img, [right_ebrow], False, (0, 255, 0), 1)
    cv2.polylines(img, [nose], False, (0, 255, 0), 1)
    cv2.polylines(img, [eye_left], False, (0, 255, 0), 1)
    cv2.polylines(img, [eye_right], False, (0, 255, 0), 1)
    cv2.polylines(img, [lips], False, (0, 255, 0), 1)
    
    return img
    
def getFacialLandmarks(isDlib, img_, numFaces=1):
    
    img = copy.deepcopy(img_)

    # use dlib or PrNetfor prediction of facial landmarks 
    if isDlib:    
        # load shape predictor model
        model_path = 'dlib_model/shape_predictor_68_face_landmarks.dat'

        # load the detector and the predictor. 
        # predictor accepts pre-trained model as input
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(model_path)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(img_gray, 1)
        
        # store landmark locations of both faces
        landmarkCoordAll = []
        
        # iterate through the points in both faces
        for r, rect in enumerate(rects):
            landmarks = predictor(img_gray, rect)

            # reshape landmarks to (68X2)
            landmarkCoord = np.zeros((68, 2), dtype='int')

            for i in range(68):
                landmarkCoord[i] = (landmarks.part(i).x, landmarks.part(i).y)
            landmarkCoordAll.append(landmarkCoord)
            
            # draw bounding box on face
            cv2.rectangle(img, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 255), 0)

            # draw facial landmarks
            img_ = drawFacialLandmarks(img, landmarkCoord)

    else:
        # prn uses dlib for face detection and its own trained model for prediction of facial landmarks
        prn = PRN(is_dlib = True, prefix='prnet/')
        landmarkCoord = []

        [h, w, c] = img.shape
        if c>3:
            img = img[:,:,:3]

        if img.shape[0] == img.shape[1]:
            img = resize(img, (256,256))
            pos = prn.net_forward(img/255.) # input image has been cropped to 256x256
        else:
            posList = []
            for i in range(numFaces):
                pos = prn.process(img, i)
                posList.append(pos)
        
        landmarkCoordAll = []
        for i, pos in enumerate(posList):

            if pos is None:
                return img_, landmarkCoordAll

            # get landmark points of face
            landmarkCoord = prn.get_landmarks(pos)
            img_ = plot_kpt(img_, landmarkCoord)

            landmarkCoord = landmarkCoord[:, 0:2]
            landmarkCoordAll.append(landmarkCoord)
    
    return img_, landmarkCoordAll
