import cv2
import dlib
import numpy as np
import copy


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
    
def getFacialLandmarks(img_=None):
    
    if(img_ is None):    
        # load image
        file_path = '../../Data/'
        cap = cv2.VideoCapture(file_path+'saket3.mp4')
        ret, img = cap.read()
    
    img = copy.deepcopy(img_)    
    # load shape predictor model
    model_path = 'dlib_model/shape_predictor_68_face_landmarks.dat'

    # load the detector and the predictor. 
    # predictor accepts pre-trained model as input
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(img_gray, 1)
    
    # store landmark locations of both faces
    landmark_coord_all = []
    
    # iterate through the points in both faces
    for r, rect in enumerate(rects):
        landmarks = predictor(img_gray, rect)

        # reshape landmarks to (68X2)
        landmark_coord = np.zeros((68, 2), dtype='int')

        for i in range(68):
            landmark_coord[i] = (landmarks.part(i).x, landmarks.part(i).y)
        landmark_coord_all.append(landmark_coord)
        
        # draw bounding box on face
        cv2.rectangle(img, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 255), 0)

        # draw facial landmarks
        img_ = drawFacialLandmarks(img, landmark_coord)
        
    return img_, landmark_coord_all