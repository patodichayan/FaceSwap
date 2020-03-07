import cv2
import dlib
import numpy as np
import copy

landmark_coord = np.zeros((68, 2), dtype='int')

def get_facial_landmarks(img):
        
    # load shape predictor model
    model_path = 'dlib_model/shape_predictor_68_face_landmarks.dat'

    # load the detector and the predictor. 
    # predictor accepts pre-trained model as input
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)

    img_ = copy.deepcopy(img)
    img_gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    rects = detector(img_gray, 1)

    for i, rect in enumerate(rects):
        landmarks = predictor(img_gray, rect)

        # reshape landmarks to (68X2)
        # landmark_coord = np.zeros((68, 2), dtype='int')

        for i in range(68):
            landmark_coord[i] = (landmarks.part(i).x, landmarks.part(i).y)

        # draw bounding box on face
        #cv2.rectangle(img_, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 255), 4)

        # draw facial landmarks
        
        jaw = landmark_coord[0:17]
        left_ebrow = landmark_coord[17:22]
        right_ebrow = landmark_coord[22:27]
        nose = landmark_coord[27:36]
        eye_left = landmark_coord[36:42]
        eye_right = landmark_coord[42:48]
        lips = landmark_coord[48:]

        cv2.polylines(img_, [jaw], False, (0, 255, 0), 1)
        cv2.polylines(img_, [left_ebrow], False, (0, 255, 0), 1)
        cv2.polylines(img_, [right_ebrow], False, (0, 255, 0), 1)
        cv2.polylines(img_, [nose], False, (0, 255, 0), 1)
        cv2.polylines(img_, [eye_left], False, (0, 255, 0), 1)
        cv2.polylines(img_, [eye_right], False, (0, 255, 0), 1)
        cv2.polylines(img_, [lips], False, (0, 255, 0), 1)

    return img_ , landmark_coord
