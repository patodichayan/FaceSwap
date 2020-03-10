import cv2
# import dlib
import numpy as np
import copy
import random

def draw_triangles(img, triangleList, face2=False, landmark_coord=None):
    
    # for getting random colors for plotting triangles
    colors = []
    color_ind = [random.choice(range(-90, 40)) for i in range(70)]
    
    blue_shades = [(150+c, 0, 0) for c in color_ind]
    green_shades = [(0, 175+c, 0) for c in color_ind]
    red_shades = [(0, 0, 140+c) for c in color_ind]

    random.shuffle(blue_shades)
    random.shuffle(red_shades)
    random.shuffle(green_shades)
    for b, g, r in zip(blue_shades, green_shades, red_shades):
        colors.append((b[0], g[1], r[2]))
    
    # define a list to store the index of which points are used in triangle formation
    triangleIdList = []
    for t in triangleList:
        
        # get the three vertices of each triangle to show
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        
        color = random.choice(colors)
        cv2.line(img, pt1, pt2, color, 1, cv2.LINE_AA, 0)
        cv2.line(img, pt2, pt3, color, 1, cv2.LINE_AA, 0)
        cv2.line(img, pt3, pt1, color, 1, cv2.LINE_AA, 0)
        
        if(not face2):
            # now find out where these points occur so that we can draw corresponding
            # triangles in second face
            pt1_id = np.where((landmark_coord==pt1).all(axis=1))
            pt2_id = np.where((landmark_coord==pt2).all(axis=1))
            pt3_id = np.where((landmark_coord==pt3).all(axis=1))

            triangleIdList.append([pt1_id, pt2_id, pt3_id])
    
    if(not face2):
        return img, triangleIdList
    else:
        return img

def triangulation(img, landmark_coord):
    
    # get the bounding rectangle of the landmarks
    h, w = img.shape[0], img.shape[1]
    rect = (0, 0, w, h)
    
    # Create an instance of Subdiv2D with the rectangle 
    # obtained in the previous step
    subdiv = cv2.Subdiv2D(rect)

    # insert the points into subdiv             
    for coord in landmark_coord:
        subdiv.insert((coord[0], coord[1]))
        
    # draw the delaunay triangles
    triangleList = subdiv.getTriangleList()
    
    img, triangleIdList = draw_triangles(img, triangleList, False, landmark_coord)
    
    return img, triangleIdList
