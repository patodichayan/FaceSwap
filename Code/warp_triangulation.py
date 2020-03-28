import cv2
# import dlib
import numpy as np
import copy
import random
import scipy.interpolate

from facial_landmark_det import getFacialLandmarks

def drawTriangles(img, triangleList, face2=False, landmark_coord=None):

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

def DelaunayTriangulation(img_, landmark_coord):
    # get the bounding rectangle of the landmarks
    img = copy.deepcopy(img_)
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

    img, triangleIdList = drawTriangles(img, triangleList, False, landmark_coord)

    return img, triangleIdList, triangleList

def getBoundingBoxPts(pt1, pt2, pt3):
    xlist = [pt1[0], pt2[0], pt3[0]]
    ylist = [pt1[1], pt2[1], pt3[1]]

    x_top_left = np.min(xlist)
    y_top_left = np.min(ylist)

    x_bottom_right = np.max(xlist)
    y_bottom_right = np.max(ylist)

    xx, yy = np.meshgrid(range(x_top_left, x_bottom_right), range(y_top_left, y_bottom_right))
    xx = xx.flatten()
    yy = yy.flatten()
    ones = np.ones(xx.shape, dtype='int')

    # convert to homogenuous coordinates
#     bb_pts = np.vstack((xx, yy, ones))
    return xx, yy, ones

def getBarycentricCoord(pt1, pt2, pt3):

    # define the B_delta matrix
    BarycentricMatrix = np.array([[pt1[0], pt2[0], pt3[0]], [pt1[1], pt2[1], pt3[1]], [1, 1, 1]])

    # get all the points in the ROI(triangle but since do not have a way to locate points in a triangle we will consider a rectangle)
    xx, yy, ones = getBoundingBoxPts(pt1, pt2, pt3)
    BoundingBoxPts = np.vstack((xx, yy, ones))

    # calculate the barycentric coordinates
    BarycentricCoord = np.dot(np.linalg.pinv(BarycentricMatrix), BoundingBoxPts)
    alpha = BarycentricCoord[0]
    beta = BarycentricCoord[1]
    gamma = BarycentricCoord[2]

    valid_alpha = np.where(np.logical_and(alpha > -0.1, alpha< 1.1))[0]
    valid_beta = np.where(np.logical_and(beta > -0.1, beta < 1.1))[0]
    valid_gamma = np.where(np.logical_and(alpha+beta+gamma> -0.1, alpha+beta+gamma < 1.1))[0]

    valid_al_bet = np.intersect1d(valid_alpha, valid_beta)
    inside_pts_loc = np.intersect1d(valid_al_bet, valid_gamma)

    # get all points inside the triangle
    BoundingBoxPts = BoundingBoxPts.T
    pts_in_triang = BoundingBoxPts[inside_pts_loc]

    # get their corresponding barycentric coordinates
    alpha_inside = alpha[inside_pts_loc]
    beta_inside = beta[inside_pts_loc]
    gamma_inside = gamma[inside_pts_loc]

    return alpha_inside, beta_inside, gamma_inside, pts_in_triang

def convexHullFace(img_, landmarks):
    img = copy.deepcopy(img_)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)

    points = np.array(landmarks, np.int32)
    convexhull = cv2.convexHull(points)
    cv2.polylines(img, [convexhull], True, (255, 255, 255), 1)
    cv2.fillConvexPoly(mask, convexhull, 255)
    convexHullFace = cv2.bitwise_and(img, img, mask=mask)

    return convexHullFace, convexhull

def showImage(img, title="FaceSwap", save=False):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyWindow(title)
    if(save):
        cv2.imwrite(title+".png", img)

def swapFaces(img_, srcImg, srcLandmarks, srcTriangles, dstImg, dstTriangles, dstLandmarks):
    img = copy.deepcopy(img_)
    x0Src, y0Src, w, h = cv2.boundingRect(srcLandmarks)

    warpedImgSrc = np.zeros((img.shape), np.uint8)
    # for each triangle in face 2 calculate barycentric coordinates and use those to obtain warped points of face 1 that will be copied to face 2
    for t1, t2 in zip(srcTriangles, dstTriangles):

        # get vertices of source face
        pt1Src = (t1[0], t1[1])
        pt2Src = (t1[2], t1[3])
        pt3Src = (t1[4], t1[5])

        # get vertices of destination face
        pt1Dst = (t2[0], t2[1])
        pt2Dst = (t2[2], t2[3])
        pt3Dst = (t2[4], t2[5])

        # get Barycentric coordinates
        alpha, beta, gamma, ptsInDstTriangle = getBarycentricCoord(pt1Dst, pt2Dst, pt3Dst)

        ptsInDstTriangle = ptsInDstTriangle[:, 0:2]

        # if no points found then move on to next triangle
        if(np.shape(ptsInDstTriangle)[0]==0):
            continue

        # apply the barycentric coordinate equation on Source face points
        BarycentricMatrix = np.array([[pt1Src[0], pt2Src[0], pt3Src[0]], [pt1Src[1], pt2Src[1], pt3Src[1]], [1, 1, 1]])
        BarycentricCoord = np.vstack((alpha, beta))
        BarycentricCoord = np.vstack((BarycentricCoord, gamma))
        WarpedPtsSrc = np.dot(BarycentricMatrix, BarycentricCoord)

        # convert back to cartesian from homogenuous
        WarpedPtsSrc = WarpedPtsSrc.T
        WarpedPtsSrc[:, 0] = WarpedPtsSrc[:, 0]/WarpedPtsSrc[:, 2]
        WarpedPtsSrc[:, 1] = WarpedPtsSrc[:, 1]/WarpedPtsSrc[:, 2]
        WarpedPtsSrc = WarpedPtsSrc[:, 0:2]

        # extract the points of Src
        xWarpedSrc = WarpedPtsSrc[:, 0]
        yWarpedSrc = WarpedPtsSrc[:, 1]

        xlist = range(0, srcImg.shape[1])
        ylist = range(0, srcImg.shape[0])

        # for blue, green and red seperately
        interp_blue = scipy.interpolate.interp2d(xlist, ylist, srcImg[:, :, 0], kind='linear')
        interp_green = scipy.interpolate.interp2d(xlist, ylist, srcImg[:, :, 1], kind='linear')
        interp_red = scipy.interpolate.interp2d(xlist, ylist, srcImg[:, :, 2], kind='linear')

        # apply the interpolation functions to obtain new pixel value and replace the corresponding pixel in Dest

        for p, x, y in zip(ptsInDstTriangle, xWarpedSrc, yWarpedSrc):

            # shift points of Src
            x = x - x0Src
            y = y - y0Src
            blue_val = interp_blue(x, y)
            red_val = interp_red(x, y)
            green_val = interp_green(x, y)

            # assign the interpolated value
            img[p[1], p[0]] = (blue_val, green_val, red_val)
            warpedImgSrc[p[1], p[0]] = (blue_val, green_val, red_val)

    return img, warpedImgSrc

def getOutsidePts(img, FaceLandmarks):

    # calculate outside points in order to perform face trimming for clean swapping output
    # get convex hull
    convexHullFaceImg, convexhullPtsFace = convexHullFace(img, FaceLandmarks)
    img_copy = copy.deepcopy(img)
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)
    maskRect = np.zeros_like(img_gray)
    cv2.fillConvexPoly(mask, convexhullPtsFace, 255)

    # calculate bounding rectangle on Face
    xiFace, yiFace, w, h = cv2.boundingRect(convexhullPtsFace)
    xRect, yRect = np.meshgrid(range(xiFace, xiFace+w+1), range(yiFace, yiFace+h+1))
    xRect = xRect.flatten()
    yRect = yRect.flatten()
    rect = np.vstack((xRect, yRect))
    rect = rect.T
    faceCenter = (xiFace+int(w/2), yiFace+int(h/2))

    cv2.fillConvexPoly(maskRect, rect, 255)

    # subtract bounding rectangle and convex hull to obtain outside points that will be trimmed off
    ptsOutsideFace = cv2.subtract(maskRect, mask)
    ptsOutsideFace = np.where(ptsOutsideFace==255)
    xptsOutsideFace = ptsOutsideFace[1]
    yptsOutsideFace = ptsOutsideFace[0]

    ptsOutsideFace = np.vstack((xptsOutsideFace, yptsOutsideFace))
    ptsOutsideFace = ptsOutsideFace.T

    return ptsOutsideFace, mask, faceCenter

def SwapTwoFacesInVid(img, landmarkCoordAll):

    landmarksFace1 = landmarkCoordAll[0]
    landmarksFace2 = landmarkCoordAll[1]

    # Extract face 1
    landmarksFace1 = np.round(landmarksFace1).astype(np.int32)
    x0Face1, y0Face1, w, h = cv2.boundingRect(landmarksFace1)
    x1Face1 = x0Face1 + w
    y1Face1 = y0Face1 + h
    imgFace1 = img[y0Face1:y1Face1, x0Face1:x1Face1]

    # calculate outside points in order to perform face trimming for clean swapping output
    ptsOutsideFace1, mask1, face1Center = getOutsidePts(img, landmarksFace1)
    ptsOutsideFace2, mask2, face2Center = getOutsidePts(img, landmarksFace2)

    # Extract face 2
    landmarksFace2 = np.round(landmarksFace2).astype(np.int32)
    x0Face2, y0Face2, w, h = cv2.boundingRect(landmarksFace2)
    x1Face2 = x0Face2 + w
    y1Face2 = y0Face2 + h
    imgFace2 = img[y0Face2:y1Face2, x0Face2:x1Face2]

    # perform triangulation on 1st face and gather their locations in the landmark_coord array
    triangulationOneFace, triangleIdList, triangleListFace1 = DelaunayTriangulation(img, landmarksFace1)

    # retrieve the locations of the triangles in face 2
    triangleListFace2 = []
    for t in triangleIdList:
        p1_id, p2_id, p3_id = t[0], t[1], t[2]
        pt1 = landmarksFace2[p1_id][0]
        pt2 = landmarksFace2[p2_id][0]
        pt3 = landmarksFace2[p3_id][0]

        triangleListFace2.append([pt1[0], pt1[1], pt2[0], pt2[1], pt3[0], pt3[1]])

    # draw triangles on face2
    triangulationTwoFaces = drawTriangles(triangulationOneFace, triangleListFace2, face2=True, landmark_coord=None)

    # gather all intensities at outside locations and then paste on swapped image
    outIntensitiesFace1 = img[ptsOutsideFace1[:, 1], ptsOutsideFace1[:, 0]]
    outIntensitiesFace2 = img[ptsOutsideFace2[:, 1], ptsOutsideFace2[:, 0]]

    swap1, warpedImgFace2 = swapFaces(img, imgFace1, landmarksFace1, triangleListFace1, imgFace2, triangleListFace2, landmarksFace2)
    swap1[ptsOutsideFace2[:, 1], ptsOutsideFace2[:, 0]] = outIntensitiesFace2
    warpedImgFace2[ptsOutsideFace2[:, 1], ptsOutsideFace2[:, 0]] = outIntensitiesFace2

    # perform seamless cloning
    swap1Cloned = cv2.seamlessClone(np.uint8(swap1), img, mask2, face2Center, cv2.NORMAL_CLONE)

    swap2, warpedImgFace1 = swapFaces(swap1Cloned, imgFace2, landmarksFace2, triangleListFace2, imgFace1, triangleListFace1, landmarksFace1)
    swap2[ptsOutsideFace1[:, 1], ptsOutsideFace1[:, 0]] = outIntensitiesFace1
    warpedImgFace1[ptsOutsideFace1[:, 1], ptsOutsideFace1[:, 0]] = outIntensitiesFace1

    # perform seamless cloning
    swap2Cloned = cv2.seamlessClone(swap2, swap1Cloned, mask1, face1Center, cv2.NORMAL_CLONE)

    # showImage(warpedImgFace1, "warpedFace1")
    # showImage(warpedImgFace2, "warpedFace2")

    return swap2Cloned

def SwapFaceWithImg(img, landmarksFace1, srcImg, landmarksFace2):


    # Extract face 1(Destination)
    landmarksFace1 = np.round(landmarksFace1).astype(np.int32)
    x0Face1, y0Face1, w, h = cv2.boundingRect(landmarksFace1)
    x1Face1 = x0Face1 + w
    y1Face1 = y0Face1 + h
    imgFace1 = img[y0Face1:y1Face1, x0Face1:x1Face1]

    ptsOutsideFace1, mask1, face1Center = getOutsidePts(img, landmarksFace1)

    # Extract face 2(Source)
    landmarksFace2 = np.round(landmarksFace2).astype(np.int32)
    x0Face2, y0Face2, w, h = cv2.boundingRect(landmarksFace2)
    x1Face2 = x0Face2 + w
    y1Face2 = y0Face2 + h
    imgFace2 = srcImg[y0Face2:y1Face2, x0Face2:x1Face2]

    # perform triangulation on 1st face and gather triangle locs
    triangulationFace1, triangleIdList, triangleListFace1 = DelaunayTriangulation(img, landmarksFace1)

    # retrieve the locations of the triangles in face 2
    triangleListFace2 = []
    for t in triangleIdList:
        p1_id, p2_id, p3_id = t[0], t[1], t[2]
        pt1 = landmarksFace2[p1_id][0]
        pt2 = landmarksFace2[p2_id][0]
        pt3 = landmarksFace2[p3_id][0]

        triangleListFace2.append([pt1[0], pt1[1], pt2[0], pt2[1], pt3[0], pt3[1]])

    # draw triangles on face2
    triangulationFace2 = drawTriangles(copy.deepcopy(srcImg), triangleListFace2, face2=True, landmark_coord=None)

    # trianglesFace1Img, _ =  convexHullFace(triangulationOneFace, landmarksFace1)
    # trianglesFace2Img, _ = convexHullFace(triangulationTwoFaces, landmarksFace2)

    # showImage(triangulationFace1, "trianglesFace1", save=True)
    # showImage(triangulationFace2, "trianglesFace2", save=True)

    # gather all intensities at outside locations and then paste on swapped image
    outIntensitiesFace1 = img[ptsOutsideFace1[:, 1], ptsOutsideFace1[:, 0]]
    swap, warpedImgSrc = swapFaces(img, imgFace2, landmarksFace2, triangleListFace2, imgFace1, triangleListFace1, landmarksFace1)
    swap[ptsOutsideFace1[:, 1], ptsOutsideFace1[:, 0]] = outIntensitiesFace1
    warpedImgSrc[ptsOutsideFace1[:, 1], ptsOutsideFace1[:, 0]] = 0

    # perform seamless cloning
    swapCloned = cv2.seamlessClone(np.uint8(swap), img, mask1, face1Center, cv2.MIXED_CLONE)

    return swapCloned
