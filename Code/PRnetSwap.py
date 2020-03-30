import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
import ast
import matplotlib.pyplot as plt
import argparse
import sys
sys.path.append("Code/prnet/")
from api import PRN
from utils.render import render_texture
import cv2

def tensorflow_shutup():
    """
    Make Tensorflow less verbose
    """
    try:
        # noinspection PyPackageRequirements
        import os
        from tensorflow import logging
        logging.set_verbosity(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Monkey patching deprecation utils to shut it up! Maybe good idea to disable this once after upgrade
        # noinspection PyUnusedLocal
        def deprecated(date, instructions, warn_once=True):
            def deprecated_wrapper(func):
                return func
            return deprecated_wrapper

        from tensorflow.python.util import deprecation
        deprecation.deprecated = deprecated

    except ImportError:
        pass

def prnetSwap(image, ref_image, numFaces):

    prn = PRN(is_dlib = True, prefix='Code/prnet/')
    [h, w, _] = image.shape
    posList = []

    if(numFaces==1):
        # get the landmarks
        pos1 = prn.process(image, 0)
        pos2 = prn.process(ref_image, 0)

        posList.append(pos1)
        posList.append(pos2)

        if (posList is None) or (pos1 is None) or (pos2 is None):
            return None, image
        elif len(posList)==2:
            output = prnetOne(prn, image, ref_image, posList[0], posList[1], h, w)
        else:
            return None, image
        return posList, output

    else:
        # get the landmarks
        pos1 = prn.process(image, 0)
        pos2 = prn.process(image, 1)

        posList.append(pos1)
        posList.append(pos2)

        if (posList is None) or (pos1 is None) or (pos2 is None):
            return None, image

        elif len(posList)==2 :
            output = prnetSwapOneFace(prn, image, ref_image, posList[0], posList[1], h, w)
            output = prnetSwapOneFace(prn, output, ref_image, posList[1], posList[0], h, w)
        else:
            return None, image

        return posList, output

def prnetSwapOneFace(prn, image, ref_image, pos, ref_pos, h, w):

    vertices = prn.get_vertices(pos)
    image = image/255.
    texture = cv2.remap(image, pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))

    ref_image = ref_image/255.
    ref_texture = cv2.remap(ref_image, ref_pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
    ref_vertices = prn.get_vertices(ref_pos)
    new_texture = ref_texture #(texture + ref_texture)/2.

    #-- 3. remap to input image.(render)
    vis_colors = np.ones((vertices.shape[0], 1))
    face_mask = render_texture(vertices.T, vis_colors.T, prn.triangles.T, h, w, c = 1)
    face_mask = np.squeeze(face_mask > 0).astype(np.float32)

    new_colors = prn.get_colors_from_texture(new_texture)
    new_image = render_texture(vertices.T, new_colors.T, prn.triangles.T, h, w, c = 3)
    new_image = image*(1 - face_mask[:,:,np.newaxis]) + new_image*face_mask[:,:,np.newaxis]

    # Possion Editing for blending image
    vis_ind = np.argwhere(face_mask>0)
    vis_min = np.min(vis_ind, 0)
    vis_max = np.max(vis_ind, 0)
    center = (int((vis_min[1] + vis_max[1])/2+0.5), int((vis_min[0] + vis_max[0])/2+0.5))
    out = cv2.seamlessClone((new_image*255).astype(np.uint8), (image*255).astype(np.uint8), (face_mask*255).astype(np.uint8), center, cv2.MIXED_CLONE)

    return out
