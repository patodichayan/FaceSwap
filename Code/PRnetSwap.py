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
sys.path.append("prnet/")
from api import PRN
from utils.render import render_texture
import cv2


def prnetSwap(image, ref_image):

    prn = PRN(is_dlib = True, prefix='prnet/')
    [h, w, _] = image.shape

    #-- 1. 3d reconstruction -> get texture.
    final_pos1 = prn.process(image, 0)
    final_pos2 = prn.process(image, 1)

    final_pos = []
    final_pos.append(final_pos1)
    final_pos.append(final_pos2)

    if final_pos == None:
        return None, image

    if len(final_pos)==2:
        output = prnetOne(prn,image,ref_image,final_pos[0],final_pos[1],h,w)
        output = prnetOne(prn,output,ref_image,final_pos[1],final_pos[0],h,w)
    else:
        return None, image

    return final_pos,output

def prnetOne(prn,image,ref_image,pos,ref_pos,h,w):

    vertices = prn.get_vertices(pos)
    image = image/255.
    texture = cv2.remap(image, pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
    ref_image = ref_image/255.
    ref_texture = cv2.remap(ref_image, ref_pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
    ref_vertices = prn.get_vertices(ref_pos)
    new_texture = ref_texture#(texture + ref_texture)/2.

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
    out = cv2.seamlessClone((new_image*255).astype(np.uint8), (image*255).astype(np.uint8), (face_mask*255).astype(np.uint8), center, cv2.NORMAL_CLONE)

    return out
