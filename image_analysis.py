#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 19:42:24 2020

@author: alessandro
"""

import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from scipy import ndimage


def extract_frames(path,video_name):
    path = path.rstrip('/')
    if not os.path.exists(path + '/' + video_name):
        raise FileNotFoundError('No such file or directory')
    
    fol = path + '/' + video_name.rsplit('.',1)[0]
    os.mkdir(fol)
    cur_dir = os.curdir
    os.replace(path + '/' + video_name, fol + '/' + video_name)
    
    os.chdir(fol)
    os.system(f'ffmpeg -i {video_name} frames_%03d.jpg')
    
    os.chdir(cur_dir)
    
    
def extend(array, new_shape=(960,1600)):
    '''
    Extend a gray scale image into a bigger one
    '''
    new_array = np.zeros(new_shape,dtype=np.uint8)
    offset_x = (new_shape[0] - array.shape[0])//2
    offset_y = (new_shape[1] - array.shape[1])//2
    
    new_array[offset_x:(offset_x + array.shape[0]), offset_y:(offset_y + array.shape[1])] = array
    return new_array