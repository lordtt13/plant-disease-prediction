# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 17:07:01 2020

@author: Tanmay Thakur
"""
import cv2
import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array


default_image_size = tuple((256, 256))

def convert_image_to_array(image_dir):
    image = cv2.imread(image_dir)
    if image is not None :
        image = cv2.resize(image, default_image_size)   
        return img_to_array(image)
    else :
        return np.array([])