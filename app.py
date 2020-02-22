# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 20:14:05 2020

@author: Tanmay Thakur
"""
import pickle
import numpy as np
import tensorflow as tf

from utils import convert_image_to_array


model = tf.keras.models.load_model("model.h5")

test_image = "PlantVillage/Potato/Potato___healthy/00fc2ee5-729f-4757-8aeb-65c3355874f2___RS_HL 1864.JPG"

img = convert_image_to_array(test_image)

img = np.array(img, dtype=np.float16) / 225.0

img = np.reshape(img,(1,img.shape[0], img.shape[1], img.shape[2]))

labels = pickle.load(open("label_transform.pkl","rb")).classes_

pred = model.predict(img)