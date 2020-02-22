# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 17:38:52 2020

@author: Tanmay Thakur
"""
import pickle
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import get_model


EPOCHS = 25
INIT_LR = 1e-3
BS = 32
width = 256
height = 256
depth = 3

image_list,label_list = pickle.load(open("data.pickle", "rb"))

image_size = len(image_list)

label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)
pickle.dump(label_binarizer, open('label_transform.pkl', 'wb'))
n_classes = len(label_binarizer.classes_)

np_image_list = np.array(image_list, dtype=np.float16) / 225.0

print("[INFO] Spliting data to train, test")
x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state = 42) 

aug = ImageDataGenerator(
    rotation_range = 25, width_shift_range = 0.1,
    height_shift_range = 0.1, shear_range = 0.2, 
    zoom_range = 0.2,horizontal_flip = True, 
    fill_mode = "nearest")

model = get_model(height, width, depth, n_classes)

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["accuracy"])

print("[INFO] training network...")

history = model.fit_generator(
    aug.flow(x_train, y_train, batch_size=BS),
    validation_data = (x_test, y_test),
    steps_per_epoch = len(x_train) // BS,
    epochs=EPOCHS, verbose=1
    )