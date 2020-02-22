# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 17:38:52 2020

@author: Tanmay Thakur
"""
import pickle
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from utils import train_val_generator
from model import get_model


EPOCHS = 25
INIT_LR = 1e-3
BS = 32
width = 256
height = 256
depth = 3

image_list,label_list = pickle.load(open("data.pickle", "wb"))

image_size = len(image_list)

label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)
pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))
n_classes = len(label_binarizer.classes_)

print(label_binarizer.classes_)

np_image_list = np.array(image_list, dtype=np.float16) / 225.0

model = get_model(height, width, depth, n_classes)

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# distribution
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
# train the network
print("[INFO] training network...")

history = model.fit_generator(
    aug.flow(x_train, y_train, batch_size=BS),
    validation_data = (x_test, y_test),
    steps_per_epoch = len(x_train) // BS,
    epochs=EPOCHS, verbose=1
    )