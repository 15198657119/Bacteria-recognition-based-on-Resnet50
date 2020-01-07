#!/usr/bin/env python
# coding: utf-8

import os, sys
import numpy as np
import scipy
from keras.applications import ResNet50
from keras.preprocessing import image
from scipy import ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import *
from tensorflow.keras.preprocessing import *
from tensorflow.keras.applications.resnet50 import *
from PIL import Image
import random



def DataSet():
    train_path_Boletus = 'jun_data/Boletus/'
    train_path_checkenchop = 'jun_data/Chickenchop/'
    train_path_Cookedbacteria = 'jun_data/Cookedbacteria/'
    train_path_Greenbacterium = 'jun_data/Greenbacterium/'

    test_path_Boletus = 'jun_data/Boletus/'
    test_path_checkenchop = 'jun_data/Chickenchop/'
    test_path_Cookedbacteria = 'jun_data/Cookedbacteria/'
    test_path_Greenbacterium = 'jun_data/Greenbacterium/'
    # test_path_glue = '../my_nn/dataset/test/glue/'
    # test_path_medicine = '../my_nn/dataset/test//medicine/'

    imglist_train_Boletus = os.listdir( train_path_Boletus)
    imglist_train_checkenchop = os.listdir(train_path_checkenchop)
    imglist_train_Cookedbacteria = os.listdir(train_path_Cookedbacteria)
    imglist_train_Greenbacterium = os.listdir(train_path_Greenbacterium )

    imglist_test_Boletus = os.listdir( test_path_Boletus )
    imglist_test_checkenchop = os.listdir(test_path_checkenchop)
    imglist_test_Cookedbacteria = os.listdir(test_path_Cookedbacteria)
    imglist_test_Greenbacterium = os.listdir(test_path_Greenbacterium)

    X_train = np.empty((len(imglist_train_Boletus) + len(imglist_train_checkenchop)+ len(imglist_train_Cookedbacteria)+ len(imglist_train_Greenbacterium), 224, 224, 3))
    Y_train = np.empty((len(imglist_train_Boletus) + len(imglist_train_checkenchop)+ len(imglist_train_Cookedbacteria)+ len(imglist_train_Greenbacterium), 4))
    count = 0
    for img_name in imglist_train_Boletus:
        img_path =  train_path_Boletus + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0

        X_train[count] = img
        Y_train[count] = np.array((1, 0))
        count += 1

    for img_name in imglist_train_checkenchop:
        img_path =train_path_checkenchop + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0

        X_train[count] = img
        Y_train[count] = np.array((0, 1))
        count += 1

    for img_name in imglist_train_Cookedbacteria:
        img_path =  train_path_Cookedbacteria + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0

        X_train[count] = img
        Y_train[count] = np.array((1, 0))
        count += 1

    for img_name in imglist_train_Greenbacterium:
        img_path =  train_path_Greenbacterium + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0

        X_train[count] = img
        Y_train[count] = np.array((0, 1))
        count += 1

    X_test = np.empty((len(imglist_test_glue) + len(imglist_test_medicine), 224, 224, 3))
    Y_test = np.empty((len(imglist_test_glue) + len(imglist_test_medicine), 2))
    count = 0
    for img_name in imglist_test_glue:
        img_path = test_path_glue + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0

        X_test[count] = img
        Y_test[count] = np.array((1, 0))
        count += 1

    for img_name in imglist_test_medicine:
        img_path = test_path_medicine + img_name
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0

        X_test[count] = img
        Y_test[count] = np.array((0, 1))
        count += 1

    index = [i for i in range(len(X_train))]
    random.shuffle(index)
    X_train = X_train[index]
    Y_train = Y_train[index]

    index = [i for i in range(len(X_test))]
    random.shuffle(index)
    X_test = X_test[index]
    Y_test = Y_test[index]

    return X_train, Y_train, X_test, Y_test


X_train, Y_train, X_test, Y_test = DataSet()
print('X_train shape : ', X_train.shape)
print('Y_train shape : ', Y_train.shape)
print('X_test shape : ', X_test.shape)
print('Y_test shape : ', Y_test.shape)

# # model


model = ResNet50(
    weights=None,
    classes=4
)

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# # train


model.fit(X_train, Y_train, epochs=1, batch_size=6)

# # evaluate


model.evaluate(X_test, Y_test, batch_size=32)

# # save


model.save('my_resnet_model.h5')

# # restore


model = tf.keras.models.load_model('my_resnet_model.h5')

# # test


img_path = "../my_nn/dataset/test/medicine/IMG_20190717_135408_BURST91.jpg"

img_path = "../my_nn/dataset/test/glue/IMG_20190717_135425_BURST91.jpg"

img = image.load_img(img_path, target_size=(224, 224))

plt.imshow(img)
img = image.img_to_array(img) / 255.0
img = np.expand_dims(img, axis=0)  # 为batch添加第四维

print(model.predict(img))
np.argmax(model.predict(img))
