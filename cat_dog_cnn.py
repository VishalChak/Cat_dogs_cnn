#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 23:32:24 2017

@author: vishal
"""


files_path = '/home/vishal/ML/datasets/Cat_Dog/'

import cv2
import numpy as np
from tqdm import tqdm
import os
from random import shuffle
import tensorflow as tf


import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression 

TRAIN_DIR = files_path+'train'
TEST_DIR = files_path+'test'
IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = 'dog-vs-cat-convnet'

# IMage preprossing

def create_label(image_name):
    ''' create one hot encoder vector for image name'''
    word_label = image_name.split('.')[-3]
    if word_label == 'cat':
        return np.array([1,0])
    elif word_label == 'dog':
        return np.array([0,1])
    

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        path = os.path.join(TRAIN_DIR, img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img_data), create_label(img)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        IMG_NUM = img.split('.')[0]
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img_data), IMG_NUM])
    
    shuffle(testing_data)
    np.save('testing.npy',testing_data)
    return testing_data

train_data = create_train_data()
test_data = create_test_data()

train = train_data[:-500]
test = train_data[-500:]

X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE,1)
y_train = [ i[1] for i in train]


X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE,IMG_SIZE,1)
y_test = [i[1] for i in test]


# convolutional net


tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name ='input')

convnet = conv_2d(convnet, 32,5, activation = 'relu')
convnet = max_pool_2d(convnet,5)


convnet = conv_2d(convnet, 64, 5, activation= 'relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation = 'relu')
convnet = dropout(convnet,0.8)

convnet = fully_connected(convnet, 2, activation = 'softmax')
convnet = regression(convnet, optimizer = 'adam', learning_rate = LR, loss = 'categorical_crossentropy', name = 'target')

model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)

model.fit(X_train,y_train,n_epoch=10,validation_set =None,show_metric=True, run_id=MODEL_NAME)
#model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10, validation_set=({'input': X_test}, {'targets': y_test}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

