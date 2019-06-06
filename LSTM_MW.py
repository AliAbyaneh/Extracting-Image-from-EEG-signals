#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 18:29:47 2019

@author: ali
"""

from DatasetParsers import *
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices())
print(tf.__version__)





infile = open("/home/ali/Documents/AI/MW.txt")
data, labels = Load_data(infile, data_size = 54934, device = "MindWave")
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(1024, 10,input_length = data.shape[1]))
model.add(tf.keras.layers.LSTM(100))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Activation('softmax'))
model.summary()



model.compile(
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-2, ),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['sparse_categorical_accuracy', 'accuracy']
)

# model.save('CNN1D_model_EPOC.h5')  # creates a HDF5 file 'my_model.h5'
filepath = "./CNN1D_model_EPOC/saved-model.hdf5"
checkPointer = tf.keras.callbacks.ModelCheckpoint(filepath, verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=500)
tensorBoard = tf.keras.callbacks.TensorBoard(log_dir='./CNN1D_model_EPOC/logs', write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)
    

History = model.fit(
    x = data,
    y = labels,
    batch_size = 512,
    epochs=5000,
    validation_split = 0.15,
    callbacks = [checkPointer,tensorBoard]
)
import pickle
with open('./CNN1D_model_EPOC/trainHistoryDict', 'wb') as file_pi:
    pickle.dump(History.history, file_pi)

