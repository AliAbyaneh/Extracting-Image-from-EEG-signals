#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:11:56 2019

@author: ali
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np

from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices())
print(tf.__version__)
import matplotlib.pyplot as plt
Brain_location = {"AF3" : 0,
                  "F7"  : 1,
                  "F3"  : 2,
                  "FC5" : 3,
                  "T7"  : 4,
                  "P7"  : 5,
                  "O1"  : 6,
                  "O2"  : 7,
                  "P8"  : 8, 
                  "T8"  : 9,
                  "FC6" : 10,
                  "F4"  : 11,
                  "F8"  : 12,
                  "AF4" : 13}
def Load_data(infile, data_size = 100, event_start_point = 67635):
    N_locations = 14
    test_size = int(0.2 * data_size)
    train_size = data_size - test_size
    N_data = 256
    arr = np.zeros([data_size, N_locations, N_data])
    label = np.zeros([data_size], dtype = 'int32')
    for i in range(N_locations*data_size):
        
        temp = infile.readline()
        if len(temp) < 10:
            break
        x = temp.split()
        header = x[0:6]
        event = int(header[1]) - event_start_point
        channel = Brain_location[header[3]]
        temp = x[6].split(',')
        while len(temp) < N_data:
            temp.append('0')
        n = int(header[4])
        if(n != -1):
            arr[event][channel] = list(map(float,temp))[:N_data]
            label[event] = n
    return arr[0:train_size], label[0:train_size], arr[train_size:data_size], label[train_size:data_size]
        
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 14, 256, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.sigmoid)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[1, 1], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 16384])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1 = 0.9, beta2 = 0.999)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def MLP(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 1, 14*256, 1])

  dense1 = tf.layers.dense(inputs=input_layer, units=4*1024, activation=tf.nn.relu)
  dense2 = tf.layers.dense(inputs=dense1, units=4*1024, activation=tf.nn.relu)
  dense3 = tf.layers.dense(inputs=dense2, units=4*1024, activation=tf.nn.relu)
  dense4 = tf.layers.dense(inputs=dense3, units=2*1024, activation=tf.nn.relu)
  
  dense = tf.layers.dense(inputs=dense4, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



infile = open("/home/ali/Documents/AI/EP1.01.txt")
train_data, train_labels, eval_data, eval_labels = Load_data(infile, data_size = 50000)
mean = np.mean(train_data)
train_data = train_data - mean
eval_data = eval_data - mean
print()
x = np.arange(0,len(eval_labels))
fig = plt.figure()
fig.set_size_inches(50,20)
plt.plot(x, eval_labels)
plt.show()
#classifier = tf.estimator.Estimator(
#    model_fn=MLP, model_dir="/tmp/concfasfwefvneasstvsdg_mol110")
#
#
## Set up logging for predictions
#tensors_to_log = {"probabilities": "softmax_tensor"}
#
#logging_hook = tf.train.LoggingTensorHook(
#    tensors=tensors_to_log, every_n_iter=50)
#with tf.device("/gpu:0"):
#    # Train the model
#    train_input_fn = tf.estimator.inputs.numpy_input_fn(
#        x={"x": train_data},
#        y=train_labels,
#        batch_size=1,
#        num_epochs=None,
#        shuffle=True)
#    # train one step and display the probabilties
#    classifier.train(
#        input_fn=train_input_fn,
#        steps=10000,
#        hooks=[logging_hook])
#    
#    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#        x={"x": eval_data},
#        y=eval_labels,
#        num_epochs=1,
#        shuffle=False)
#    
#    eval_results = classifier.evaluate(input_fn=eval_input_fn)
#    print(eval_results)
#    
#    