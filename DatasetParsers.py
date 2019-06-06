#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 18:15:54 2019

@author: ali
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np


EPOC_Brain_location = {"AF3" : 0,
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
MW_Brain_location = {"FP1" : 0}
N_Samples_device = {"MindWave" : 1024,
          "EPOC"     : 256 
          }
def Load_data(infile, device, data_size = 100, event_start_point = 67635, split_data = False):
    if device == "EPOC":
        Brain_location = EPOC_Brain_location
    elif device == "MindWave":
        Brain_location = MW_Brain_location
    N_locations = len(Brain_location)
    test_size = int(0.2 * data_size)
    train_size = data_size - test_size
    N_data = N_Samples_device[device]
    if N_locations > 1:
        arr = np.zeros([data_size, N_locations, N_data])
    else:
        arr = np.zeros([data_size, N_data], dtype = 'int32')
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
#        print(n)
        if N_locations > 1:
            arr[i//N_locations][channel] = list(map(float,temp))[:N_data]
        else:
            arr[i//N_locations] = list(map(float,temp))[:N_data]
        label[i//N_locations] = n
    if split_data == True:
        return arr[0:train_size], label[0:train_size], arr[train_size:data_size], label[train_size:data_size]
    return arr, label
def EPOC_Load_data(infile, data_size = 100, event_start_point = 67635):
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
        channel = EPOC_Brain_location[header[3]]
        temp = x[6].split(',')
        while len(temp) < N_data:
            temp.append('0')
        n = int(header[4])
        if(n != -1):
            arr[event][channel] = list(map(float,temp))[:N_data]
            label[event] = n
    return arr,label

def get_specific_data(arr, labels, l1, l2):
    data = [[] for i in range(10)]
    for i in range(len(arr)):
        data[labels[i]].append(arr[i])
    length = 0
    for i in range(10):
        length = length + len(data[i])
    tags = []
    content = []
    print(len(data[9]))
    print(len(data[2]))
    for i in range(length):
        if i < len(data[l1]):
            tags.append(0)
            content.append(data[l1][i])
        elif i < len(data[l1]) + len(data[l2]):
            tags.append(1)
            content.append(data[l2][i - len(data[l1])])
        else:
            tags.append(2)
    for i in range(10):
        if i != l1 and i != l2:
            for j in range(len(data[i])):
                content.append(data[i][j])
    return content, tags
    

infile = open("/home/ali/Documents/AI/EP1.01.txt")
#data, labels = EPOC_Load_data(infile, data_size = 40000)
content, tags = get_specific_data(data, labels, 0, 1)