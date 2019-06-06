#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:40:16 2019

@author: ali
"""
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout
from keras.layers import LSTM

from os import listdir
import os
import _csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#print(csv.__file__)
def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]
def defineOutputs():

    d = {}
    with open("dataDict.txt") as f:
        for line in f:
           (key, val) = line.split(',')
           d[key] = int(val)
#    print(d)
    return d

def make_data():
    max_size = 3000
    test_size = 600
    filenames = find_csv_filenames("MindBigData-Imagenet")
    outputstageDict = defineOutputs()
    #for name in filenames:
     # print (name)
    print(filenames[1])
    data     = np.zeros((max_size - test_size, 2300), dtype = 'f')
    test_data = np.zeros((test_size, 2300), dtype = 'f')
    Ys       = np.zeros((max_size - test_size, 569))
    Ys_test  = np.zeros((test_size, 569))
#    print(data)
    for j in range(0,max_size):  
        print(j)
        key = filenames[j][filenames[j].find('n0'):filenames[j].find('n0') + 9]
        
        df = pd.read_csv("MindBigData-Imagenet/" + filenames[j], sep = ',', header = None)
        df = df.values
        data_AF3 = np.array([], dtype= 'f')
        data_AF4 = np.array([], dtype= 'f')
        data_T7  = np.array([], dtype= 'f')
        data_T8  = np.array([], dtype= 'f')
        data_Pz  = np.array([], dtype= 'f')
        
        for i in range(1,len(df[0])):
            data_AF3 = np.append(data_AF3, df[0][i])
        
        for i in range(1,len(df[1])):
            data_AF4 = np.append(data_AF4, df[1][i])
        
        for i in range(1,len(df[2])):
            data_T7  = np.append(data_T7, df[2][i])
            
        for i in range(1,len(df[3])):
            data_T8  = np.append(data_T8, df[3][i])
            
        for i in range(1,len(df[4])):
            data_Pz  = np.append(data_Pz, df[4][i])
        temp = np.array([], dtype= 'f')
        temp = np.append(data_AF3, data_AF4)
        temp = np.append(temp, data_T7)
        temp = np.append(temp, data_T8)
        temp = np.append(temp, data_Pz)
        if j >= max_size - test_size:
            test_data[j - (max_size - test_size)][:len(temp)] = temp
            Ys_test[j - (max_size - test_size)][outputstageDict[key]] = 1
        else:
            data[j][:len(temp)] = temp

        #print(data_Pz)
        #plt.plot(data_T7)
        #plt.show()
    return data,Ys, test_data,Ys_test
    
x_train,y_train, x_test, y_test = make_data()

maxlen = 1920
max_features = 10000
batch_size = 64
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 1280))
model.add(LSTM(1280, dropout=0.5, recurrent_dropout=0.2))
model.add(LSTM(640, dropout=0.5, recurrent_dropout=0.2))
model.add(LSTM(320, dropout=0.5, recurrent_dropout=0.2))
model.add(Dropout(0.1))
model.add(Dense(569, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')

hist = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=2,
          validation_data=(x_test, y_test))
