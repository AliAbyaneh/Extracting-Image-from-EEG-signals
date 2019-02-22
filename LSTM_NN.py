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
import keras.utils as KU

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


class MY_Generator(KU.Sequence):

    def __init__(self, _filenames, batch_size):
        self.filenames = _filenames
        self.batch_size = batch_size


    def __len__(self):
        return np.floor((len(self.filenames)) / float(self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
#        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        data     = np.zeros((self.batch_size, 2300), dtype = int)
        Ys       = np.zeros((self.batch_size, 569), dtype=bool)
        outputstageDict = defineOutputs()
        for j in range(0,self.batch_size):  
            key = batch_x[j][batch_x[j].find('n0'):batch_x[j].find('n0') + 9]
            
            df = pd.read_csv("MindBigData-Imagenet/" + batch_x[j], sep = ',', header = None)
            df = df.values
            data_AF3 = (np.array([], dtype= int))
            data_AF4 = (np.array([], dtype= int))
            data_T7  = (np.array([], dtype= int))
            data_T8  = (np.array([], dtype= int))
            data_Pz  = (np.array([], dtype= int))
            
            for i in range(1,len(df[0])):
                data_AF3 = np.append(data_AF3, int(df[0][i]))
            
            for i in range(1,len(df[1])):
                data_AF4 = np.append(data_AF4, int(df[1][i]))
            
            for i in range(1,len(df[2])):
                data_T7  = np.append(data_T7, int(df[2][i]))
                
            for i in range(1,len(df[3])):
                data_T8  = np.append(data_T8, int(df[3][i]))
                
            for i in range(1,len(df[4])):
                data_Pz  = np.append(data_Pz, int(df[4][i]))
            temp = np.array([], dtype= 'f')
            temp = np.append(data_AF3, data_AF4)
            temp = np.append(temp, data_T7)
            temp = np.append(temp, data_T8)
            temp = np.append(temp, data_Pz)
            data[j][:len(temp)] = temp      
            Ys[j][int(outputstageDict[key])] = True
            
        return data, Ys


def make_data():
    max_size = 200
    num_validation_samples = 40
    filenames = find_csv_filenames("MindBigData-Imagenet")
    outputstageDict = defineOutputs()
    #for name in filenames:
     # print (name)
    print(filenames[1])
    data     = np.zeros((max_size - num_validation_samples, 2300), dtype = 'f')
    test_data = np.zeros((num_validation_samples, 2300), dtype = 'f')
    Ys       = np.zeros((max_size - num_validation_samples, 569))
    Ys_test  = np.zeros((num_validation_samples, 569))
#    print(data)
    for j in range(0,max_size):  
        key = filenames[j][filenames[j].find('n0'):filenames[j].find('n0') + 9]
        print(j)
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
        if j >= max_size - num_validation_samples:
            test_data[j - (max_size - num_validation_samples)][:len(temp)] = temp
            Ys_test[j - (max_size - num_validation_samples)][outputstageDict[key]] = 1
        else:
            data[j][:len(temp)] = temp            
            Ys[j][outputstageDict[key]] = 1

        #print(data_Pz)
        #plt.plot(data_T7)
        #plt.show()
    return data,Ys, test_data,Ys_test
    
#x_train,y_train, x_test, y_test = make_data()
    

filenames = find_csv_filenames("MindBigData-Imagenet")
num_training_samples = 9000
num_validation_samples = 1000

maxlen = 1920
max_features = 3000
batch_size = 8
num_epochs = 5
my_training_batch_generator = MY_Generator(filenames[0:num_training_samples], batch_size)
my_validation_batch_generator = MY_Generator(filenames[num_training_samples:num_training_samples + num_validation_samples], batch_size)
#a, b = my_validation_batch_generator.__getitem__(0)
#print(my_training_batch_generator)
model = Sequential()
model.add(Embedding(max_features, 1280))
model.add(LSTM(1280, dropout=0.5, recurrent_dropout=0.2))
model.add(Dropout(0.1))
model.add(Dense(569, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')

#hist = model.fit(x_train, y_train,
#          batch_size=batch_size,
#          epochs=20,
#          validation_data=(x_test, y_test))

model.fit_generator(generator=my_training_batch_generator,
                                          steps_per_epoch=(num_training_samples // batch_size),
                                          epochs=num_epochs,
                                          verbose=1,
                                          validation_data=my_validation_batch_generator,
                                          validation_steps=(num_validation_samples // batch_size),
                                          use_multiprocessing=True,
                                          workers=16,
                                          max_queue_size=32)
