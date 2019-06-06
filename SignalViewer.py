#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 16:02:37 2019

@author: ali
"""


from DatasetParsers import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.ndimage.filters import gaussian_filter1d


def plot_signal(data, labels, N , label):
    my_data = []
    for i in range(len(data)):
        if labels[i] == label:
            my_data.append(data[i])
    choices = np.random.randint(0, len(my_data), N * 12)
    data_to_be_shown = []
    for i in range(len(choices)):
        data_to_be_shown.append(my_data[choices[i]])
    fig = plt.figure(figsize = (30,30))
    x = np.arange(1024)
    for i in range(12):
        print(i)
        fig.add_subplot(4,3,i+1)
        for j in range(N):
            plt.plot(x, gaussian_filter1d(data_to_be_shown[i * N + j], sigma=100))

    
    plt.savefig("data.png")
    plt.show()

def plot_signal2(data, labels, N):
    my_data = [[0] for y in range(10)] 
    for i in range(len(data)):
        my_data[labels[i]].append(data[i])
    choices = np.random.randint(1, len(my_data), N)
    data_to_be_shown = [[0 for i in range(N)] for y in range(10)] 
    for j in range(10):
        for i in range(len(choices)):
            data_to_be_shown[j][i]=my_data[j][choices[i]]
    fig = plt.figure(figsize = (30,30))
    x = np.arange(1024)
    for i in range(10):
        print(i)
        fig.add_subplot(4,3,i+1)
        s = 0
        std = 0
        for j in range(N//2):
            plt.plot(x, gaussian_filter1d(data_to_be_shown[i][j], sigma=10))
#            s = s + np.average(gaussian_filter1d(data_to_be_shown[i][j], sigma=1))
#            std = std + np.std(gaussian_filter1d(data_to_be_shown[i][j], sigma=1))
#        print(s/N)
#        print(std/N)
    
    plt.savefig("data.png")
    plt.show()
    
def plot_PSD(data, labels, N):
    my_data = [[0] for y in range(10)] 
    for i in range(len(data)):
        my_data[labels[i]].append(data[i])
    choices = np.random.randint(1, len(my_data), N)
    data_to_be_shown = [[0 for i in range(N)] for y in range(10)] 
    for j in range(10):
        for i in range(len(choices)):
            data_to_be_shown[j][i]=my_data[j][choices[i]]
    fig = plt.figure()
    fig.set_size_inches([30,50])
    x = np.arange(1024)
    for i in range(10):
        print(i)
        fig.add_subplot(4,3,i+1)
        s = 0
        std = 0
        for j in range(N//2):
            x,y = signal.periodogram(data_to_be_shown[i][j], fs=125)
            plt.plot(x,y[0])
#           
    plt.savefig("PSD.png")
    plt.show()
infile = open("/home/ali/Documents/AI/EP1.01.txt")
data, labels = Load_data(infile, data_size = 54934, device = "EPOC")

plot_PSD(data, labels, N = 5000)