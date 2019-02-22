from os import listdir
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
    print(d)
    return d

def make_data():
    filenames = find_csv_filenames("MindBigData-Imagenet")
    outputstageDict = defineOutputs()
    #for name in filenames:
     # print (name)
    print(filenames[0])
    data     = np.zeros((len(filenames), 2300), dtype = 'f')
    Ys       = np.zeros((len(filenames), 569))
    print(data)
    for j in range(0,len(filenames)):  
        print(j)
        key = filenames[j][filenames[j].find('n0'):filenames[j].find('n0') + 9]
        Ys[j][outputstageDict[key]] = 1
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
        data[j][:len(temp)] = temp
    
        #print(data_Pz)
        #plt.plot(data_T7)
        #plt.show()
make_data()