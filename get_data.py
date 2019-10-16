import cv2
import numpy as np

from keras.utils import to_categorical
import os

def get_data(data_folder):

    print('loading data')
    x = []
    y = []
    idx=1
    for folder in os.listdir(data_folder):
        for img in os.listdir(os.path.join(data_folder,folder)):
            if folder=='NORMAL': 
                x.append(os.path.join(data_folder,folder,img))
                y.append(to_categorical(0,3))

            elif img.find('bacteria') >= 0:
                x.append(os.path.join(data_folder,folder,img))
                y.append(to_categorical(1,3))

            elif img.find('virus') >= 0:
                x.append(os.path.join(data_folder,folder,img))
                y.append(to_categorical(2,3))

            
            if idx % 200 == 0:
                print('{} image path loaded'.format(idx))
            idx+=1
    print('{} image path loaded'.format(idx))
            
    return x, y
