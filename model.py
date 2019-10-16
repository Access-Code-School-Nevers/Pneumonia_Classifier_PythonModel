from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization, Reshape
from tensorflow.keras.layers import Conv2D, SeparableConv2D, GlobalAveragePooling2D
from tensorflow.compat.v1.keras.optimizers import Adam
import tensorflowjs as tfjs

import cv2
import numpy as np

def get_model(input_shape=(640,480,3)):
    model=Sequential()
    model.add(Input((None,None,3)))
    model.add(Reshape(input_shape))

    model.add(Conv2D(input_shape=(640,480,3),use_bias=False, filters=32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    model.add(SeparableConv2D(use_bias=False, filters=32, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu', depth_multiplier=3))
    model.add(BatchNormalization())

    model.add(SeparableConv2D(use_bias=False, filters=32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', depth_multiplier=3))
    model.add(SeparableConv2D(use_bias=False, filters=64, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu', depth_multiplier=3))
    model.add(BatchNormalization())

    model.add(SeparableConv2D(use_bias=False, filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', depth_multiplier=3))
    model.add(SeparableConv2D(use_bias=False, filters=128, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu', depth_multiplier=3))
    model.add(BatchNormalization())

    model.add(SeparableConv2D(use_bias=False, filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', depth_multiplier=3))
    model.add(SeparableConv2D(use_bias=False, filters=256, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu', depth_multiplier=3))
    model.add(BatchNormalization())

    model.add(SeparableConv2D(use_bias=False, filters=256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', depth_multiplier=3))
    model.add(SeparableConv2D(use_bias=False, filters=512, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu', depth_multiplier=3))
    model.add(BatchNormalization())

    model.add(SeparableConv2D(use_bias=False, filters=512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', depth_multiplier=3))
    model.add(SeparableConv2D(use_bias=False, filters=1024, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu', depth_multiplier=3))
    model.add(BatchNormalization())

    model.add(GlobalAveragePooling2D())

    model.add(Dropout(0.25))

    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.25))

    model.add(Dense(3, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

    return model


