from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Lambda, Reshape, Dropout, ZeroPadding2D
from keras.optimizers import RMSprop
import numpy as np 
import matplotlib.pyplot as pyplot
import os
import cv2


# Loading images
data = "./data/"



# Creating the model
# VGG16 Model studied from: https://neurohive.io/en/popular-networks/vgg16/

def add_convolutional_block(model, layers, filters):
    for i in range(filters):
        model.add(ZeroPadding2D(1,1))
        model.add(Conv2D(filters, 3, 3, activation='relu'))
    model.add(MaxPool2D((2,2), strides=(2,2)))

def add_fully_connected_block(model):
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

# Model implementation is based on VGG16 CNN Architecture
def buildModel():
    model = Sequential()
    model.add(Lambda(lambda x : x, input_shape=(3, 106, 106)))
    add_convolutional_block(model, 2, 64)
    add_convolutional_block(model, 2, 128)
    add_convolutional_block(model, 3, 256)
    add_convolutional_block(model, 3, 512)
    add_convolutional_block(model, 3, 512)

    model.add(Flatten())
    add_fully_connected_block(model)
    add_fully_connected_block(model)
    model.add(Dense(37, activation = 'sigmoid'))
    return model

optimizer = RMSprop(lr=1e-6)
model = buildModel()
model.compile(loss='mean_squared_error', optimizer=optimizer)