#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers import MaxPooling2D, ZeroPadding2D
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation
from keras import backend as K
import keras.backend.tensorflow_backend as ktf
from keras.utils import np_utils
import os
import struct
import cv2
from persistence import *


''' This module shows a letter from the EMNIST data set at different 
convolutional and pooling layers. 

Run it using  the command "python print_convs.py"

'''

def get_session(gpu_fraction=0.333):
    '''Prevents memory errors with tensorflow-gpu.

    Solves a bug whereby tensorflow was using 100% of gpu and crashing.
    '''

    gpu_options = \
        tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                      allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# Model
ktf.set_session(get_session())
input_shape = (1, 28, 28)
number_classes = 10
pool_size = (2, 2)  # size of pooling area for max pooling
model = Sequential()

# Convolutional input Layer.
# Takes an input vector of size (28,28,1) so 784 inputs

model.add(Convolution2D(
    12,
    (5, 5),
    activation='relu',
    data_format='channels_first',
    input_shape=input_shape,
    init='he_normal',
    ))
convout1 = Activation('relu')
model.add(convout1)

# Max pooling layer using pool_size (2,2)

poolout2 = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')
model.add(poolout2)

# Convolutional layer
# 25 filters with kernel size of (5,5)

model.add(Convolution2D(25, (5, 5), activation='relu',
          data_format='channels_first', init='he_normal'))
convout3 = Activation('relu')
model.add(convout3)

# Max pooling layer using pool_size (2,2)

poolout4 = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')
model.add(poolout4)

# Flatten to 1d tensor
# flatten5 = Flatten()

model.add(Flatten())

# Regular densely-connected layer
# ReLU activation
# initialize weights using he_normal

model.add(Dense(180, activation='relu', init='he_normal'))

# half of units dropped out

model.add(Dropout(0.5))

# Regular densely-connected layer
# ReLU activation
# initialize weights using he_normal

model.add(Dense(100, activation='relu', init='he_normal'))

# half of units dropped out

model.add(Dropout(0.5))

# Output layer
# outputs are the number of classes in the data set
# Softmax to squash all values between 0 and 1
# initialize weights using he_normal

model.add(Dense(number_classes, activation='softmax', init='he_normal'))

# Compile the model
# uses adamax to optimize loss
# categorical crossentophy becuase of multi-class classification

model.compile(loss='categorical_crossentropy', optimizer='adamax',
              metrics=['accuracy'])


def load_data_channels_first(loc):
    '''load the data using channels_first
    Data must be channels_first or causes a crash.

    Args:
        loc: the location of the data
    Returns:
        tuple containing labels and images, mapping and number of classes.
    '''

    out = []
    for file in os.listdir(loc):
        filename = os.fsdecode(file)
        if not filename.endswith('.txt'):
            with open(loc + '/' + filename, 'rb') as f:
                print(filename)
                (zero, data_type, dims) = struct.unpack('>HBB',
                        f.read(4))
                shape = tuple(struct.unpack('>I', f.read(4))[0]
                              for d in range(dims))
                out.append(np.fromstring(f.read(),
                           dtype=np.uint8).reshape(shape))

    # Rotate images

    (test_img, test_label, train_img, train_label) = out

    for i in range(len(test_img)):
        test_img[i] = rotate(test_img[i])
    for i in range(len(train_img)):
        train_img[i] = rotate(train_img[i])

    print(test_img[0].shape)

    (img_rows, img_cols) = (28, 28)
    depth = 1

    mapping = load_mapping(loc)
    number_classes = len(mapping)

    # retype data to float 32

    train_img = train_img.astype('float32')
    test_img = test_img.astype('float32')

    # Normalize image data

    value_range = 255
    train_img /= value_range
    test_img /= value_range
    print ('shape', train_img.shape)

    # Shape Data

    train_img = train_img.reshape(train_img.shape[0], depth, img_rows,
                                  img_cols)
    test_img = test_img.reshape(test_img.shape[0], depth, img_rows,
                                img_cols)
    input_shape = (img_rows, img_cols, depth)

    if len(mapping) == 26:
        train_label = train_label - 1
        test_label = test_label - 1

    # One hot encoding

    train_label = keras.utils.to_categorical(train_label,
            number_classes)
    test_label = keras.utils.to_categorical(test_label, number_classes)

    return ((test_img, test_label), (train_img, train_label), mapping,
            number_classes)


def layer_to_visualize(layer, name='conv'):
    '''Shows the internal workings of a layer of the neural network.

    Plots the graphic in a graph and displays it on screen.

    Args:
        layer: the layer to be analysed
        name: the name of the layer

    '''

    inputs = [K.learning_phase()] + model.inputs
    _convout1_f = K.function(inputs, [layer.output])

    def convout1_f(X):

        # The [0] is to disable the training phase flag

        return _convout1_f([0] + [X])

    convolutions = convout1_f(img_to_visualize)
    convolutions = np.squeeze(convolutions)
    print ('Shape of {}:'.format(name), convolutions.shape)
    n = convolutions.shape[0]
    n = int(np.ceil(np.sqrt(n)))

    # Visualization of each filter of the layer

    fig = plt.figure(figsize=(12, 8))
    for i in range(len(convolutions)):
        ax = fig.add_subplot(n, n, i + 1)
        ax.imshow(convolutions[i], cmap='gray')
    plt.title = name
    plt.show()


data = load_data('test_train_data/emnist/emnist-letters/')
((test_img, test_label), (train_img, train_label), mapping,
 number_classes) = data
train_img = train_img.reshape(train_img.shape[0], 1, 28, 28)
test_img = test_img.reshape(test_img.shape[0], 1, 28, 28)
img_to_visualize = test_img[5]
img_to_visualize = np.expand_dims(img_to_visualize, axis=0)

# Specify the layer to visualize

layer_to_visualize(convout1, name='Conv 1')
layer_to_visualize(poolout2, name='Pool 2')
layer_to_visualize(convout3, name='Conv 3')
layer_to_visualize(poolout4, name='Pool 4')
