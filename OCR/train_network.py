#!/usr/bin/python
# -*- coding: utf-8 -*-
import keras
from keras.layers import Conv2D, MaxPooling2D, Convolution2D, Dropout, \
    Dense, Flatten, LSTM
from keras.preprocessing.image import ImageDataGenerator, array_to_img, \
    img_to_array, load_img
from keras.models import Sequential, save_model
import keras.backend.tensorflow_backend as ktf
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from persistence import *
import numpy as np
import argparse
import struct
import sys
import os
import cv2
''' This file is for creating and training neural networks.
Instuctions on how to run it can be found in the readme file, or by typing

"python train_network.py -h" into the command line.

'''


def get_session(gpu_fraction=0.333):
    ''' Prevents memory errors with tensorflow.
    '''

    gpu_options = \
        tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                      allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def create_lenet5(training_data, width=28, height=28):
    '''Create a lenet-5 inspired network
    Creates a neural network which uses two convolutional layers and two
    pooling layers. This is the architecture for the netowork used in
    experiments carried out in the project.
    Args:
        training_data: the training data, used to tailor the network to
        the data.
        width: 28 pixels
        height: 28 pixels

    Returns:
        model: a sequential model of a neural network
    '''

    ((test_img, test_label), (train_img, train_label), mapping,
     number_classes) = training_data
    input_shape = (height, width, 1)
    number_filters = 32  # number of convolutional filters to use
    pool_size = (2, 2)  # size of pooling area for max pooling

    # kernel_size = (3, 3) # convolution kernel size

    model = Sequential()

    # Convolutional input Layer.
    # Takes an input vector of size (28,28,1) so 784 inputs

    model.add(Convolution2D(12, kernel_size=(5, 5), activation='relu',
              input_shape=input_shape, init='he_normal'))

    # Max pooling layer using pool_size (2,2)

    model.add(MaxPooling2D(pool_size=pool_size))

    # Convolutional layer
    # 25 filters with kernel size of (5,5)

    model.add(Convolution2D(25, kernel_size=(5, 5), activation='relu',
              init='he_normal'))

    # Max pooling layer using pool_size (2,2)

    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten to 1d tensor

    model.add(Flatten())
    model.add(Dense(180, activation='relu', init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu', init='he_normal'))
    model.add(Dropout(0.5))

    # Output layer
    # outputs are the number of classes in the data set
    # Softmax to squash all values between 0 and 1
    # initialize weights using he_normal

    model.add(Dense(number_classes, activation='softmax',
              init='he_normal'))
    model.compile(loss='categorical_crossentropy', optimizer='adamax',
                  metrics=['accuracy'])
    return model


def create_recurrent(training_data, width=28, height=28):
    '''Creates a recurrent neural network
    Accuracy is much lower than lenet as it does not use stacked Convolutional
    and pooling layers.
    I have kept it to show that other networks were explored.

    Args:
        training_data: the training data, used to tailor the network to
        the data.
        width: 28 pixels
        height: 28 pixels
    Returns:
        model: a sequential model of a neural network
    '''

    ((test_img, test_label), (train_img, train_label), mapping,
     number_classes) = training_data
    input_shape = (height, width, 1)
    number_filters = 32
    pool_size = (2, 2)
    kernel_size = (3, 3)
    model = Sequential()
    model.add(Convolution2D(number_filters, kernel_size, padding='valid'
              , input_shape=input_shape, activation='relu'))
    model.add(Convolution2D(number_filters, kernel_size,
              activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(number_classes, activation='softmax'))  # Number of outputs = 512

    model.compile(loss='categorical_crossentropy', optimizer='adadelta'
                  , metrics=['accuracy'])
    return model




def train_model(model, training_data, epochs, name, transform=False, rotate=False,
                shift=False,zoom=False,callback=False,batch_size=256,save_name=''):
    '''This method trains a model using the training data and parameters
    specified.

    Args:
        model: the model to be trained
        training_data: the data to train the model on
        epochs: number of epochs for training
        transform: wether to apply trnasofrmations or not
        callback: use tensorboard callbacks?
        batch_size: size of training batches

    '''

    # unpack training data

    ((test_img, test_label), (train_img, train_label), mapping,
     number_classes) = training_data
    if callback == True:

        # TensorBoard callback run: tensorboard --logdir path_to_current_dir/Graph

        tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph',
                histogram_freq=0, write_graph=True, write_images=True)

    # checkpoint

    filepath = save_name + 'weights.best.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc',
                                 verbose=1, save_best_only=True,
                                 mode='max')
    callbacks_list = [checkpoint]

    # Fit the model

    if transform == False:
        history = model.fit(
            train_img,
            train_label,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(test_img, test_label),
            callbacks=callbacks_list,
            )
        save_history(history, name, 'history')

    if transform == True:

        # Train on transformed data

        datagen = \
            ImageDataGenerator(rotation_range=(15 if rotate else 0),
                               width_shift_range=(0.2 if shift else 0),
                               height_shift_range=(0.2 if shift else 0),
                               zoom_range=(0.3 if zoom else 0))
        datagen.fit(train_img)
        tf_history = model.fit_generator(
            datagen.flow(train_img, train_label,
                         batch_size=batch_size),
                         steps_per_epoch=len(train_img) / batch_size,
                         epochs=epochs,
                         verbose=1,
                         validation_data=(test_img, test_label),
                         callbacks=callbacks_list)

        save_history(tf_history, name, 'history')

    score = model.evaluate(test_img, test_label, verbose=0)
    print ('Test score:', score[0])
    print ('Test accuracy:', score[1])
    save_yaml(model, name)


def move_graph(loc):
    '''For tensorboard
    Moves the graph from the project Graph folder to the correct folder

    Args:
        loc: the new location
    '''

    old_loc = os.path.dirname(os.path.realpath(__file__)) + '/Graph/'
    new_loc = loc + '/graph/'
    os.makedirs(new_loc)
    for file in os.listdir(old_loc):
        filename = os.fsdecode(file)
        os.rename(old_loc + filename, new_loc + filename)


def save_mapping(mapping, save_loc):
    '''Save the mapping to the location

    Args:
        mapping: the mapping file
        save_loc: the location to be saved
    '''

    f = open(save_loc + 'mapping.txt', 'w')
    f.write(str(input(mapping)))
    f.close()


if __name__ == '__main__':
    ktf.set_session(get_session())  # Prevents errors from using too much gpu memory
    parser = \
        argparse.ArgumentParser(usage='A training program for classifying the EMNIST dataset')
    parser.add_argument('-d', '--data', type=str,
                        help='Name of training data: balanced, byclass, bymerge, digits, letters'
                        , default='letters')
    parser.add_argument('-t', '--type', type=str,
                        help='Type of network to train: lenet or rec',
                        default='lenet')
    parser.add_argument('-e', '--epochs', type=int, default=12,
                        help='Number of epochs to train on')
    parser.add_argument('-r', '--rotate', action='store_true',
                        default=False,
                        help='Apply rotate transformations to training data')
    parser.add_argument('-s', '--shift', action='store_true',
                        default=False,
                        help='Apply shift transformations to training data')
    parser.add_argument('-z', '--zoom', action='store_true',
                        default=False,
                        help='Apply zoom transformations to training data')
    args = parser.parse_args()

    data_loc = 'test_train_data/emnist/emnist-{}'.format(args.data)

    save_name = data_loc.split('/').pop() + '_' + str(args.epochs) \
        + 'epoch_' + args.type

    transform = False
    if args.rotate == True:
        save_name += '_rotate'
        transform = True
    if args.shift == True:
        save_name += '_shift'
        transform = True
    if args.zoom == True:
        save_name += '_zoom'
        transform = True
    out_dir = 'model/' + save_name

    if os.path.exists(out_dir):
        out_dir += '_new'
        save_name += '_new'
        os.makedirs(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data = load_data(data_loc)

    if args.type.lower() == 'rec':
        model = create_recurrent(data)
    if args.type.lower() == 'lenet':
        model = create_lenet5(data)

    save_loc = 'model/{}/'.format(save_name)
    train_model(model,
                data,
                args.epochs,
                save_name,
                transform=transform,
                rotate=args.rotate,
                shift=args.shift,
                zoom=args.zoom,
                save_name=save_loc)
