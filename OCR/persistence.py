#!/usr/bin/python
# -*- coding: utf-8 -*-
from keras.models import model_from_json, model_from_yaml, save_model
from scipy.misc import imread, imresize, imshow
import matplotlib.pyplot as plt
import struct
import numpy as np
import keras.models
import tensorflow as tf
import os
import pickle

''' Used to save/load models and data.
'''

def load_mapping(loc):
    ''' load mapping from location loc '''

    mapping = []
    for file in os.listdir(loc):
        filename = os.fsdecode(file)
        if filename.endswith('.txt'):
            with open(loc + '/' + filename) as f:
                for line in f:
                    mapping.append(chr(int(line.split()[1])))
    return mapping


def rotate(img):
    '''Rotate an image 90 degrees
    Some datasets are read in oriented the wrong way.
    '''

    flipped = np.fliplr(img)
    return np.rot90(flipped)


def create_dict(loc):
    '''Create a dictionary of mappings'''

    d = {}
    with open(loc) as f:
        for line in f:
            (key, val) = line.split()
            d[int(key)] = int(val)
    return d


def load_data(loc):
    '''load the mapping, and training and testing data sets.
    Args:
        loc: the location of the data
    Returns:
        tuple containing labels and images, mapping and number of classes.

    '''

    out = []
    file_list = os.listdir(loc)
    file_list.sort()  # sort alphabetical order for UNIX fs
    for file in file_list:
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

    # Shape Data

    train_img = train_img.reshape(train_img.shape[0], img_rows,
                                  img_cols, depth)
    test_img = test_img.reshape(test_img.shape[0], img_rows, img_cols,
                                depth)
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


def save_json(model, name):
    '''save model as a JSON'''

    model_json = model.to_json()
    with open('model/{}/model.json'.format(name), 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('model/{}/model.h5'.format(name))
    print('Saved model to disk as model.json')


def save_yaml(model, name):
    '''save model as YAML'''

    model_yaml = model.to_yaml()
    with open('model/{}/model.yaml'.format(name), 'w') as yaml_file:
        yaml_file.write(model_yaml)
    save_model(model, 'model/{}/model.h5'.format(name))
    print('Saved model to disk as model.yaml')


def save_history(history, name, loc):
    ''' save the classification history as Pickle '''

    with open('model/{}/{}.pickle'.format(name, loc), 'wb') as f:
        pickle.dump(history.history, f)


def load_history(location):
    ''' Load history from a Pickle file'''

    return pickle.load(open(location, 'rb'))


def load_json(location):
    ''' load a model from JSON file

    Args:
        location: the location of the model
    Returns:
        the loaded model and graph
    '''

    json_str = location
    h5_str = location
    h5_str += '.h5'
    json_str += '.json'
    json_file = open(json_str, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model

    loaded_model.load_weights(h5_str)
    print('Loaded Model from disk')

    # compile and evaluate loaded model

    loaded_model.compile(loss='categorical_crossentropy',
                         optimizer='adam', metrics=['accuracy'])
    graph = tf.get_default_graph()
    return (loaded_model, graph)


def load_hdf5(location):
    '''load model from hdf5

        Args:
            location: the location of the model
        Returns:
            model: the loaded model
        '''

    yaml_file = open('%s/model.yaml' % location, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)
    model.load_weights('%s/weights.best.hdf5' % location)
    return model


def load_yaml(location):
    '''load model from yaml
        Args:
            location: the location of the model
        Returns:
            model: the loaded model
    '''

    yaml_file = open('%s/model.yaml' % location, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)
    model.load_weights('%s/model.h5' % location)
    return model


def plot_model_p(history):
    '''plot graphs using Pickle file
    Args:
        history: the data to plot
    '''

    (f, (ax1, ax2)) = plt.subplots(1, 2)  # , sharey=True)
    ax1.plot(history['acc'], 'r-')
    ax1.plot(history['val_acc'])
    ax1.set_title('model accuracy')
    ax1.set_ylabel('error %')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'test'], loc='lower right')
    ax2.plot(history['loss'], 'r-')
    ax2.plot(history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'test'], loc='upper right')

    # f.set_figwidth(30)
    # ax2.set_figwidth(30)

    plt.rcParams['figure.figsize'] = [16, 9]
    f.set_size_inches((12, 5))
    plt.show()


    # plt.figure(figsize=(20,20))

def plot_model_d(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
