#!/usr/bin/python
# -*- coding: utf-8 -*-
from scipy.misc import imsave, imread, imresize
from train_network import load_mapping, get_session
import keras.backend.tensorflow_backend as ktf
import numpy as np
import keras.models
from keras.models import load_model
import pickle
import uuid
import re
import sys
import os
import cv2
from char_extractor import get_classifiable
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from os import listdir
from persistence import *
from feature_extractor import *

'''This module is the main back-end, which handles classification and
    character extraction.
'''


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ktf.set_session(get_session())
model = load_yaml('model/emnist-digits_200epoch_lenet_rotate_zoom')
mapping = \
    pickle.load(open('model/emnist-digits_200epoch_lenet_rotate_zoom/mapping.p'
                , 'rb'))
doc_analysis = 1    # select the type of feature extraction algorithm


def set_network(network_type):
    '''Sets the current network to the type specifed

    Args:
        networkType: The type of classificaiton to be done
    '''

    network_type = network_type.replace("'", '')
    network_type = network_type.strip('b')
    ktf.set_session(get_session())
    global model
    global mapping
    if network_type == 'numbers':
        model = \
            load_hdf5('model/emnist-digits_200epoch_lenet_rotate_zoom')
        mapping = \
            pickle.load(open('model/emnist-digits_200epoch_lenet_rotate_zoom/mapping.p', 'rb'))
    if network_type == 'letters':
        model = load_model('model/ensemble_3_3/ensemble')
        mapping = \
            create_dict('model/emnist-letters_300epoch_lenet_2_4/mapping.txt')
    if network_type == 'class':
        model = \
            load_hdf5('model/emnist-byclass_300epoch_lenet_rotate_zoom')
        mapping = \
            pickle.load(open('model/20_epoch_byClass_recurrent/mapping.p', 'rb'))

def norm(img):
    '''Normalise the pixel values in an
    image to values between 0 and 1.

    Args:
        img : the image to be normalised
    Returns:
        img: normalised image
    '''

    img = np.float32(img)
    img = cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
    return img


def test_network_arr(img):
    ''' Classify a single character
    1.Invert the image to have black background and white foreground.
    2.Reshape to fit network input layer.
    3.Make a prediciton

    Args:
        img: The image to be classified
    Returns:
        out: The prediciton
    '''

    img = np.invert(img)
    img = imresize(img, (28, 28))  # resize image
    img = img.reshape(1, 28, 28, 1)  # shape to fit network input
    img = norm(img)
    out = model.predict(img)
    return out



def create_dict(loc):
    '''Create a dictionary for one-hot encoding

    Args:
        loc: the location of the mapping file
    '''
    d = {}
    with open(loc) as f:
        for line in f:
            (key, val) = line.split()
            d[int(key)] = int(val)
    return d


def get_result(path):
    ''' Classify all features from within an image

    This method classifies an image. It is split into three parts:
        1. detect features using the methods in char_extractor.py
        2. classify the detected features using the current neural netowrk
        3. format the output
    Args:
        path: the location of the image to be classified
    Returns:
        result: the classified string
    '''

    img = cv2.imread(path, 0)
    (x, y) = img.shape
    area = x * y
    if doc_analysis == 0:  # first deprecated segmentation algorithm
        lines = multi_line_ext(img)
        for line in lines:
            output = ''
            arr = window_ext(line)
            if line != []:
                for a in arr:
                    out = test_network_arr(a)
                    res = chr(mapping[int(np.argmax(out, axis=1)[0])])
                    output += res
    else:                               # second segmentation algorithm
        lines = get_classifiable(img)   # detect characters
        if lines == []:                 # if no chars detected
            return 'Error: No characters detected.'
        output = ''
        for line in lines:
            for char in line:
                out = test_network_arr(char) # classify a single char
                res = chr(mapping[int(np.argmax(out, axis=1)[0])])  # one-hot
                output += res                # add to output string
            output += '\n'
    if output == '':
        output = 'Classification error'
    return output


def format_result(arr):
    '''Formats the prediciton result to a readable string

    Args:
        arr:a list of predicted chars sorted into lines

    Returns:
        string: the formatted string
    '''
    string = ''
    for a in arr:
        string += a
        string += '\n'
    string = string.strip()
    return string


def save_thumbnail(user, img):
    ''' Reduce the size of an image and generate a thumbnail.
    The image is saved to the users file. It is used in the hub page to
    display previous predictions.

    Args:
        user: the current user
        img: the image to be resized

    Returns:
        the location of the image
    '''

    max_height = 200
    if img.shape[0] < img.shape[1]:
        img = np.rot90(img)
    hpercent = max_height / float(img.shape[0])
    wsize = int(float(img.shape[1]) * float(hpercent))
    img = cv2.resize(img, (wsize, max_height))
    newname = uuid.uuid4()
    cv2.imwrite('static/users/{}/{}.png'.format(user, newname), img)
    return 'static/users/{}/{}.png'.format(user, newname)


def get_session(gpu_fraction=0.333):
    '''Prevents memory errors with tensorflow-gpu.

    Solves a bug whereby tensorflow was using 100% of gpu and crashing.
    '''

    gpu_options = \
        tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                      allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
