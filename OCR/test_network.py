#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import os
import struct
import numpy as np
from persistence import *
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import argparse
import keras.backend.tensorflow_backend as ktf
from keras.preprocessing.image import ImageDataGenerator
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

''' This file is for creating and testing neural networks.
Instuctions on how to run it can be found in the readme file, or by typing

"python test_network.py -h" into the command line.

'''


def get_session(gpu_fraction=0.333):
    ''' Prevents memory errors with TensorFlow
    '''
    gpu_options = \
        tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                      allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def show_random_mnist(loc):
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

    (test_img, test_label, train_img, train_label) = out
    for i in range(0, 8):
        show(test_img[i], i)


def show(img, num):
    flipped = np.fliplr(img)
    cv2.imwrite('{}.png'.format(num), np.rot90(flipped))


def create_confusion_matrix(model, test_img, test_label):
    ''' Create a confusion confusion matrix.
        Tests a network model against test data to generate a
        confusion matrix.
        Args:
            model: the neural network model
            test_img: the test images
            test_label: the test labels for the images
        '''

    trim = test_img
    history = model.predict(trim)
    model.compile(loss='categorical_crossentropy', optimizer='adamax',
                  metrics=['accuracy'])
    test = model.evaluate(test_img, test_label, verbose=1)
    print (test[0], 100 - test[1] * 100)
    confusion = np.zeros((len(test_label[0]), len(test_label[0])))
    for n in range(len(history)):
        enum_history = history[n]
        result = np.where(enum_history == max(enum_history))[0][0]
        actual = np.where(test_label[n] == max(test_label[n]))[0][0]
        confusion[actual][result] += 1
    confusion = confusion.astype(int)
    return confusion


def plot_confusion_matrix(cm, classes,normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.RdYlGn):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Args:
        cm: the confusion matrix
        classes: the mapping for the predictions
        normalize: normalise the confusion matrix values
        title: title of the plot
        cmap: colours of the matrix
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        cm = cm.astype('int')
        print ('Normalized confusion matrix')
    else:
        print ('Confusion matrix, without normalization')
    print (cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    miss_list = [0] * len(classes)
    for (i, j) in itertools.product(range(cm.shape[0]),
                                    range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        if i != j:
            miss_list[i] += cm[i][j]
    print ('Incorrect classifications: ')
    print (sum(miss_list))

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def norm(img):
    '''Normalise data for testing custom data

    Args:
        img: the image to normalize
    Returns:
        img: the normalised image
    '''

    img = np.float32(img)
    img = cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
    return img


def test_custom_data(model, location):
    ''' Create a confusion matrix for custom data
    In this case it can be used for PDS
    Args:
        model: the model to be tested
        location: the location of the training data
    Returns:
        confusion: the confusion matrix for the predictions
    '''

    datagen = ImageDataGenerator(preprocessing_function=norm,
                                 data_format = 'channels_last')
    test_generator = datagen.flow_from_directory(location,
                                                target_size=(28, 28),
                                                batch_size=32,
                                                color_mode='grayscale',
                                                shuffle=False)

    model.compile(loss='categorical_crossentropy', optimizer='adamax',
                  metrics=['accuracy'])
    eval_history = model.evaluate_generator(test_generator)
    pred_history = model.predict_generator(test_generator)
    test_label = create_one_hots(test_generator.classes)
    count = 1

    for x in range(len(pred_history)):
        if get_largest(pred_history[x]) \
            == test_label[x].index(max(test_label[x])):
            count += 1
    print (count)
    print (model.metrics_names[0], eval_history[0])
    print (model.metrics_names[1], (1 - eval_history[1]) * 100)
    confusion = np.zeros((len(test_label[0]), len(test_label[0])))
    for n in range(len(pred_history)):
        enum_history = pred_history[n]
        result = np.where(enum_history == max(enum_history))[0][0]
        actual = test_label[n].index(max(test_label[n]))
        confusion[actual][result] += 1
    confusion = confusion.astype(int)
    return confusion


def create_one_hots(classes):
    res = []
    for c in classes:
        n = [0] * 26
        n[c] = 1
        res.append(n)
    return res


def get_largest(ipt):
    return np.where(ipt == max(ipt))[0][0]


if __name__ == '__main__':
    ktf.set_session(get_session())  # Prevents errors from using too much gpu memory
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = \
        argparse.ArgumentParser(usage='A testing program for classifying the EMNIST dataset')
    parser.add_argument('-m', '--model', type=str,
                        help='Name of folder containing model. \n'
                        + ' found in project folder model/',
                        default='emnist-letters_300epoch_lenet_2_4')

    parser.add_argument('-d', '--data', type=str,
                        help='Type of data to test on: balanced, byclass, bymerge, digits, letters, mnist',
                        default='letters')
    parser.add_argument('-p', '--pds', action='store_true',
                        default=False, help='test on the PDS data set')
    parser.add_argument('-n', '--norm', action='store_true',
                        default=False,
                        help='normalise the confusion matrix')

    args = parser.parse_args()
    normal = args.norm
    data_str = 'test_train_data/emnist/emnist-' + args.data
    model_str = 'model/' + args.model
    model = load_hdf5(model_str)
    data = load_data(data_str)
    ((test_img, test_label), (train_img, train_label), mapping, number_classes) = data

    if args.pds == False:
        conf = create_confusion_matrix(model, test_img, test_label)
        plt.figure(figsize=(15, 10))
        plot_confusion_matrix(conf, mapping, normalize=normal)
        plt.savefig('plot.png', dpi=100)
        history = load_history(model_str + '/history.pickle')
        plot_model_p(history)

    if args.pds == True:
        location = 'test_train_data/pds'
        cm = test_custom_data(model, location)
        plt.figure(figsize=(10, 5))
        plot_confusion_matrix(cm, mapping, normalize=normal)
        plt.show()
