from train_network import load_mapping, get_session
from test_network import test_custom_data, create_confusion_matrix, plot_confusion_matrix
import keras.backend.tensorflow_backend as ktf
from keras.layers import *
from keras.models import Sequential, save_model, load_model, Model
from keras.utils import *
import numpy as np
import keras.models
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from persistence import *
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import argparse

''' Used to create ensembles from existing networks.
'''

def ensemble_models(models, model_input):
    # collect outputs of models in a list
    yModels=[model(model_input) for model in models]

    # averaging outputs
    yAvg=keras.layers.average(yModels)

    # build model from same input and avg output
    modelEns = Model(inputs=model_input, outputs=yAvg,name='ensemble')

    return modelEns

def create_1():
    ''' Create ensemble_1
    Returns:
        ensemble_model: ensemble_1
    '''
    model_input = Input(shape=(28,28,1))
    models = []
    models.append(load_hdf5('model/emnist-letters_200epoch_lenet_1_1'))
    models.append(load_hdf5('model/emnist-letters_200epoch_lenet_1_2'))
    models.append(load_hdf5('model/emnist-letters_200epoch_lenet_1_3'))
    models.append(load_hdf5('model/emnist-letters_200epoch_lenet_1_4'))
    models.append(load_hdf5('model/emnist-letters_200epoch_lenet_1_5'))
    models.append(load_hdf5('model/emnist-letters_200epoch_lenet_1_6'))
    mapping = create_dict('model/emnist-letters_300epoch_lenet_2_4/mapping.txt')
    ensemble_model = ensemble_models(models,model_input)

    ensemble_model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])

    return ensemble_model

def create_2():
    ''' Create ensemble_2
    Returns:
        ensemble_model: ensemble_2
    '''
    model_input = Input(shape=(28,28,1))
    models = []
    models.append(load_hdf5('model/emnist-letters_300epoch_lenet_2_1'))
    models.append(load_hdf5('model/emnist-letters_300epoch_lenet_2_2'))
    models.append(load_hdf5('model/emnist-letters_300epoch_lenet_2_3'))
    models.append(load_hdf5('model/emnist-letters_300epoch_lenet_2_4'))
    models.append(load_hdf5('model/emnist-letters_300epoch_lenet_2_5'))
    models.append(load_hdf5('model/emnist-letters_300epoch_lenet_2_6'))
    models.append(load_hdf5('model/emnist-letters_300epoch_lenet_2_7'))

    ensemble_model = ensemble_models(models,model_input)
    ensemble_model.compile(loss='categorical_crossentropy',
                           optimizer='adadelta',
                           metrics=['accuracy'])

    return ensemble_model

def create_3():
    ''' Create ensemble_3
    Returns:
        ensemble_model: ensemble_3
    '''
    model_input = Input(shape=(28,28,1))
    models = []
    models.append(load_hdf5('model/emnist-letters_200epoch_lenet_1_1'))
    models.append(load_hdf5('model/emnist-letters_200epoch_lenet_1_2'))
    models.append(load_hdf5('model/emnist-letters_200epoch_lenet_1_5'))
    models.append(load_hdf5('model/emnist-letters_200epoch_lenet_1_6'))
    models.append(load_hdf5('model/emnist-letters_300epoch_lenet_2_2'))
    models.append(load_hdf5('model/emnist-letters_300epoch_lenet_2_1'))
    models.append(load_hdf5('model/emnist-letters_300epoch_lenet_2_4'))
    ensemble_model = ensemble_models(models,model_input)

    ensemble_model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])

    return ensemble_model


def get_session(gpu_fraction=0.333):
    '''Prevents memory errors with tensorflow-gpu.

    Solves a bug whereby tensorflow was using 100% of gpu and crashing.
    '''
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


if __name__ == '__main__':
    ktf.set_session(get_session())
    parser = \
        argparse.ArgumentParser(usage='A training program for creating ensemble classifiers')
    parser.add_argument('-1', '--ensemble_1', action ='store_true',
                        help='Create ensemble 1',
                        default = False)
    parser.add_argument('-2', '--ensemble_2', action = 'store_true',
                        help='Create ensemble 1',
                        default = False)
    parser.add_argument('-3', '--ensemble_3', action = 'store_true',
                        help='Create ensemble 1',
                        default = False)
    parser.add_argument('-p', '--pds', action = 'store_true',
                        help='Test against PDS',
                        default = False)
    args = parser.parse_args()
    data = load_data('test_train_data/emnist/emnist-letters')
    (test_img, test_label), (train_img, train_label), mapping, number_classes = data
    if args.ensemble_1 == True:
        model_1 = create_1()
        if args.pds == True:
            cm = test_custom_data(model_1, 'C:/matlab/test_train_data/pds')
        else:
            cm = create_confusion_matrix(model_1, test_img, test_label)
        plot_confusion_matrix(cm, mapping)
        plt.show()
    if args.ensemble_2 == True:
        model_2 = create_2()
        if args.pds == True:
            cm = test_custom_data(model_2, 'C:/matlab/test_train_data/pds')
        else:
            cm = create_confusion_matrix(model_2, test_img, test_label)
        plot_confusion_matrix(cm, mapping)
        plt.show()
    if args.ensemble_3 == True:
        model_3 = create_3()
        if args.pds == True:
            cm = test_custom_data(model_3, 'C:/matlab/test_train_data/pds')
        else:
            cm = create_confusion_matrix(model_3, test_img, test_label)
        plot_confusion_matrix(cm, mapping)
        plt.show()
