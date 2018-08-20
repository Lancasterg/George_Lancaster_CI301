Project Created by George Lancaster
===============================================================================

Installing Dependencies
-------------------------------------------------------------------------------
To be able to run the files in this project, create a new Python 3.5
environment and install the packages from the requirements.txt using pip.
Instructions for creating a new Python 3.5 environment can be found at the
bottom of this readme file, although there are many tutorials online.

- requirements-cpu.txt, if you have not configured an Nvidia gpu with CUDA;
- requirements-gpu.txt, if you have configured an Nvidia gpu with CUDA.

Use the following command in your newly created, and activated Python 3.5 environment:

- pip install -r requirements-cpu.txt

Project Layout
-------------------------------------------------------------------------------
The testing and training data can be found in the test_train_data folder.
The various splits of the emnist data set can be found in:

- test_train_data/emnist

When using the "-d" modifier in the commands, the string given must be the type
of dataset from within this folder.

For example "letters"

Running the Project
===============================================================================
The files in this project perform two actions.
- training and testing neural networks;
- running the web application.

Web Application
-------------------------------------------------------------------------------
To run the web application, navigate to the project folder using cd in the
command line and enter the following command:

- python app_main.py

The application will start and can be found on port 5000.

There are a variety of images that can be uploaded and tested found in:

- OCR/test_train_data/app_testing_images

Training a Neural Network
-------------------------------------------------------------------------------
To run train_network.py type into command line:

- python train_network.py -d "dataType" -t "type" -s name -e 1

-d: the type of data to test on. String, must be one of:
- balanced
- byclass
- bymerge
- digits
- letters

-r: apply rotate transformations

-s: apply shift transformations

-z: apply zoom transformations

For example:
- python train_network.py

- python train_network.py -d "letters" -t "lenet" -e "200" -r -z

Creating an Ensemble Classifier
-------------------------------------------------------------------------------
To run ensemble.py type into command line:

- python ensemble.py -1 -2 -3 -p

-1: create ensemble_1

-2: create ensemble_2

-3: create ensemble_3

-p: test on the PDS data set

examples:

- python ensemble.py -1

- python ensemble.py -2 -p


Testing a Neural Network
-------------------------------------------------------------------------------
To run test_network.py type into command line:

- python test_network.py -m "model name" -d "data type"
or
- python test_network.py -h

-d: the type of data to test on. String, must be one of:
- balanced
- byclass
- bymerge
- digits
- letters

-m: the name of the model. All models can be found in the "model" folder
    in the project directory. Give this as a string.

-n: normalise the data in the confusion matrix

examples:
- python test_network.py -m "emnist-digits_200epoch_lenet_rotate_zoom" -d "digits"

- python test_network.py -m "emnist-letters_300epoch_lenet_2_4" -d "letters"


Additional Help
===============================================================================
The application has been tested on both Windows and Unix systems, so any
problems will be due to an incorrectly configured Python 3.5 environment, or
errors when installing the requirements-xxx.txt.

Testing and Training Neural Networks
-------------------------------------------------------------------------------
All arguments must be written as is. Any spelling mistakes will prevent the
program from running.


Create a Python 3.5 Environment
-------------------------------------------------------------------------------
To create and activate a new Python 3.5 environment:
1. install anaconda
2. use the command:
  2.1 conda create -n ktfo python=3.5
3. activate the new environment
  3.1 Windows: activate ktfo
  3.2 Unix: source activate ktfo

Additional Help
------------------------------------------------------------------------------
For additional help, please don't hesitate to contact me on:
- Lancaster0180@gmail.com
- gl162@brighton.ac.uk
- 07581781417

A version of the web application is hosted on:

http://georgelancaster.pythonanywhere.com/
This version is slow, as it is being hosted cheaply.
