#!/usr/bin/python
# -*- coding: utf-8 -*-
from persistence import *
from char_extractor import *
import argparse
import numpy as np

''' This file was used for the creation of the PDS dataset

'''


def generate_data(img, letter):
    '''Detect features from a photo and save them to a file.
    '''

    lines = get_classifiable(img)
    count = 0
    for line in lines:
        for char in line:
            char = np.invert(char)
            cv2.imwrite('c:/matlab/test_data/{}/{}.jpg'.format(letter,
                        str(count)), char)
            count += 1
    print(count)


def invert():
    '''Loop through all images in directories and invert their pixels
    '''

    loc = 'c:/matlab/letters/'
    directory = os.fsencode(loc)
    for (subdir, dirs, files) in os.walk(loc):
        for file in files:
            filename = os.path.join(subdir, file)
            im = cv2.imread(filename, 0)
            im = np.invert(im)
            cv2.imwrite(filename, im)


def resizeimgs(loc='c:/matlab/temp/'):
    '''Resize all images in directory
    '''

    directory = os.fsencode(loc)
    for (subdir, dirs, files) in os.walk(loc):
        for file in files:
            filename = os.path.join(subdir, file)
            im = cv2.imread(filename, 0)
            preshape = im.shape
            im = cv2.resize(im, (0, 0), fx=0.25, fy=0.25)
            cv2.imwrite(filename, im)
            print(filename + ' was resized from: ' + str(preshape) \
                + ' to: ' + str(im.shape))


def move_img_change_name(old='c:/matlab/letters/',
                         new='c:/matlab/test_data'):
    '''Move all images from one dir to another, change their names to not overwrite
    images already in the file.
    '''

    all_img = []
    for (subdir, dirs, files) in os.walk(old):
        letter_list = []
        for file in files:
            filename = os.path.join(subdir, file)
            img = cv2.imread(filename, 0)
            letter_list.append(img)
        all_img.append(letter_list)
    count = 0
    for (subdir, dirs, files) in os.walk(new):
        for img in all_img[count]:
            newname = str(len(os.listdir(subdir)) + 1)
            filename = os.path.join(subdir, newname + '.jpg')
            cv2.imwrite(filename, img)
            print('image saved to: ' + filename)
        count += 1


if __name__ == '__main__':
    parser = \
        argparse.ArgumentParser(usage='A training program for classifying the EMNIST dataset'
                                )
    parser.add_argument('-f', '--file', type=str, help='Path to image')

    parser.add_argument('-t', '--type', type=str, help='Type of letter')

    parser.add_argument('-m', '--model', type=str, help='Path to model')
    args = parser.parse_args()
    location = args.model
    print(location)
