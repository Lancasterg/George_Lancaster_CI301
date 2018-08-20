#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
from os import listdir

'''This file is the current character extraction algorithm.
    It detects features from an image, sorts them into read order then
    converts them to a classifiable format. 

'''


def get_classifiable(img, mode='use'):
    ''' This method takes an image as an input, and attempts to extract
    any significaant features. In order for it to work effectively, the picture
    must have a white background. For example, a photograph of a piece of paper.

    Example photographs can be found in the xyz folder.

    Args:
        img: the image to be processed
        mode: use or test - for testing purposes

    Result:
        lines: A list of features sorted into lines

    '''

    img = cv2.bitwise_not(img)
    org = img

    (ret, thresh) = cv2.threshold(img, 150, 255, 0)
    (rows, cols) = img.shape

    # rotate 5 degrees

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 355, 1)
    img = cv2.warpAffine(thresh, M, (cols, rows),
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(0, 0, 0))

    # dilate and find features

    kernel = np.ones((2, 2), np.uint8)
    dilation = cv2.dilate(thresh, kernel, iterations=1)
    (cnts, _) = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)[-2:]
    contours = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w * h > 200 and w < 2 * h:
            contours.append((x, y, w, h))
            boundingbox = cv2.rectangle(dilation.copy(), (x, y), (x
                    + w, y + h), (143, 143, 143), 1)

    # sort all rect by their y

    if len(contours) == 0:
        return contours
    contours.sort(key=lambda b: b[1])

    # initially the line bottom is set to be the bottom of the first rect

    line_bottom = contours[0][1] + contours[0][3] - 1
    line_begin_idx = 0
    for i in range(len(contours)):

        # when a new box's top is below current line's bottom
        # it's a new line

        if contours[i][1] > line_bottom:

            # sort the previous line by their x

            contours[line_begin_idx:i] = \
                sorted(contours[line_begin_idx:i], key=lambda b: b[0])
            line_begin_idx = i

        # regardless if it's a new line or not
        # always update the line bottom

        line_bottom = max(contours[i][1] + contours[i][3] - 1,
                          line_bottom)

    # sort the last line

    contours[line_begin_idx:] = sorted(contours[line_begin_idx:],
            key=lambda b: b[0])
    if mode == 'use':
        lines = []
        line = []
        line_bottom = contours[0][1] + contours[0][3] - 1
        print('{} features detected'.format(len(contours)))
        for i in range(len(contours)):
            (x, y, w, h) = contours[i]

            # new line

            if contours[i][1] > line_bottom:
                lines.append(line)
                line = []
            line_bottom = max(contours[i][1] + contours[i][3] - 1,
                              line_bottom)
            single = np.asarray(org[y:y + h, x:x + w])
            (rows, col) = single.shape

            # rotate back 5 degrees

            M = cv2.getRotationMatrix2D((col / 2, rows / 2), 365, 1)
            rot = cv2.warpAffine(single, M, (col, rows),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))

            # invert pixels

            single = cv2.bitwise_not(rot)
            (ret, single) = cv2.threshold(single, 120, 255, 0)
            blank_image = np.full((28, 28), 255, np.uint8)
            single = resize_to_scale(single)

            # superpose onto background image

            x_offset = int((28 - single.shape[1]) * 0.5)
            y_offset = int((28 - single.shape[0]) * 0.5)

            blank_image[y_offset:y_offset + single.shape[0], x_offset:
                        x_offset + single.shape[1]] = single
            line.append(blank_image)
        lines.append(line)
        return lines
    elif mode == 'test':

                            # return image showing order of contours

        theImg = org
        theImg = cv2.cvtColor(theImg, cv2.COLOR_GRAY2RGB)
        res = []
        print('x,y,w,h')
        for n in range(len(contours)):
            (x, y, w, h) = contours[n]
            print (x, y, w, h)
            cv2.rectangle(theImg, (x, y), (x + w, y + h), (0, 255, 0),
                          2)
            theImg = cv2.putText(theImg,
                                str(n),(x, y),
                                cv2.FONT_HERSHEY_PLAIN,
                                2,(0, 255, 0),)
        return theImg
    return lines


def resize_to_scale(image):
    '''Resize an image based on its smallest side. This is done to make the
    fit in a 28x28 classifiable image.

    Args:
        image: the image to be resized
    Returns:
        image: the resized image
    '''

    MAX_SIZE = 28
    original_size = max(image.shape[0], image.shape[1])
    if image.shape[0] > image.shape[1]:
        resized_width = MAX_SIZE
        resized_height = int(round(MAX_SIZE / float(image.shape[0])
                             * image.shape[1]))
    else:
        resized_height = MAX_SIZE
        resized_width = int(round(MAX_SIZE / float(image.shape[1])
                            * image.shape[0]))
    print (resized_height, resized_width)
    image = cv2.resize(image, (resized_height, resized_width))
    image = cv2.resize(image, (0, 0), fx=0.7, fy=0.7)
    return image
