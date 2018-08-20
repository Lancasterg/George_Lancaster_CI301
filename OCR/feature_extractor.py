#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
from os import listdir

''' A deprecated implementation to detect features from an image.
    The newest implementation can be found in char_extractor.py
'''

def multi_line_ext(ipt):
    ''' Split a paragraph of test into lines.
    Args:
        ipt: the input image.
    '''

    ipt = cv2.bitwise_not(ipt)
    (rows, cols) = ipt.shape
    (ret, thresh) = cv2.threshold(ipt, 127, 255, 0)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 355, 1)
    img = cv2.warpAffine(thresh, M, (cols, rows),
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(0, 0, 0))
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=2)  # Dilate to increase area
    (ret, thresh) = cv2.threshold(dilation, 127, 255, 0)
    (im2, cnts, hierarchy) = cv2.findContours(thresh, 1, 2)  # Detect Objects
    arr = []
    order = []
    for c in cnts:  # For each object detected
        epsilon = 0.1 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        (x, y, w, h) = cv2.boundingRect(c)  # Use boudning box to capture detected
        boundingbox = cv2.rectangle(img.copy(), (x, y), (x + w, y + h),
                                    (0, 255, 0), 1)
        bar = img[y:y + h, x:x + w]
        order.append(y)  # Append to order (top down)
        arr.append(bar)  # Append detected to array
    if len(arr) > 1:
        arr = read_order_height(arr, order)  # Order array from top to bottom
    return arr


def window_ext(img):
    '''Split a line into letters.
    Args:
        img: the input image.
    '''

    edged = cv2.Canny(img, 10, 250)
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)[-2:]
    arr = []
    order = []  # for the left/right order
    total = 0

    if len(cnts) > 0:

        # loop through to find average size

        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            total += w * h
        size_thresh = total / len(cnts)

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            (x, y, w, h) = cv2.boundingRect(c)
            (ret, thresh) = cv2.threshold(img, 127, 255,
                    cv2.THRESH_BINARY)
            if w * h > 200:  # filter out missclassifications
                pre_crop = np.asarray(thresh[y:y + h, x:x + w])
                (rows, cols) = pre_crop.shape
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 365,
                        1)
                crop = cv2.warpAffine(pre_crop, M, (cols, rows),
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0,
                        0, 0))

                # Resizing and stitching

                blank_image = np.full((28, 28), 0, np.uint8)
                crop = cv2.resize(crop, (28, 28))
                crop = cv2.resize(crop, (0, 0), fx=0.6, fy=0.6)
                x_offset = y_offset = 7
                blank_image[y_offset:y_offset + crop.shape[0], x_offset:
                            x_offset + crop.shape[1]] = crop

                # Resizing

                ipt = cv2.bitwise_not(blank_image)
                arr.append(ipt)
                order.append(x)
        if len(arr) > 1:
            arr = read_order_width(arr, order)
    return arr


def read_order_width(arr, order):
    ''' Sort the order to be read left to right
    Args:
        arr: the array to be sorted
        order: the order to sort
    '''

    (list1, list2) = [list(x) for x in zip(*sorted(zip(order, arr),
                      key=lambda pair: pair[0]))]
    return list2


def read_order_height(arr, order):
    ''' Sort the order to be read top to bottom
    Args:
        arr: the array to be sorted
        order: the order to sort
    '''

    (list1, list2) = [list(x) for x in zip(*sorted(zip(order, arr),
                      key=lambda pair: pair[0]))]
    return list2
