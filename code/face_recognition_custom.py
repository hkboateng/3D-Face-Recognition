#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 23:03:38 2020

@author: hkyeremateng-boateng
"""

import os
import cv2 as cv
import numpy as np
encodings = []
names = []
vector_row = np.array([3, 2, 3])
image = cv.imread("dataset/kofi_annan/kofi_annan_1.jpeg")

image_blob = cv.dnn.blobFromImage(image)
cascade = cv.CascadeClassifier()
det = cascade.detectMultiScale(image,vector_row)