# -*- coding: utf-8 -*-
"""
Created on Tue May 12 21:17:54 2020

@author: Hubert Kyeremateng-Boateng
"""
import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
green = np.uint8([[[0,255,0 ]]])
hsv_green = cv.cvtColor(green,cv.COLOR_BGR2HSV)
while(True):
    
    #take each frame
    _,frame = cap.read()
    
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([230,255,255])
    green_bound = np.array([10,50,50])
        # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    #filtered = cv.Sobel(frame,cv.CV_64F,1,0,ksize=1)
    edge_detection = cv.Canny(frame,100,200)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= mask)
    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('edge_detection',edge_detection)
    cv.imshow('res',res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()