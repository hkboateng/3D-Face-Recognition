# -*- coding: utf-8 -*-
"""
Created on Tue May 12 22:54:29 2020

@author: asus
"""

import cv2 as cv
import numpy as np

capture = cv.VideoCapture(0)

backSub = cv.createBackgroundSubtractorMOG2()

#backSub = cv.createBackgroundSubtractorKNN()
while(True):
    ret, frame = capture.read()
    if frame is None:
        break
    
    fgMask = backSub.apply(frame)
    
    
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
cv.destroyAllWindows()