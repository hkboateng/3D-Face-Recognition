#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 21:17:50 2020

@author: hkyeremateng-boateng
"""

import face_recognition
import cv2 as cv
from PIL import Image
from sklearn import svm
import os

encodings = []
names = []
# Training directory
train_dir = os.listdir('dataset/')
train_dir.remove('All images')
for person in train_dir:
    pix = os.listdir("dataset/" + person)
    
    for pic in pix:
        img_path = "dataset/" + person + "/" +pic
        image = face_recognition.load_image_file(img_path)
        face_bounding_boxes = face_recognition.face_locations(image,model="cnn")
        #If training image contains exactly one face
        if len(face_bounding_boxes) == 1:

            face_enc = face_recognition.face_encodings(image)
            # Add face encoding for current image with corresponding label (name) to the training data
            encodings.append(face_enc)
            names.append(person)
        else:
            print(person + "/" + pic + " was skipped and can't be used for training")        
    pass

#model = svm.SVC(gamma='scale')
image = face_recognition.load_image_file("children_4.jpeg")
face_locations = face_recognition.face_locations(image,model="cnn")
print("I found {} face(s) in this photograph.".format(len(face_locations)))
face_landmarks = face_recognition.face_landmarks(image)
#face_enc = face_recognition.face_encodings(image)
for face_location in face_locations:
    
    # Print the location of each face in this image
    top, right, bottom, left = face_location
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))


    # You can access the actual face itself like this:
    face_image = image[top:bottom, left:right]
    
        #Blur face
    blur_image = cv.GaussianBlur(face_image,(99,99),1)
    cv.imshow('blur image',blur_image)
    pil_image = Image.fromarray(face_image)
    pil_image.show()

"""
encodings = []
names = []

# Training directory
train_dir = os.listdir('dataset/')
train_dir.remove('All images')
# Loop through each person in the training directory
for person in train_dir:
    pix = os.listdir("dataset/" + person)

    # Loop through each training image for the current person
    for person_img in pix:
        # Get the face encodings for the face in each image file
        face = face_recognition.load_image_file("dataset/" + person + "/" + person_img)
        face_bounding_boxes = face_recognition.face_locations(face)

        #If training image contains exactly one face
        if len(face_bounding_boxes) == 1:
            print(len(face_bounding_boxes))
            face_enc = face_recognition.face_encodings(face)[0]
            # Add face encoding for current image with corresponding label (name) to the training data
            encodings.append(face_enc)
            names.append(person)
        else:
            print(person + "/" + person_img + " was skipped and can't be used for training")

# Create and train the SVC classifier
clf = svm.SVC(gamma='scale')
clf.fit(encodings,names)

# Load the test image with unknown faces into a numpy array
test_image = face_recognition.load_image_file('dataset/All images/kofi_annan_and_bill_clinton_1.jpeg')

# Find all the faces in the test image using the default HOG-based model
face_locations = face_recognition.face_locations(test_image)
no = len(face_locations)
print("Number of faces detected: ", no)

# Predict all the faces in the test image using the trained classifier
print("Found:")
for i in range(no):
    test_image_enc = face_recognition.face_encodings(test_image)[i]
    name = clf.predict([test_image_enc])
    print(*name)
    """
