import os
import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier('haar_face.xml')

DIR = 'Resources/Faces/train'
people = []

for i in os.listdir(DIR):
    people.append(i)

# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread('Resources/Faces/val/madonna/2.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detect the face in the image
faces_react = haar_cascade.detectMultiScale(gray, 1.1, 2)

for (x, y, w, h) in faces_react:
    faces_roi = gray[y:y+h, x:x+h]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label: {people[label]} with a confidence of {confidence}')

    cv.putText(img, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), thickness=2)
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

cv.imshow('Detect Face', img)
cv.waitKey(0)
