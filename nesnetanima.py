from cvzone.ClassificationModule import Classifier
import cvzone
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
# teachablemachine.withgoogle.com sitesinden okutulan resimler
myClassifier = cvzone.ClassificationModule.Classifier("keras_model.h5","labels.txt")

while True:
    _,img = cap.read()
    predictions=myClassifier.getPrediction(img)
    #ekrana labels.txt içindeki nesnelere verdiğimiz adları yazdırır

    cv2.imshow("image",img)
    cv2.waitKey(1)