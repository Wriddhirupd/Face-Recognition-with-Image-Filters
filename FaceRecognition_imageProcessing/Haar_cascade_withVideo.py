# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 18:31:16 2020

@author: wridd
"""
import cv2
import numpy as np

def haar_cascade_face_detect():
    
    face_cascade = cv2.CascadeClassifier('opencv-files//haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('opencv-files//haarcascade_eye.xml')
    nose_cascade = cv2.CascadeClassifier('opencv-files//Nariz.xml')
    mouth_cascade = cv2.CascadeClassifier('opencv-files//Mouth.xml')
    smile_cascade = cv2.CascadeClassifier('opencv-files//haarcascade_smile.xml')

    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter('output_normal.avi', fourcc, 20.0, (640,480))
    out2 = cv2.VideoWriter('output_detected.avi', fourcc, 20.0, (640,480))

    while True: 

        ret, img = cap.read()
        frame = img
        out.write(frame)
        cv2.imshow('normal_capture',frame)
        #if ret!=0:
        gray = cv2. cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)
    
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
        
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh),(0,255,0),2)

            nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in nose_rects:
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 3)
                
            mouth_rects = mouth_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in mouth_rects:
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,128,0), 3)

            smile = smile_cascade.detectMultiScale(roi_gray,1.7,22)
            for (sx,sy,sw,sh) in smile:
                cv2.rectangle(roi_color, (sx,sy), (sx+sw,sy+sh),(0,255,255),1)
        
        cv2.imshow('face_detected',img)
        cv2.imwrite('detected_face_eyes_nose_mouth_smile.png',img)

        out2.write(img)
        
        k = cv2.waitKey(1) or 0xFF == ord('q')
        if k ==1:
            break
        else:
            pass
       
    cap.release()
    out.release()
    out2.release()
    cap.destroyAllWindows()


haar_cascade_face_detect()

