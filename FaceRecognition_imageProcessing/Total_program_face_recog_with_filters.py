# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 18:40:28 2020

@author: wridd
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from os import listdir
from os.path import isfile, join

##.............................................................VIDEO_TO_FRAME......................................................................................

def vid_to_frame(vidcap,vidcap2):

  path = os.getcwd()+"\\Normal_Face"
  path2 = os.getcwd()+"\\Detected_Face"

  success,image = vidcap.read()
  success2, image2 = vidcap2.read()

  count = 0
  print("Printing Normal Face.....\n")
  while success:
    cv2.imwrite(path + "\\%d.jpg" % count, image)     # save frame as JPEG file
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
  print('  ')
  print(' *************************************************************** ')
  count2 = 0
  print("\nPrinting Detected Face.....")
  while success2:
    cv2.imwrite(path2 + "\\%d.jpg" % count2, image2)     # save frame as JPEG file
    success2,image2 = vidcap2.read()
    print('Read a new frame: ', success2)
    count2 += 1

vidcap = cv2.VideoCapture('output_normal.avi')
vidcap2 = cv2.VideoCapture('output_detected.avi')

##..............................................................IMAGE HISTOGRAM.....................................................................................

def image_histogram(image_path):

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite('Gray_'+image_path,gray)
    gray_img = cv2.imread('Gray_'+image_path,0)

    equ = cv2.equalizeHist(gray_img)
    res = np.hstack((gray_img,equ)) #stacking images side-by-side
    hist = cv2.calcHist([gray_img],[0],None,[256],[0,256])

    color = ('b','g','r')
    fig2 = plt.figure(1)
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.subplot(122),plt.plot(histr,color = col),plt.xlim([0,256]),plt.title('Histogram of BGR Image')
        plt.subplot(121),plt.imshow(img),plt.title('BGR Image')
    plt.show()

    cv2.imwrite('histogramed'+image_path,res)

    fig = plt.figure(2)

    plt.subplot(221),plt.imshow(img),plt.title('Original Image')
    plt.subplot(222),plt.imshow(gray_img,'gray'),plt.title('Gray Image of Original')
    plt.subplot(223),plt.hist(gray_img.ravel(),256,[0,256]),plt.title('Histogram for gray scale picture')
    plt.subplot(224),plt.imshow(res,'gray'),plt.title('Equalised Histogram')
    plt.show()

    cv2.destroyAllWindows()

##....................................................................COLOUR IMAGE EQUALISED HISTOGRAM ........................................................................

def BGR_Equalised_Histogram(image_path):
    img = cv2.imread(image_path)

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    cv2.imshow('Color input_image', img)
    cv2.imshow('Histogram equalized', img_output)
    cv2.imwrite('Coloured_histo_2.jpg',img_output)
    

##...................................................................IMAGE FILTERS...................................................................................


def image_filters(image_path):

    img = cv2.imread(image_path)
    dst = cv2.fastNlMeansDenoisingColored(img,None,3,3,7,21)
    blur = cv2.blur(img,(5,5),50)
    kernel_sharpening = np.array([[0,-1,0], 
                              [-1, 5,-1],
                              [0,-1,0]])
    sharpened = cv2.filter2D(img, -1, kernel_sharpening)

    #bright_img = cv2.add(img,30)
    contrast_img = cv2.multiply(img,1.1)

    enhanced_img = cv2.add(contrast_img,15)
    def gamma(img,g=1.00):
      invGamma = 1.0 / g
      table = np.array([((i / 255.0) ** invGamma) * 255
          for i in np.arange(0, 256)]).astype("uint8")
      return cv2.LUT(img, table)

    def bright_contr(img):
      brightness = 70
      contrast = 60
      img = np.int16(img)
      img0 = img * (contrast/127+1) - contrast + brightness
      img1 = np.clip(img0, 0, 255)
      img11 = np.uint8(img1)
      return img11    

    img11 = bright_contr(img)

    img12 = gamma(img,1.2)
    

    fig = plt.figure(1)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.subplot(521),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),plt.title('Original Image')
    plt.subplot(522),plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)),plt.title('Denoised Image')
    plt.subplot(523),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),plt.title('Original Image'),plt.xticks([]), plt.yticks([])
    plt.subplot(524),plt.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)),plt.title('Blurred (Mean) Image'),plt.xticks([]), plt.yticks([])
    plt.subplot(525),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),plt.title('Original Image')
    plt.subplot(526),plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)),plt.title('Sharpened Image')
    plt.subplot(527),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),plt.title('Original Image')
    plt.subplot(528),plt.imshow(cv2.cvtColor(img11, cv2.COLOR_BGR2RGB)),plt.title('Bright  and contrasted')
    plt.subplot(529),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),plt.title('Original Image')
    plt.subplot(5,2,10),plt.imshow(cv2.cvtColor(img12, cv2.COLOR_BGR2RGB)),plt.title('Gamma-ed Image')
    #plt.subplot(6,2,11),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),plt.title('Original Image')
    #plt.subplot(6,2,12),plt.imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)),plt.title('Enhanced Image')

    plt.show()

    cv2.imwrite('denoised_image.png',dst)
    cv2.imwrite('blurred_image.png',blur)
    cv2.imwrite('sharpened_image.png',sharpened)
    cv2.imwrite('brightened  and contrasted.png',img11)
    cv2.imwrite('Gammaed_image.png',img12)
    cv2.imwrite('Enhanced image.png',enhanced_img)

##.....................................................................MULTIPLE IMAGES ENHANCEMENT ..................................................................

def mul_img(img, n, dirpath):
    
    sharp = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharp_img = cv2.filter2D(img, -1, sharp)
    gamma = 2.0 
    invGamma= 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    gamma_img =  cv2.LUT(sharp_img, table)
    cv2.imshow('sharp',gamma_img)
    
    # Create target Directory if don't exist
    dirName = [os.getcwd()+"\\training-data-edited\\s1",os.getcwd()+"\\training-data-edited\\s2",
               os.getcwd()+"\\training-data-edited\\s3"]
    # dirName = dirpath
    for dirNames in dirName:
        if not os.path.exists(dirNames):
            os.mkdir(dirNames)
            print("Directory " ,dirNames ," Created ")
        else:    
            print("Directory " ,dirNames ," already exists")
    

    cv2.imwrite(dirpath+"%d.png"%n,gamma_img)
    cv2.waitKey(200)
    cv2.destroyAllWindows()

##.....................................................................FACIAL RECOGNITION............................................................................

def fac_recog():
    subjects = ["","Person 1", "Person 2","Person 3"]

    def detect_face(img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        face_cascade = cv2.CascadeClassifier('opencv-files//lbpcascade_frontalface.xml')
    
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
        if (len(faces) == 0):
            return None, None

        (x, y, w, h) = faces[0]
    
        return gray[y:y+w, x:x+h], faces[0]

    def prepare_training_data(data_folder_path):    
    
        dirs = os.listdir(data_folder_path)

        faces = []

        labels = []

        for dir_name in dirs:

            if not dir_name.startswith("s"):
                continue;          
        
            label = int(dir_name.replace("s", ""))

            subject_dir_path = data_folder_path + "/" + dir_name

            subject_images_names = os.listdir(subject_dir_path)

            for image_name in subject_images_names:

                if image_name.startswith("."):
                    continue;

                image_path = subject_dir_path + "/" + image_name

                image = cv2.imread(image_path)

                cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
                cv2.waitKey(100)

                face, rect = detect_face(image)

                if face is not None:

                    faces.append(face)
                
                    labels.append(label)
            
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
    
        return faces, labels

    print("Preparing data...")
    faces, labels = prepare_training_data("training-data")
    print("Data prepared")

    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    face_recognizer.train(faces, np.array(labels))

    def draw_rectangle(img, rect):
        (x, y, w, h) = rect
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    def draw_text(img, text, x, y):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    def predict(test_img):

        img = test_img.copy()

        face, rect = detect_face(img)

        label, confidence = face_recognizer.predict(face)

        label_text = subjects[label]  

        draw_rectangle(img, rect)

        draw_text(img, label_text, rect[0], rect[1]-5)
    
        return img

    print("Predicting images...")

    test_img1 = cv2.imread("test-data//test1.jpg")
    test_img2 = cv2.imread("test-data//test2.jpg")
    test_img3 = cv2.imread("test-data//test3.jpg")

    predicted_img1 = predict(test_img1)
    predicted_img2 = predict(test_img2)
    predicted_img3 = predict(test_img3)
    
    print("Prediction complete")

    cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400, 500)))
    cv2.imshow(subjects[2], cv2.resize(predicted_img2, (400, 500)))
    cv2.imshow(subjects[3], cv2.resize(predicted_img3, (400, 500)))

    cv2.imwrite("Predicted_Person1.jpg",predicted_img1)
    cv2.imwrite("Predicted_Person2.jpg",predicted_img2)
    cv2.imwrite("Predicted_Person3.jpg",predicted_img3)

##....................................................................FACIAL RECOGNITION EDITED .....................................................................

def fac_recog_edited():
    subjects = ["","Person 1", "Person 2","Person 3"]

    def detect_face(img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        face_cascade = cv2.CascadeClassifier('opencv-files//lbpcascade_frontalface.xml')
    
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
        if (len(faces) == 0):
            return None, None

        (x, y, w, h) = faces[0]
    
        return gray[y:y+w, x:x+h], faces[0]

    def prepare_training_data(data_folder_path):    
    
        dirs = os.listdir(data_folder_path)

        faces = []

        labels = []

        for dir_name in dirs:

            if not dir_name.startswith("s"):
                continue;          
        
            label = int(dir_name.replace("s", ""))

            subject_dir_path = data_folder_path + "/" + dir_name

            subject_images_names = os.listdir(subject_dir_path)

            for image_name in subject_images_names:

                if image_name.startswith("."):
                    continue;

                image_path = subject_dir_path + "/" + image_name

                image = cv2.imread(image_path)

                cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
                cv2.waitKey(100)

                face, rect = detect_face(image)

                if face is not None:

                    faces.append(face)
                
                    labels.append(label)
            
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
    
        return faces, labels

    print("Preparing data...")
    faces, labels = prepare_training_data("training-data-edited")
    print("Data prepared")

    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    face_recognizer.train(faces, np.array(labels))

    def draw_rectangle(img, rect):
        (x, y, w, h) = rect
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    def draw_text(img, text, x, y):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    def predict(test_img):

        img = test_img.copy()

        face, rect = detect_face(img)

        label, confidence = face_recognizer.predict(face)

        label_text = subjects[label]  

        draw_rectangle(img, rect)

        draw_text(img, label_text, rect[0], rect[1]-5)
    
        return img

    print("Predicting images...")

    test_img1 = cv2.imread("test-data//test1.jpg")
    test_img2 = cv2.imread("test-data//test2.jpg")
    test_img3 = cv2.imread("test-data//test3.jpg")

    predicted_img1 = predict(test_img1)
    predicted_img2 = predict(test_img2)
    predicted_img3 = predict(test_img3)
    
    print("Prediction complete")

    cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400, 500)))
    cv2.imshow(subjects[2], cv2.resize(predicted_img2, (400, 500)))
    cv2.imshow(subjects[3], cv2.resize(predicted_img3, (400, 500)))

    cv2.imwrite("Predicted_Person1.jpg",predicted_img1)
    cv2.imwrite("Predicted_Person2.jpg",predicted_img2)
    cv2.imwrite("Predicted_Person3.jpg",predicted_img3)


##.....................................................................MENU FOR PROGRAM .............................................................................
while True:
    print("MENU FOR FACIAL RECOGNITION WITH IMAGE ENHANCEMENTS\n")
    print("\n1. Convert the video to frames\n")
    print("\n2. Histogram Analysis of Image in BW Format\n")
    print("\n3. Equalised Histogram of Coloured Image\n")
    print("\n4. Appliying Filters in Spatial Domain\n")
    print("\n5. Enhance Training Pictures\n")
    print("\n6. Facial Recognition \n")
    print("\n7. Facial Recognition with Enhanced Images\n")
    x = int(input("\nEnter Your Choice (1-7): "))

    if x == 1:
        vid_to_frame(vidcap,vidcap2)  

    elif x == 2:
        image_histogram('wriddhirup.jpeg')

    elif x == 3:
        BGR_Equalised_Histogram('wriddhirup.jpeg')
        
    elif x == 4: 
        image_filters('wriddhirup.jpeg')

    elif x == 5:  
        mypath=['training-data/s1/','training-data/s2/','training-data/s3/']
        for mypaths in mypath:
            onlyfiles = [ f for f in listdir(mypaths) if isfile(join(mypaths,f)) ]
            images = np.empty(len(onlyfiles), dtype=object)
            for n in range(0, len(onlyfiles)):
              images[n] = cv2.imread( join(mypaths,onlyfiles[n]) )
              x = mul_img(images[n],n, mypaths)       

    elif x == 6:
        fac_recog()

    elif x == 7:
        print('Face Recognition with Enchanced Images')
        fac_recog_edited()

##    elif x == 8:
##        print('Eigen Faces')

    else:
        print('\nMenu Terminated')
        break

