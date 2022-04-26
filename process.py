import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import scipy.ndimage as sci


#if you want help than contact me on instagram
# https://www.instagram.com/webfun_official/


def cropIt(gray,top=10,left=290,right=290,down=10):
    w, h = gray.shape
    croped_image = gray[top:(w-down), right:(h-left)]
    return croped_image


def resizeIt(img,size=100,median=2):
    img=np.float32(img)
    r,c=img.shape

    resized_img=cv2.resize(img,(size,size))
    filtered_img=sci.median_filter(resized_img,median)
    return np.uint8(filtered_img)

def preprocessing(img0,IMG_SIZE=100):
    img_resized=resizeIt(img0,IMG_SIZE,1) 
    #cv2.imshow("intermidieate",img_resized)
    img_blur = cv2.GaussianBlur(img_resized,(5,5),0)
    #cv2.imshow("img_blur",img_blur)
    imgTh=cv2.adaptiveThreshold(img_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,3)
    #cv2.imshow("imgTh",imgTh)
    ret,img_th = cv2.threshold(imgTh,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
    #cv2.imshow("intermidieate",img_th)
    #edges = cv2.Canny(img_resized,170, 300)
    return img_th

DATADIR = "G:\\Acollegefianalyear\\shortdata\\"

ALPHABET = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','1','2','3','4','5','6','7','8','9'] #array containing letters to categorize and create path to video

training_data=[]


IMG_SIZE=200

for category in ALPHABET:
    path = os.path.join(DATADIR,category)  
    print(path)
    for img_path in os.listdir(path): 
        #print(img_path)
        img0 = cv2.imread(os.path.join(path,img_path) ,cv2.IMREAD_GRAYSCALE) 
        img_processed=preprocessing(img0,IMG_SIZE)
        cv2.imshow("input",img_processed)
        cv2.waitKey(1)

        class_num =ALPHABET.index(category)
        training_data.append([img_processed, class_num]) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




cv2.destroyAllWindows()


import random

random.shuffle(training_data)

x = []
y = []

for features,label in training_data:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)


import pickle

pickle_out = open("x.pickle","wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

print('doneqqqq')



















