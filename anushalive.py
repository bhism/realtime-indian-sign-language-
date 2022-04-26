# import the opencv library
import cv2
import tensorflow as tf
import numpy as np
import scipy.ndimage as sci
import time
import os

  
# define a video capture object
vid = cv2.VideoCapture(0)
def resizeIt(img,size=100,median=2):
    img=np.float32(img)
    r,c=img.shape
    resized_img=cv2.resize(img,(size,size))
    filtered_img=sci.median_filter(resized_img,median)
    return np.uint8(filtered_img)

def preprocessing(img0,IMG_SIZE=200):
    img_resized=resizeIt(img0,IMG_SIZE,1)
    img_blur = cv2.GaussianBlur(img_resized,(5,5),0)
    imgTh=cv2.adaptiveThreshold(img_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,3)
    ret,img_th = cv2.threshold(imgTh,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
    return img_th


ALPHABET =  ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','1','2','3','4','5','6','7','8','9','callme','fingercross'] 

prev=""
model = tf.keras.models.load_model("model_name.model")
prev_time = time.time()
# vid.set(3,80000)

while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    image = cv2.flip(frame,1)

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    IMG_SIZE = 200
    img_test = preprocessing(img_gray,IMG_SIZE)
    prediction = model.predict([img_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)])
    text = ALPHABET[int(np.argmax(prediction[0]))]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_test, 
                text, 
                (50, 50), 
                font, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)
    # Display the resulting frame
    cv2.imshow('frame', img_test)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()




# python anushalive.py 
















# cam_capture = cv2.VideoCapture(0)
# cv2.destroyAllWindows()
# upper_left = (50, 50)
# bottom_right = (300, 300)
# while True:
#     _, image_frame = cam_capture.read()
    
#     #Rectangle marker
#     r = cv2.rectangle(image_frame, upper_left, bottom_right, (100, 50, 200), 5)
#     rect_img = image_frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]
    
#     sketcher_rect = rect_img
#     sketcher_rect = sketch_transform(sketcher_rect)
    
#     #Conversion for 3 channels to put back on original image (streaming)
#     sketcher_rect_rgb = cv2.cvtColor(sketcher_rect, cv2.COLOR_GRAY2RGB)
    
#     #Replacing the sketched image on Region of Interest
#     image_frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]] = sketcher_rect_rgb
# cv2.imshow("Sketcher ROI", image_frame)
#     if cv2.waitKey(1) == 13:
#         break
        
# cam_capture.release()
# cv2.destroyAllWindows()