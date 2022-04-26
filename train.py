import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, InputLayer

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

import pickle
import numpy as np


#if you want help than contact me on instagram
# https://www.instagram.com/webfun_official/

x = pickle.load(open("x.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

x = x/255.0


model = Sequential() 


model.add(Conv2D(32,(2,2), input_shape=x.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(64,(2,2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(128 ,(2,2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3)))


model.add(Flatten())

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(35))  #if this not work than replace 35 to 37
model.add(Activation('softmax'))

opt=tf.keras.optimizers.Adam(learning_rate=1e-7)
model.compile(loss="sparse_categorical_crossentropy",
                        optimizer='adam',
                        metrics=['accuracy']) 
model.summary()

model.fit(x,y, batch_size=5, epochs=2, validation_split=0.1)

model.save('model_name.model')