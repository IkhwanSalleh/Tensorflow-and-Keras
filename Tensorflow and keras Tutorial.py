# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 12:09:50 2019

@author: 60111
"""
#This is an example of Keras and tensorflow machines learning.
#This code is used to train the model to diffrentiate between a dog or a cat from the pictures
#This is a Sequational model code.
#Images of Dog and cat can be download from https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765
#enjoy!!

import tensorflow as tf
import numpy as np
import os
import cv2
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time

NAME = f"CATS-VS-DOG-ANALYZER-64X2-{int(time.time())}"
tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

ImageDIR = "C:/Users/60111/Desktop/Keras tensorflow/kagglecatsanddogs_3367a/PetImages"
CATEGORIES = ["Dog","Cat"]
training_data =[]
IMG_SIZE = 70

def Create_Training_data():
    for category in  CATEGORIES:
        path = os.path.join(ImageDIR,category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array= cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                
                new_array=cv2.resize(img_array, (IMG_SIZE , IMG_SIZE))
                training_data.append([new_array,class_num])
            except:
                pass
            
Create_Training_data()
random.shuffle(training_data)


x = []
y = []

for feature,label in training_data:
    x.append(feature)
    y.append(label)  
    print(label)
    print(len(y))    
X = np.array(x).reshape(-1 , IMG_SIZE , IMG_SIZE , 1)

X =tf.keras.utils.normalize(X)

model = Sequential()
model.add(Conv2D(64, (3,3) , input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss = "binary_crossentropy",
              optimizer = "adam",
              metrics=['accuracy'])

model.fit(X,y , batch_size = 32 ,validation_split = 0.33 , epochs = 30, callbacks= [tensorboard])
model.save('Cat-dog-CNN.model')