# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 20:53:50 2019

@author: 60111
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train,axis =1)
x_test = tf.keras.utils.normalize(x_test,axis =1)

model=tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer ='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train, epochs=3)
model.save('epic_num_reader.model')
new_model = tf.keras.models.load_model('epic_num_reader.model')
prediction =new_model.predict(x_test)
i=0
for result in prediction:
    print(np.argmax(result))
    plt.imshow(x_test[i])
    plt.show()
    i=i+1