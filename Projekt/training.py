# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 10:12:15 2021

@author: LernSven
"""

import numpy as np
from tensorflow import keras
import tensorflow as tf

import time

DATA = np.load('fashion_mnist.npz')
x_train, y_train = DATA['x_train'].reshape(60000,28,28), DATA['y_train'] 
x_test, y_test = DATA['x_test'].reshape(10000,28,28), DATA['y_test']
x_train, x_test = x_train / 255.0, x_test / 255.0

x = x_train[:,:,:,np.newaxis] # add new axis to fit tf's expectation (could be nr. of channels)

# categories
ct = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
      'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

# setup of hyperparameters
bs = 128 # batchsize
ep = 30 # number of epochs
eta = 0.002 # learning rate

# setup of Tensorflow model

activation = "relu"
model = keras.Sequential()

model.add(keras.Input(shape=(28, 28, 1)))  # pixels^2 * Channels
model.add(keras.layers.Conv2D(32, (3, 3), activation=activation,
                              kernel_initializer="he_uniform"))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(2048, activation=activation,
                             kernel_initializer="he_uniform"))
model.add(keras.layers.Dense(10, activation="softmax",
                             kernel_initializer="he_uniform"))

# configure model for training
model.compile(optimizer=keras.optimizers.Adam(eta),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.summary() # overview of nr. of parameters before training

print("Start training")
model.fit(x, y_train, batch_size=bs, epochs=ep)

#print(tf.config.list_physical_devices("GPU"))


# save models with different names
#time.gmtime()
model.save(filepath="saved_models/"+str(int(time.time())))


# quick evaluation

y_tilde = model.predict(x_test.reshape(10000,28,28,1))
guess   = np.argmax(y_tilde, 1).T # round to integer

y_converted = np.zeros(len(guess), dtype=np.int64)
for index, y in enumerate(y_test):
    for entry in y:
        if entry < 0.5:
            y_converted[index] += 1
        else: 
            break

accuracy = np.count_nonzero(guess==y_converted)/len(guess)  #

print("The accuracy of the trained model is:", accuracy)    

# save models with different names
#time.gmtime()
model.save(filepath="saved_models/"+str(int(time.time()))+str(accuracy))
            
        


#print('Accuracy with TensorFlow =', np.sum(guess == y_test)/100)

# =============================================================================
# new_model = keras.models.load_model("saved_models/1640685355")
# new_model.summary()
# =============================================================================






