# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 10:12:15 2021

@author: LernSven
"""

import numpy as np
from tensorflow import keras
import tensorflow as tf

import time

class MnistDataset:
    def __init__(self, data):
        self.x_train = data["x_train"].reshape(60000,28,28,1).astype(dtype=np.float32)  # use float precision
        self.y_train = data["y_train"].astype(dtype=np.float32)
        #print("inside class:", data["y_train"].size)
        self.x_test = data["x_test"].reshape(10000,28,28,1).astype(dtype=np.float32)
        self.y_test = data["y_test"].astype(dtype=np.float32)
        self.x_train = self.x_train#/255.0 # norm to one
        self.x_test = self.x_test#/255.0
 
        
class OptimizerParameters:
    def __init__(self, bs, ep, eta):
        self.bs = bs
        self.ep = ep
        self.eta = eta

def create_model(net, summary=False):
    # setup of Tensorflow model
    activation = "relu" # lambda x: tf.keras.activations.relu(x, alpha=0.3) # leaky relu
    model = keras.Sequential()    
    if net == 0:
        model.add(keras.Input(shape=(28, 28, 1)))  # pixels^2 * Channels
        model.add(keras.layers.Conv2D(32, (3, 3), activation=activation,
                                      kernel_initializer="he_uniform"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation=activation,
                                      kernel_initializer="he_uniform"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(2048, activation=activation,
                                     kernel_initializer="he_uniform"))
        model.add(keras.layers.Dense(10, activation="softmax",
                                     kernel_initializer="he_uniform"))
        
    if net == 1:    # idea: less conv layers, less pooling
        model.add(keras.Input(shape=(28, 28, 1)))  # pixels^2 * Channels
        model.add(keras.layers.Conv2D(32, (3, 3), activation=activation,
                                      kernel_initializer="he_uniform"))
        #model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(192, activation=activation,
                                     kernel_initializer="he_uniform"))
        model.add(keras.layers.Dense(10, activation="softmax",
                                     kernel_initializer="he_uniform"))
        
    if net == 2:    # idea: pooling, but bigger dense layer
        model.add(keras.Input(shape=(28, 28, 1)))  # pixels^2 * Channels
        model.add(keras.layers.Conv2D(32, (3, 3), activation=activation,
                                      kernel_initializer="he_uniform"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(192*4, activation=activation,
                                     kernel_initializer="he_uniform"))
        model.add(keras.layers.Dense(10, activation="softmax",
                                     kernel_initializer="he_uniform"))
        
    if net == 3:    # idea: more convolutional channels, less dense layer
        model.add(keras.Input(shape=(28, 28, 1)))  # pixels^2 * Channels
        model.add(keras.layers.Conv2D(64, (3, 3), activation=activation,
                                      kernel_initializer="he_uniform"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(192*2, activation=activation,
                                     kernel_initializer="he_uniform"))
        model.add(keras.layers.Dense(10, activation="softmax",
                                     kernel_initializer="he_uniform"))
        
    if net == 4:    # idea: split dense layer into two
        model.add(keras.Input(shape=(28, 28, 1)))  # pixels^2 * Channels
        model.add(keras.layers.Conv2D(64, (3, 3), activation=activation,
                                      kernel_initializer="he_uniform"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(256+128, activation=activation,
                                     kernel_initializer="he_uniform"))
        model.add(keras.layers.Dense(256, activation=activation,
                                     kernel_initializer="he_uniform"))
        model.add(keras.layers.Dense(10, activation="softmax",
                                     kernel_initializer="he_uniform"))
        
    if net == 5:    # idea: same as 4, but another dense layer #
        model.add(keras.Input(shape=(28, 28, 1)))  # pixels^2 * Channels
        model.add(keras.layers.Conv2D(64, (3, 3), activation=activation,
                                      kernel_initializer="he_uniform"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(256+128, activation=activation,
                                     kernel_initializer="he_uniform"))
        model.add(keras.layers.Dense(256, activation=activation,
                                     kernel_initializer="he_uniform"))
        model.add(keras.layers.Dense(256, activation=activation,
                                     kernel_initializer="he_uniform"))
        model.add(keras.layers.Dense(10, activation="softmax",
                                     kernel_initializer="he_uniform"))
        
    if net == 6:    # idea: generalize 3 more with dropout layer
        model.add(keras.Input(shape=(28, 28, 1)))  # pixels^2 * Channels
        model.add(keras.layers.Conv2D(64, (3, 3), activation=activation,
                                      kernel_initializer="he_uniform"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(192*2, activation=activation,
                                     kernel_initializer="he_uniform"))
        model.add(keras.layers.Dense(10, activation="softmax",
                                     kernel_initializer="he_uniform"))
        
    if net == 7:    # idea: generalize 3 more with dropout layer; MORE DROPOUT, Dropout everywhere!
        model.add(keras.Input(shape=(28, 28, 1)))  # pixels^2 * Channels
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Conv2D(64, (3, 3), activation=activation,
                                      kernel_initializer="he_uniform"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(192*2, activation=activation,
                                     kernel_initializer="he_uniform"))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(10, activation="softmax",
                                     kernel_initializer="he_uniform"))

    if net == 8:    # idea: less extreme dropout
        model.add(keras.Input(shape=(28, 28, 1)))  # pixels^2 * Channels
        model.add(keras.layers.Dropout(0.05))
        model.add(keras.layers.Conv2D(64, (3, 3), activation=activation,
                                      kernel_initializer="he_uniform"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(192*2, activation=activation,
                                     kernel_initializer="he_uniform"))
        model.add(keras.layers.Dropout(0.1))
        model.add(keras.layers.Dense(10, activation="softmax",
                                     kernel_initializer="he_uniform"))
        
    if net == 9:    # idea: more convolutions () (Inspiration: Imagenet NN)
        model.add(keras.Input(shape=(28, 28, 1)))  # pixels^2 * Channels
        model.add(keras.layers.Conv2D(128, (3, 3), activation=activation,
                                      kernel_initializer="he_uniform"))
        #model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(keras.layers.Conv2D(128, (3, 3), activation=activation,
                                      kernel_initializer="he_uniform"))
        #model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation=activation,
                                      kernel_initializer="he_uniform"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        #model.add(keras.layers.Dropout(0.3))
# =============================================================================
#         model.add(keras.layers.Dense(512, activation=activation,
#                                      kernel_initializer="he_uniform"))
# =============================================================================
        model.add(keras.layers.Dense(512, activation=activation,
                                     kernel_initializer="he_uniform"))
        #model.add(keras.layers.Dropout(0.1))
        model.add(keras.layers.Dense(10, activation="softmax",
                                     kernel_initializer="he_uniform"))
        
    if net == 10:    # idea: more convolutions () (Inspiration: Imagenet NN)
        model.add(keras.Input(shape=(28, 28, 1)))  # pixels^2 * Channels
        layer = tf.keras.layers.GaussianNoise(0.05)
        model.add(keras.layers.Conv2D(128, (3, 3), activation=activation,
                                      kernel_initializer="he_uniform"))
        #model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(keras.layers.Conv2D(128, (3, 3), activation=activation,
                                      kernel_initializer="he_uniform"))
        #model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation=activation,
                                      kernel_initializer="he_uniform"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        #model.add(keras.layers.Dropout(0.3))
# =============================================================================
#         model.add(keras.layers.Dense(512, activation=activation,
#                                      kernel_initializer="he_uniform"))
# =============================================================================
        model.add(keras.layers.Dense(512, activation=activation,
                                     kernel_initializer="he_uniform"))
        #model.add(keras.layers.Dropout(0.1))
        model.add(keras.layers.Dense(10, activation="softmax",
                                     kernel_initializer="he_uniform"))
        
    if net == 11:    # idea: more convolutional channels, less dense layer
        model.add(keras.Input(shape=(28, 28, 1)))  # pixels^2 * Channels
        layer = tf.keras.layers.GaussianNoise(0.05) # THIS NOISE IS NOT ACTUALLY ADDED TO THE MODEL!
        model.add(keras.layers.Conv2D(64, (3, 3), activation=activation,
                                      kernel_initializer="he_uniform"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(192*2, activation=activation,
                                     kernel_initializer="he_uniform"))
        model.add(keras.layers.Dense(10, activation="softmax",
                                     kernel_initializer="he_uniform"))
        
    if net == 12:    # idea: more convolutional channels, less dense layer
        model.add(keras.Input(shape=(28, 28, 1)))  # pixels^2 * Channels
        model.add(tf.keras.layers.GaussianNoise(0.005)) #layer = tf.keras.layers.GaussianNoise(0.05) 
        model.add(keras.layers.Conv2D(64, (5, 5), activation=activation,
                                      kernel_initializer="he_uniform"))
        model.add(keras.layers.Conv2D(64, (3, 3), activation=activation,
                                      kernel_initializer="he_uniform"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(192*2, activation=activation,
                                     kernel_initializer="he_uniform"))
        model.add(keras.layers.Dense(10, activation="softmax",
                                     kernel_initializer="he_uniform"))
    
    if net == 13:    # idea: 11 with more generalization
        model.add(keras.Input(shape=(28, 28, 1)))  # pixels^2 * Channels
        layer = tf.keras.layers.GaussianNoise(0.05)
        model.add(keras.layers.Conv2D(64, (3, 3), activation=activation,
                                      kernel_initializer="he_uniform"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(192*2, activation=activation,
                                     kernel_initializer="he_uniform"))
        model.add(keras.layers.Dense(10, activation="softmax",
                                     kernel_initializer="he_uniform"))

    if net == 14:    # idea: 11 but with more channels, less dense
        model.add(keras.Input(shape=(28, 28, 1)))  # pixels^2 * Channels
        layer = tf.keras.layers.GaussianNoise(0.05)
        model.add(keras.layers.Conv2D(128, (3, 3), activation=activation,
                                      kernel_initializer="he_uniform"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        #model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(192, activation=activation,
                                     kernel_initializer="he_uniform"))
        model.add(keras.layers.Dense(10, activation="softmax",
                                     kernel_initializer="he_uniform"))
        
    if net == 15:    # idea: just like 11 but with added noise (actually this time)
        model.add(keras.Input(shape=(28, 28, 1)))  # pixels^2 * Channels
        model.add(tf.keras.layers.GaussianNoise(0.05))
        model.add(keras.layers.Conv2D(64, (3, 3), activation=activation,
                                      kernel_initializer="he_uniform"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(192*2, activation=activation,
                                     kernel_initializer="he_uniform"))
        model.add(keras.layers.Dense(10, activation="softmax",
                                     kernel_initializer="he_uniform"))
        
    if net == 16:    # idea: net 3 with random flip
        model.add(keras.Input(shape=(28, 28, 1)))  # pixels^2 * Channels
        #model.add(keras.layers.RandomFlip())
        model.add(keras.layers.experimental.preprocessing.RandomFlip())

        model.add(keras.layers.Conv2D(64, (3, 3), activation=activation,
                                      kernel_initializer="he_uniform"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(192*2, activation=activation,
                                     kernel_initializer="he_uniform"))
        model.add(keras.layers.Dense(10, activation="softmax",
                                     kernel_initializer="he_uniform"))
        
    if net == 17:    # idea: net 3 with random flip
        model.add(keras.Input(shape=(28, 28, 1)))  # pixels^2 * Channels
        #model.add(keras.layers.RandomFlip())
        model.add(tf.keras.layers.GaussianNoise(0.05))
        model.add(keras.layers.experimental.preprocessing.RandomFlip()) # in 2.7 this function is moved to normal layer

        model.add(keras.layers.Conv2D(64, (3, 3), activation=activation,
                                      kernel_initializer="he_uniform"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(192*2, activation=activation,
                                     kernel_initializer="he_uniform"))
        model.add(keras.layers.Dense(10, activation="softmax",
                                     kernel_initializer="he_uniform"))
        
    if net == 18:    # idea: 3, but bigger kernel
        model.add(keras.Input(shape=(28, 28, 1)))  # pixels^2 * Channels
        model.add(keras.layers.Conv2D(64, (5, 5), activation=activation,
                                      kernel_initializer="he_uniform"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(192*2, activation=activation,
                                     kernel_initializer="he_uniform"))
        model.add(keras.layers.Dense(10, activation="softmax",
                                     kernel_initializer="he_uniform"))
        
    if net == 19:    # idea: 3, but smaller kernel
        model.add(keras.Input(shape=(28, 28, 1)))  # pixels^2 * Channels
        model.add(keras.layers.Conv2D(64, (2, 2), activation=activation,
                                      kernel_initializer="he_uniform"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(192*2, activation=activation,
                                     kernel_initializer="he_uniform"))
        model.add(keras.layers.Dense(10, activation="softmax",
                                     kernel_initializer="he_uniform"))
        
    if net == 20:    # idea: 3, but bigger kernel
        model.add(keras.Input(shape=(28, 28, 1)))  # pixels^2 * Channels
        model.add(keras.layers.Conv2D(64, (4, 4), activation=activation,
                                      kernel_initializer="he_uniform"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(192*2, activation=activation,
                                     kernel_initializer="he_uniform"))
        model.add(keras.layers.Dense(10, activation="softmax",
                                     kernel_initializer="he_uniform"))
        
    if net == 21: # 98% accuracy from kaggle https://www.kaggle.com/rutvikdeshpande/fashion-mnist-cnn-beginner-98
        model.add(keras.Input(shape=(28, 28, 1)))  # pixels^2 * Channels
        model.add(keras.layers.Conv2D(32, (3, 3), padding="same", activation=activation,
                                      kernel_initializer="he_uniform"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), padding="same", activation=activation,
                                      kernel_initializer="he_uniform"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation=activation,
                                     kernel_initializer="he_uniform"))
        model.add(keras.layers.Dense(10, activation="softmax",
                                     kernel_initializer="he_uniform"))
    if net == 22: # 98% accuracy from kaggle https://www.kaggle.com/rutvikdeshpande/fashion-mnist-cnn-beginner-98        
        model.add(keras.Input(shape=(28, 28, 1)))
        model.add(keras.layers.Conv2D(32, 3, padding='same', activation='relu',
                  kernel_initializer='he_normal', input_shape=(28, 28, 1)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Conv2D(
            64, 3, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(
            128, 3, padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(
            128, 3, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(512, activation='relu'))

        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Dense(10, activation='softmax'))

    if net == 23: # kaggle, my improvements          
        model.add(keras.Input(shape=(28, 28, 1)))
        model.add(keras.layers.Conv2D(32, 3, padding='same', activation='relu',
                  kernel_initializer='he_normal', input_shape=(28, 28, 1)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Conv2D(
            64, 3, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(
            128, 3, padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(
            128, 3, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(
            128, 3, padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(
            128, 3, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(512, activation='relu'))

        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Dense(10, activation='softmax'))
        
    if net == 24: # double kaggle with less pooling --> Moore parameters
        model.add(keras.Input(shape=(28, 28, 1)))
        model.add(keras.layers.Conv2D(32, 3, padding='same', activation='relu',
                  kernel_initializer='he_normal', input_shape=(28, 28, 1)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Conv2D(
            64, 3, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(
            128, 3, padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(
            128, 3, padding='same', activation='relu'))
        #model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(
            128, 3, padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(
            128, 3, padding='same', activation='relu'))
        #model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(512, activation='relu'))

        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Dense(10, activation='softmax'))
        
    if net == 25: # 98% accuracy from kaggle https://www.kaggle.com/rutvikdeshpande/fashion-mnist-cnn-beginner-98        
        model.add(keras.Input(shape=(28, 28, 1)))
        model.add(keras.layers.Conv2D(32, 3, padding='same', activation='relu',
                  kernel_initializer='he_normal', input_shape=(28, 28, 1)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Conv2D(
            64, 3, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Dropout(0.3))
        #model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(
            128, 3, padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(
            128, 3, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Flatten())
        #model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(512, activation='relu'))

        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Dense(10, activation='softmax'))
        
    if net == 26: # Dense net      
        model.add(keras.Input(shape=(28, 28, 1)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(512*4, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Dense(512*4, activation='relu'))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(512*4, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Dense(512*4, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(512*4, activation='relu'))

        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Dense(10, activation='softmax'))
        
    if net == 27: # Bigger dense net     
        model.add(keras.Input(shape=(28, 28, 1)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(512*4*4, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Dense(512*4*4, activation='relu'))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(512*4*4, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Dense(512*4, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(512, activation='relu'))

        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Dense(10, activation='softmax'))
        
    if net == 28: # Kaggle, but without dropouts     
        model.add(keras.Input(shape=(28, 28, 1)))
        model.add(keras.layers.Conv2D(32, 3, padding='same', activation='relu',
                  kernel_initializer='he_normal', input_shape=(28, 28, 1)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Conv2D(
            64, 3, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        #model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(
            128, 3, padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(
            128, 3, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        #model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(512, activation='relu'))

        #model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Dense(10, activation='softmax'))
        
    if net == 29: # 98% accuracy from kaggle simply bigger dense layer https://www.kaggle.com/rutvikdeshpande/fashion-mnist-cnn-beginner-98        
        model.add(keras.Input(shape=(28, 28, 1)))
        model.add(keras.layers.Conv2D(64, 3, padding='same', activation='relu',
                  kernel_initializer='he_normal', input_shape=(28, 28, 1)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Conv2D(
            256, 3, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(
            128*2, 3, padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(
            128+64, 3, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(2048, activation='relu'))

        model.add(keras.layers.Dropout(0.4))    # bigger dropout since dense layer is bigger
        model.add(keras.layers.Dense(10, activation='softmax'))
        
    if net == 30: # 98% accuracy from kaggle simply bigger dense layer less dropout https://www.kaggle.com/rutvikdeshpande/fashion-mnist-cnn-beginner-98        
        model.add(keras.Input(shape=(28, 28, 1)))
        model.add(keras.layers.Conv2D(64, 3, padding='same', activation='relu',
                  kernel_initializer='he_normal', input_shape=(28, 28, 1)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Conv2D(
            256, 3, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(
            128*2, 3, padding='same', activation='relu'))
        model.add(keras.layers.Conv2D(
            128+64, 3, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(2048, activation='relu'))

        model.add(keras.layers.Dropout(0.25))    # bigger dropout since dense layer is bigger
        model.add(keras.layers.Dense(10, activation='softmax'))
 
    if summary:
        model.summary() # overview of nr. of parameters before training        
        
    return model

def calc_accuracy(model, x_test, y_test):    
   # y_tilde = model.predict(x_test.reshape(10000,28,28,1))
    y_tilde = model.predict(x_test)
    guess   = np.argmax(y_tilde, 1).T # round to integer
    
    y_converted = np.zeros(len(guess), dtype=np.int64)
    for index, y in enumerate(y_test):
        for entry in y:
            if entry < 0.5:
                y_converted[index] += 1
            else: 
                break
    
    accuracy = np.count_nonzero(guess==y_converted)/len(guess)  #
    return accuracy

def calc_cce_loss(model, x_test, y_test):
    y_tilde = model.predict(x_test)
    cce = tf.keras.losses.CategoricalCrossentropy()
    return cce(y_test, y_tilde).numpy()


def evaluate_model(model, data, params):       
    # quick evaluation
    accuracy = calc_accuracy(model, data.x_test, data.y_test)
    cce_loss = calc_cce_loss(model, data.x_test, data.y_test)
    test_accuracy = calc_accuracy(model, data.x_train, data.y_train)
    test_cce_loss = calc_cce_loss(model, data.x_train, data.y_train)
    
    print("The accuracy of the trained model is:", accuracy)    
    
    # save models with different names
    filepath = "saved_models/"+str(int(time.time()))+"_"+str(accuracy)
    model.save(filepath=filepath) # save model
    # save summary:
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    
    with open(file=filepath+"/summary.txt", mode="w") as f:
        f.writelines("\n".join(stringlist)) # create single string from list, with \n separator
        f.write("\n")
        f.write("Optimizer (Adam) Parameters:\n")
        f.write(("batchsize:"+str(params.bs)+"\n"))
        f.write(("nr. of epochs:"+ str(params.ep)+"\n"))
        f.write(("used learningrate:"+ str(params.eta)+"\n"))
        f.write("\n")
        f.write(("Accuracy of the model in training data:"+str(test_accuracy)+"\n"))
        f.write(("CCE loss of the model in training data:"+str(test_cce_loss)+"\n"))
        f.write("\n")
        f.write(("Accuracy of the model in test data:"+str(accuracy)+"\n"))
        f.write(("CCE loss of the model in test data:"+str(cce_loss)+"\n"))
    return test_accuracy
        
def train(data, params, model_number):
    model = create_model(net=model_number)
    # configure model for training
    model.compile(optimizer=keras.optimizers.Adam(params.eta),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(data.x_train, data.y_train, batch_size=params.bs, epochs=params.ep)
    return model

# =============================================================================
# def train_callback(data, params, model_number, callback):
#     model = create_model(net=model_number)
#     # configure model for training
#     model.compile(optimizer=keras.optimizers.Adam(params.eta),
#                   loss="categorical_crossentropy",
#                   metrics=["accuracy"])
#     model.fit(data.x_train, data.y_train, batch_size=params.bs,
#               epochs=params.ep,
#               callbacks=[callback])
#     # maybe use history of fit!
#     return model
# =============================================================================

def train_callback(data, params, model_number, callback):
    model = create_model(net=model_number)
    # configure model for training
    model.compile(optimizer=keras.optimizers.Adam(params.eta),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(data.x_train, data.y_train, batch_size=params.bs,
              epochs=params.ep,
              callbacks=[callback], 
              validation_data=(data.x_test, data.y_test))
    # maybe use history of fit!
    return model




      

# =============================================================================
# not used right now
# # categories
# ct = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
#       'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
# =============================================================================        


#print('Accuracy with TensorFlow =', np.sum(guess == y_test)/100)

# =============================================================================
# new_model = keras.models.load_model("saved_models/1640685355")
# new_model.summary()
# =============================================================================






