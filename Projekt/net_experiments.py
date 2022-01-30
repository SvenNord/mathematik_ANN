# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 14:04:06 2021

@author: LernSven
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 12:54:52 2021

@author: LernSven
"""

import numpy as np
from tensorflow import keras
import tensorflow as tf

import time
import mymllib as lib
from mymllib import MnistDataset, OptimizerParameters

import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def test_loop(data, params, model_number):
    params_internal = OptimizerParameters(params.bs, params.ep, params.eta)
    for i in range(1,3):
        params_internal.ep = params.ep*3**i
        #print("I do things!", params.ep*3**i)
        for j in range(0,3):
            #print("I do internal things!", params_internal.ep)
            model = lib.train(data, params_internal, model_number)
            lib.evaluate_model(model, data, params_internal)
            
def data_insight(data):
    print(type(data.x_train[0][0][0][0]))
    #data = data = np.arange(20).reshape(5, 4).astype(np.float32)
    data = data.x_train[1]
    #layer = tf.keras.layers.Dropout(0.1)
    layer = tf.keras.layers.GaussianNoise(0.05)
    data_new = layer(data, training=True)

    for i in range(0, 1):
        plt.figure()
        plt.imshow(data)
        plt.figure()
        plt.imshow(data_new)
         
    
    #test = data.x_train[0]


def main():
    # setup of hyperparameters
    bs = 128 # batchsize
    ep = 1 # number of epochs
    eta = 0.002 # learning rate
    
    model_number = 29
    data = MnistDataset(np.load("datasets/fashion_mnist.npz"))
    params = OptimizerParameters(bs, ep, eta)
# =============================================================================
#     data_insight(data)
# =============================================================================
    model = lib.train(data, params, model_number)
    model.summary()
    #lib.evaluate_model(model, data, params)
    
        
if __name__ == "__main__":
    main()