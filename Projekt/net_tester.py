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

# =============================================================================
# DATA = np.load('datasets/fashion_mnist.npz')
# x_train, y_train = DATA['x_train'].reshape(60000,28,28,1), DATA['y_train'] 
# x_test, y_test = DATA['x_test'].reshape(10000,28,28,1), DATA['y_test']
# x_train, x_test = x_train / 255.0, x_test / 255.0
# =============================================================================

# =============================================================================
# not used right now
# # categories
# ct = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
#       'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
# =============================================================================
# =============================================================================
# class MnistDataset:
#     def __init__(self, data):
#         self.x_train = data["x_train"].reshape(60000,28,28,1)
#         self.y_train = data["y_train"]
#         #print("inside class:", data["y_train"].size)
#         self.x_test = data["x_test"].reshape(10000,28,28,1)
#         self.y_test = data["y_test"]
#         self.x_train = self.x_train/255.0
#         self.x_test = self.x_test/255.0
#  
#         
# class OptimizerParameters:
#     def __init__(self, bs, ep, eta):
#         self.bs = bs
#         self.ep = ep
#         self.eta = eta
# =============================================================================


def test_loop(data, params, model_number):
    params_internal = OptimizerParameters(params.bs, params.ep, params.eta)
    for i in range(1,10):
        params_internal.ep = params.ep*i
        model = lib.train(data, params_internal, model_number)
        test_accuracy = lib.evaluate_model(model, data, params_internal)
        if test_accuracy > 0.9999999:#5:
            print("Training model reched perfect accuracy after epochs:", i*params.ep)
            break
        
def test_callback(data, params, model_number):
    for i in range(0,10):
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=30, restore_best_weights="True")
        model = lib.train_callback(data, params, model_number, callback)
        test_accuracy = lib.evaluate_model(model, data, params)



def main():
    # setup of hyperparameters
    bs = 128 # batchsize
    ep = 500	 # number of epochs
    eta = 0.001 # learning rate
    
    model_number = 15
    data = MnistDataset(np.load("datasets/fashion_mnist.npz"))

    params = OptimizerParameters(bs, ep, eta)
   #test_loop(data, params, model_number)
    test_callback(data, params, model_number)

        
if __name__ == "__main__":
    main()