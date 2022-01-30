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
    for i in range(0,5):
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=30, restore_best_weights="True")
        model = lib.train_callback(data, params, model_number, callback)
        test_accuracy = lib.evaluate_model(model, data, params)
        
def test_callback2(data, params, model_number):
    for i in range(0,10):
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=50, restore_best_weights="True")
        model = lib.train_callback(data, params, model_number, callback)
        test_accuracy = lib.evaluate_model(model, data, params)
        
def test_callback_quick(data, params, model_number):
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=10, restore_best_weights="True")
    model = lib.train_callback(data, params, model_number, callback)
    test_accuracy = lib.evaluate_model(model, data, params)



def main():
    # setup of hyperparameters
    bs = 128 # batchsize
    ep = 500 # number of epochs
    eta = 0.001 # learning rate
    
    model_number = 29
    data = MnistDataset(np.load("datasets/fashion_mnist.npz"))

    params = OptimizerParameters(bs, ep, eta)
   #test_loop(data, params, model_number)
# =============================================================================
#     for i in range(0,26):
#         test_callback_quick(data, params, i)
# =============================================================================
    test_callback2(data, params, model_number)
    #test_callback(data, params, model_number)
    #test_callback2(data, params, model_number)

        
if __name__ == "__main__":
    main()