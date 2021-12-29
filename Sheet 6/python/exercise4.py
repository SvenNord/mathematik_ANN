import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as tfk

from random import randrange
from time import time      # For time measuring

from networks import SequentialNet
from layers import *
from optimizers import *
from activations import *
from initializers import *

DATA = np.load('fashion_mnist.npz')
x_train, y_train = DATA['x_train'].reshape(60000,28,28), DATA['y_train'] 
x_test,  y_test  = DATA['x_test'].reshape(10000,28,28), DATA['y_test']
x_train, x_test  = x_train / 255.0, x_test / 255.0

x    = x_train[:,np.newaxis,:,:]
x_TF = x_train[:,:,:,np.newaxis]

"""
Categories for Fashion MNIST. Category i is ct[i].
"""
ct = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
      'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

"""
Set up the network with our library
"""
bs, ep, eta = 128, 30, 0.001


net = SequentialNet((1,28,28))
net.add_conv2D((32,3,3), 
               afun=ReLU(), 
               optim=Adam(eta), 
               initializer=
               HeUniform(),
               eval_method='im2col')
net.add_pool2D((2,2))
net.add_conv2D((64,3,3),
               afun=ReLU(),
               optim=Adam(eta),
               initializer=HeUniform(),
               eval_method='im2col')
net.add_pool2D((2,2))
net.add_flatten() 
net.add_dense(2048,
              afun=ReLU(),
              optim=Adam(eta),
              initializer=HeUniform())
net.add_dense(10,
              afun=SoftMax(),
              optim=Adam(eta),
              initializer=HeUniform())


"""
Set up the network with TensorFlow
"""
activation = 'relu'
input_shape=(28,28,1)

net_TF = tfk.Sequential()

net_TF.add(tfk.Input(shape=input_shape))
net_TF.add(tfk.layers.Conv2D(32, (3,3), 
                             activation=activation,
                             kernel_initializer='he_uniform'))
net_TF.add(tfk.layers.MaxPool2D(pool_size=(2,2))) 
net_TF.add(tfk.layers.Conv2D(64, (3,3), 
                             activation=activation,
                             kernel_initializer='he_uniform'))
net_TF.add(tfk.layers.MaxPool2D(pool_size=(2,2))) 
net_TF.add(tfk.layers.Flatten())
# =============================================================================
# net_TF.add(tfk.layers.Dense(512, activation=activation,
#                             kernel_initializer='he_uniform'))
# =============================================================================
net_TF.add(tfk.layers.Dense(2048, activation=activation,
                            kernel_initializer='he_uniform'))
net_TF.add(tfk.layers.Dense(10, activation='softmax',
                            kernel_initializer='he_uniform'))

opt = tfk.optimizers.Adam(eta)
net_TF.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
net_TF.summary()
start = time() 
"""
net.train(x, y_train, batch_size=bs, epochs=ep)
"""
t_train = time() - start 

start = time()
net_TF.fit(x_TF, y_train, batch_size=bs, epochs=ep)
t_train_TF = time() - start

y_test = np.argmax(y_test, 1).T

y_tilde_TF = net_TF.predict(x_test.reshape(10000,28,28,1))
guess_TF   = np.argmax(y_tilde_TF, 1).T
print('Accuracy with TensorFlow =', np.sum(guess_TF == y_test)/100)
print('Time needed for training:', t_train_TF)

# Our implementation: 3630s (60,5 min)
# Tensorflow: 206 s (3,4 min)
