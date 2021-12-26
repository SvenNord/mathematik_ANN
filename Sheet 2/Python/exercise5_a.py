import sys
import numpy             as np
import matplotlib.pyplot as plt

from networks    import SequentialNet
from activations import Abs
from optimizers  import SGD 
from layers      import DenseLayer

#################################################
# Aufgabe 6a)                                   #
#################################################

x = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])

y = np.array([[0, 0, 0, 1],
              [0, 1, 1, 1],
              [0, 1, 1, 0]])

bs, ep, eta = 1, 5000, .1

######################################
# Tests with ReLU activation funtion #
######################################

layers = [DenseLayer(2, 3, optim=SGD(eta)),
          DenseLayer(3, 3, optim=SGD(eta))]
netz = SequentialNet(layers)
netz.train(x, y, bs, ep)

ytilde = netz.evaluate(x)

print('y =\n', y)
print('ytilde =\n', ytilde)
print('error =', np.linalg.norm(y - ytilde))
print('n = 3, ReLU')

layers = [DenseLayer(2, 4, optim=SGD(eta)),
          DenseLayer(4, 3, optim=SGD(eta))]
netz = SequentialNet(2, layers)
netz.train(x, y, bs, ep)

ytilde = netz.evaluate(x)

print('y =\n', y)
print('ytilde =\n', ytilde)
print('error =', np.linalg.norm(y - ytilde))
print('n = 4, ReLU')

layers = [DenseLayer(2, 10, optim=SGD(eta)),
          DenseLayer(10, 3, optim=SGD(eta))]
netz = SequentialNet(2, layers)
netz.train(x, y, bs, ep)

ytilde = netz.evaluate(x)

print('y =\n', y)
print('ytilde =\n', ytilde)
print('error =', np.linalg.norm(y - ytilde))
print('n = 10, ReLU')

layers = [DenseLayer(2, 100, optim=SGD(eta)),
          DenseLayer(100, 3, optim=SGD(eta))]
netz = SequentialNet(2, layers)

print('y =\n', y)
print('ytilde =\n', ytilde)
print('error =', np.linalg.norm(y - ytilde))
print('n = 100, ReLU')

######################################
# Tests with Abs activation funtion #
######################################

layers = [DenseLayer(2, 3, afun=Abs(), optim=SGD(eta)),
          DenseLayer(3, 3, afun=Abs(), optim=SGD(eta))]
netz = SequentialNet(2, layers)
netz.train(x, y, bs, ep)


ytilde = netz.evaluate(x)

print('y =\n', y)
print('ytilde =\n', ytilde)
print('error =', np.linalg.norm(y - ytilde))
print('n = 3, Abs')

layers = [DenseLayer(2, 4, afun=Abs(), optim=SGD(eta)),
          DenseLayer(4, 3, afun=Abs(), optim=SGD(eta))]
netz = SequentialNet(2, layers)
netz.train(x, y, bs, ep)

ytilde = netz.evaluate(x)

print('y =\n', y)
print('ytilde =\n', ytilde)
print('error =', np.linalg.norm(y - ytilde))
print('n = 4, Abs')

layers = [DenseLayer(2, 10, afun=Abs(), optim=SGD(eta)),
          DenseLayer(10, 3, afun=Abs(), optim=SGD(eta))]
netz = SequentialNet(2, layers)
netz.train(x, y, bs, ep)

ytilde = netz.evaluate(x)

print('y =\n', y)
print('ytilde =\n', ytilde)
print('error =', np.linalg.norm(y - ytilde))
print('n = 10, Abs')

layers = [DenseLayer(2, 100, afun=Abs(), optim=SGD(eta)),
          DenseLayer(100, 3, afun=Abs(), optim=SGD(eta))]
netz = SequentialNet(2, layers)
netz.train(x, y, bs, ep)

ytilde = netz.evaluate(x)

print('y =\n', y)
print('ytilde =\n', ytilde)
print('error =', np.linalg.norm(y - ytilde))
print('n = 100, Abs')


