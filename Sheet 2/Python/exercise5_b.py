import numpy             as np
import matplotlib.pyplot as plt

from networks    import SequentialNet
from activations import Abs
from layers      import DenseLayer
from optimizers  import SGD

################################################
# Aufgabe 6b)                                  #
################################################

#######################################
# ReLU as activations function        #
#######################################

x = np.expand_dims(np.linspace(0, np.pi/2, 2000), 0)
y = np.concatenate((np.sin(x), np.cos(x)))/2 + .5 

x_train, y_train = x[:,::2], y[:,::2]
x_test, y_test = x[:,1::2], y[:,1::2]

bs, ep, eta = 1, 100, .1

layers=[DenseLayer(1, 100, optim=SGD(eta)),
        DenseLayer(100, 2, optim=SGD(eta))]
netz = SequentialNet(1, layers)
netz.train(x_train, y_train, bs, ep)

ytilde = netz.evaluate(x_test) 

print('error =', np.linalg.norm(y_test - ytilde)) 

plt.title('approximate sine and cosine using ReLU')
plt.plot(x.T, y.T, '-')
plt.plot(x_test.T, ytilde.T, '-.') 
plt.plot(x_test.T, np.log10(np.abs(y_test.T - ytilde.T)), '--')
plt.legend(['sin', 'cos',
            'appr. sin', 'appr. cos',
            'err. sin', 'err cos'])
plt.show()
##########################################
# Abs as activation function             #
##########################################

x = np.expand_dims(np.linspace(0, np.pi/2, 2000), 0)
y = np.concatenate((np.sin(x), np.cos(x)))/2 + .5 

x_train, y_train = x[:,::2], y[:,::2]
x_test, y_test = x[:,1::2], y[:,1::2]

bs, ep, eta = 1, 100, .1

layers=[DenseLayer(1, 100, afun=Abs(), optim=SGD(eta)),
        DenseLayer(100, 2, afun=Abs(), optim=SGD(eta))]
netz = SequentialNet(1, layers)
netz.train(x_train, y_train, bs, ep)

ytilde = netz.evaluate(x_test) 

print('error =', np.linalg.norm(y_test - ytilde)) 

plt.title('approximate sine and cosine using Abs')
plt.plot(x.T, y.T, '-')
plt.plot(x_test.T, ytilde.T, '-.') 
plt.plot(x_test.T, np.log10(np.abs(y_test.T - ytilde.T)), '--')
plt.legend(['sin', 'cos',
            'appr. sin', 'appr. cos',
            'err. sin', 'err cos'])
plt.show()
