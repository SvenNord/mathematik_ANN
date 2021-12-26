import numpy             as np 
from   random            import randrange
import matplotlib.pyplot as plt

from networks    import SequentialNet 
from layers      import DenseLayer 
from activations import * 
from optimizers  import *

DATA = np.load('mnist.npz')
x_train, y_train = DATA['x_train'], DATA['y_train'] 
x_test, y_test   = DATA['x_test'], DATA['y_test']
x_train, x_test = x_train / 255.0, x_test / 255.0

x = x_train.reshape(60000, 784).T
I = np.eye(10)
y = I[:, y_train]

bs, ep, eta = 10, 10, .01

functions1 = [Logistic(), TanH(), ReLU(), leakyReLU(), 
              ELU(), SoftPlus(), Abs(), LOCo(), Exp()]
functions2 = [Logistic(), TanH(), ReLU(), leakyReLU(), 
              ELU(), SoftPlus(), Abs(), LOCo(), Exp()]

for fun1, fun2 in zip(functions1, functions2):

    print('Activation function =', fun1.name)
    layers = [DenseLayer(784, 100, afun=fun1, optim=SGD(eta)),
              DenseLayer(100, 10 , afun=fun2, optim=SGD(eta))]
    netz = SequentialNet(784, layers)
    netz.train(x, y, bs, ep)

    y_tilde = netz.evaluate(x_test.reshape(10000, 784, 1))
    guess   = np.argmax(y_tilde, 1).T
    print('accuracy =', np.sum(guess == y_test)/100)

    for i in range(4):

        k = randrange(y_test.size)
        plt.title('Label is {lb}, guess is {gs}'.format(lb=y_test[k], gs=guess[0,k]))
        plt.imshow(x_test[k], cmap='gray')
        plt.show()
