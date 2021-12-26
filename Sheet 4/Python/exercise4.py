import numpy             as np 
from   random            import randrange
import matplotlib.pyplot as plt   

from networks    import SequentialNet
from layers      import *
from optimizers  import *
from activations import *

DATA = np.load('mnist.npz')
#DATA = np.load('fashion_mnist.npz')
x_train, y_train = DATA['x_train'], DATA['y_train'] 
x_test, y_test   = DATA['x_test'], DATA['y_test']
x_train, x_test = x_train / 255.0, x_test / 255.0

categories = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

x = x_train.reshape(60000, 784)
I = np.eye(10)
y = I[y_train,:]

bs, ep, eta = 1000, 10, .001

print('Ohne Dropout')
layers = [DenseLayer(784, 100, afun=Logistic(), optim=SGD(eta)),
          DenseLayer(100, 10 , afun=Logistic(), optim=SGD(eta))]
netz = SequentialNet(784,layers)
netz.train(x, y, bs, ep)

y_tilde = netz.evaluate(x_test.reshape(10000,784))
guess   = np.argmax(y_tilde, 1).T
print('accuracy =', np.sum(guess == y_test)/100)

for i in range(4):

    k = randrange(y_test.size)
    plt.title('Label is {lb}, guess is {gs}'.format(lb=y_test[k], gs=guess[k]))
    plt.imshow(x_test[k], cmap='gray')
    plt.show()

prop = np.linspace(.5, .8, 7, True)
for p in prop:
    print('Mit Dropout, p =', p)
    layers = [DenseLayer(784, 100, afun=Logistic(), optim=SGD(eta*10)),
              DropoutLayer(p),
              DenseLayer(100, 10, afun=Logistic(), optim=SGD(eta*10))]
    netz = SequentialNet(784, layers)
    netz.train(x, y, bs, ep)

    y_tilde = netz.evaluate(x_test.reshape(10000,784))
    guess   = np.argmax(y_tilde, 1)
    print('accuracy =', np.sum(guess == y_test)/100)

    for i in range(4):

        k = randrange(y_test.size)
        plt.title('Label is {lb}, guess is {gs}'.format(lb=y_test[k], gs=guess[k]))
        plt.imshow(x_test[k], cmap='gray')
        plt.show()
