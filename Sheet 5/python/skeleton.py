import numpy             as np 
import matplotlib.pyplot as plt

from   random            import randrange
# If you use TF:
# import tensorflow.keras as tfk  

from networks    import SequentialNet
from layers      import *
from optimizers  import *
from activations import *

DATA = np.load('mnist.npz')
x_train, y_train = DATA['x_train'], DATA['y_train'] 
x_test, y_test   = DATA['x_test'], DATA['y_test']
x_train, x_test = x_train / 255.0, x_test / 255.0

"""
TODO Implement the network you have developed for exercise 4
     Note that x_train and x_test are of shape (60000,28,28) 
     and (10000,28,28). You need to add an additional axis. 
     This can be done with np.newaxis, e.g 
     x_test = x_test[:,np.newaxis,:,:]
"""
netz=None

y_tilde = netz.evaluate(x_test)
guess   = np.argmax(y_tilde, 1).T
print('accuracy =', np.sum(guess == y_test)/100)

for i in range(4):

    k = randrange(y_test.size)
    plt.title('Label is {lb}, guess is {gs}'.format(lb=y_test[k], gs=guess[k]))
    plt.imshow(x_test[k], cmap='gray')
    plt.show()
