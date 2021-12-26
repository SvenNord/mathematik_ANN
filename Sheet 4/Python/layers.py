import numpy as np 

from activations import ReLU
from optimizers  import SGD, Adam
from initializers import RandnAvarage

class DenseLayer:

    def __init__(self,
                 ni, # Number of inputs
                 no, # Number of outputs
                 afun = None, # Activationfunction for the layer
                 optim = None,
                 initializer = None,
                 kernel_regularizer = None
    ):
        
        self.ni   = ni
        self.no   = no

        if afun is None:
            self.afun = ReLU()
        else:
            self.afun = afun 
        
        if optim is None:
            self.optim = SGD() 
        else:
            self.optim = optim

        self.kernel_regularizer = kernel_regularizer
        
        if initializer is None:
          self.initializer = RandnAvarage()
        else:
          self.initializer = initializer
        
        self.W    = self.afun.factor * self.initializer.wfun(ni, no)
        self.b    = np.zeros(no)
        self.__z  = None
        self.__a  = None
        self.dW   = np.zeros_like(self.W)
        self.db   = np.zeros_like(self.b)

    def evaluate(self, a):
        
        self.__a = a
        self.__z = self.__a @ self.W + self.b
        return self.afun.evaluate(self.__z)

    def set_weights(self, W):

        assert(W.shape == (self.ni, self.no))
        self.W = W 

    def set_bias(self, b):
        
        assert(b.size == self.no)
        self.b = b

    def backprop(self, delta):
        

        delta   = self.afun.backprop(delta)
        self.dW = self.__a.T @ delta  
        self.db = np.sum(delta, 0)

        if not self.kernel_regularizer is None:
            self.kernel_regularizer.update(self.W, self.dW)

        return delta @ self.W.T  

    def update(self):

        self.optim.update([self.W, self.b],
                          [self.dW, self.db])

class DropoutLayer:

    def __init__(self, p=.5, trainable=True):

        self.p = p
        self.trainable = trainable
        self.mask = None

    def evaluate(self, a):
        
      if self.trainable:
        
        shape = (1,) + a.shape[1:] # same mask for minibatch
        scale = 1/(1-self.p)
        self.mask = scale * \
            np.random.binomial(1, self.p, size=shape)
        drop_a = a * self.mask
        return drop_a
      else:
        return a

    def backprop(self, delta):

      delta *= self.mask
      return delta

    def update(self):
        pass


class L2Regularizer:

    def __init__(self, l2):

        self.l2 = l2
    
    def update(self, p, dp):

        dp += self.l2 * p 

