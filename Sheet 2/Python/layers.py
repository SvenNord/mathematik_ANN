import numpy as np 

from activations import ReLU
from optimizers  import SGD

class DenseLayer:

    def __init__(self,
                 ni, # Number of inputs
                 no, # Number of outputs
                 afun = None, # Activationfunction for the layer
                 optim = None
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
        
        self.W    = self.afun.wfun(no, ni)
        self.b    = np.zeros((no,1))
        self.__z  = None
        self.__a  = None
        self.dW   = np.zeros_like(self.W)
        self.db   = np.zeros_like(self.b)

    def evaluate(self, a):

        self.__a = a
        self.__z = self.W @ self.__a + self.b
        return self.afun.evaluate(self.__z)

    def set_weights(self, W):

        assert(W.shape == (self.no, self.ni))
        self.W = W 

    def set_bias(self, b):
        
        assert(b.size == self.no)
        self.b = b

    def backprop(self, delta):

        delta   = self.afun.backprop(delta)
        self.dW = delta @ self.__a.T 
        self.db = delta @ np.ones((delta.shape[1],1))

        return self.W.T @ delta

    def update(self):

        self.optim.update([self.W, self.b],
                          [self.dW, self.db])
