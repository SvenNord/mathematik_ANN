import numpy as np 

from activations import ReLU

class DenseLayer:

    def __init__(self,
                 ni, # Number of inputs
                 no, # Number of outputs
                 afun = None # Activationfunction for the layer
    ):

        self.ni   = ni
        self.no   = no

        # Set default activation function to ReLU
        if afun is None:
            self.afun = ReLU()
        else:
            self.afun = afun 
        
        self.W    = np.zeros((no, ni))
        self.b    = np.zeros((no,1))

    def evaluate(self, a):

        z = self.W @ a + self.b
        return self.afun.evaluate(z)

    def set_weights(self, W):

        assert(W.shape == (self.no, self.ni))
        self.W = W 

    def set_bias(self, b):
        
        assert(b.size == self.no)
        self.b = b


    
