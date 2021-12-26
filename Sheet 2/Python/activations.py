import numpy as np

class ReLU():

    def __init__(self):
        
        self.data = None
        self.name = 'ReLU'

    def wfun(self, m, n):
        return np.random.randn(m,n)*np.sqrt(4/(m + n))

    def evaluate(self, x):

        self.data = x 
        return x.clip(min = 0)

    def backprop(self, delta):

        return (self.data >= 0) * delta

class Abs():

    def __init__(self):

        self.data  = None
        self.name  = 'Abs'

    def wfun(self, m, n):
        return np.random.randn(m,n)*np.sqrt(1/(m + n))

    def evaluate(self, x):

        self.data = x
        return np.abs(x)

    def backprop(self, delta):

        return np.sign(self.data)*delta

    