import numpy as np

# ReLU family activation functions

class ReLU():

    def __init__(self):

        self.name = 'ReLU'

    def evaluate(self, x):

        self.data = x 
        return x.clip(min = 0)
