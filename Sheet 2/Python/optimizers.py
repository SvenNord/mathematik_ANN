import numpy as np 

class SGD:

    def __init__(self, eta=.1):
        
        self.eta = eta 
    
    def update(self, data, ddata):

        for p, dp in zip(data, ddata):

            p -= self.eta * dp
                
            