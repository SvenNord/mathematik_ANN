import numpy as np 

class SGD:

    def __init__(self, eta=.01, momentum=0.0):
        
        self.eta = eta 
        self.momentum = momentum
        if self.momentum > 0.0:
            self.v = []
            self.name = 'SGD({}) with momentum {}'.format(eta, momentum)
        else:
            self.name = 'SGD({})'.format(eta)

    def update(self, data, ddata):
        
        if self.momentum > 0.0:
            
            if self.v == []:
                self.v = [np.zeros_like(p) for p in data]
                
            for v, p, dp in zip(self.v, data, ddata):
                
                v  = v * self.momentum + dp * self.eta
                p -= v 
                
        else:

       	  for p, dp in zip(data, ddata):
                
            p -= self.eta * dp 
                    
                
            
class Adam:
    
    def __init__(self, eta =.001, beta1 =.9, beta2 =.999, eps=1e-8):
    
        self.eta   = eta 
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps 
        
        self.w = [] 
        self.v = []
        self.k = 0
        
        self.name = 'Adam'

    def update(self, data, ddata):

        if self.v == []:
            self.v = [np.zeros_like(p) for p in data]
        if self.w == []:
            self.w = [np.zeros_like(p) for p in data]
        
        self.k += 1 
        alpha   = self.eta*np.sqrt(1 - self.beta2**self.k)/(1 - self.beta1**self.k)
        for v, w, p, dp in zip(self.v, self.w, data, ddata):
            
            v = self.beta1*v + (1 - self.beta1)*dp
            w = self.beta2*w + (1 - self.beta2)*dp**2 

            p -= v*alpha/np.sqrt(w + self.eps)
            
    
