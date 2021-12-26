import numpy as np

# Heaviside family activation functions
class HeavisideLike():

  def __init__(self):
    
    self.factor = 1.0

class Heaviside(HeavisideLike):

    def __init__(self):
      
        super().__init__()
        self.data = None 
        self.name = 'Heavyside'

    def evaluate(self, x):

        return (x >= 0) * 1.0 

    def backprop(self, delta):

        return 0.0*delta

class ModifiedHeaviside(HeavisideLike):

    def __init__(self): 
      
        super().__init__()
        self.data = None
        self.name = 'Modified Heaviside'

    def evaluate(self, x):

        return (x > 0) * 1.0 + (x == 0) * 0.5

    def backprop(self, delta):

        return 0.0 * delta

class Logistic(HeavisideLike):

    def __init__(self, k=1.0):
        
        super().__init__()
        self.data = None 
        self.name = 'Logistic'
        self.k    = k

    def evaluate(self, x):

        self.data = 1.0/(1.0 + np.exp(-self.k * x))
        return self.data 

    def backprop(self, delta):

        return self.k * self.data * (1.0 - self.data) * delta 

class SoftMax(HeavisideLike):

    def __init__(self):
      
        super().__init__()
        self.name = 'SoftMax'
        self.data = None # store data for faster execution

    def evaluate(self, x):
        x_exp = np.exp(x)
        scale = x_exp.sum(axis=1, keepdims=True)
        return x_exp / scale

    def backprop(self, delta):
        return delta
        
class Exp(HeavisideLike):

    def __init__(self):
        
        super().__init__()
        self.data = None
        self.name = 'Exp'

    def evaluate(self, x):

        self.data = np.exp(x)
        return self.data 

    def backprop(self, delta):

        return self.data * delta
        
# Sign family activation functions
class SignLike:

    def __init__(self):
      
      self.factor = 1.0

class Sign(SignLike):

    def __init__(self):
        
        super().__init__()
        self.data = None 
        self.name = 'Sign'

    def evaluate(self, x):

        return np.sign(x)

    def backprop(self, delta):

        return 0.0 * delta

class TanH(SignLike):

    def __init__(self, k=1.0):
        
        super().__init__()
        self.data = None
        self.name = 'TanH'
        self.k    = k

    def evaluate(self, x):

        self.data = np.tanh(self.k * x)
        
        return self.data

    def backprop(self, delta):

        return self.k * (1.0 - self.data**2) * delta

class SoftSign(SignLike):

    def __init__(self, k=1.0):
        
        super().__init__()
        self.data = None 
        self.name = 'SoftSign'
        self.k    = k 

    def evaluate(self, x):

        self.data = (self.k * x)/(1.0 + np.abs(self.k * x))
        return self.data 

    def backprop(self, delta):

        return self.k/(self.data**2) * delta

# ReLU family activation functions
class ReLULike:

    def __init__(self):
      
      self.factor = np.sqrt(2)

class ReLU(ReLULike):

    def __init__(self):
        
        super().__init__()
        self.data = None
        self.name = 'ReLU'

    def evaluate(self, x):

        self.data = x 
        return x.clip(min = 0)

    def backprop(self, delta):

        return (self.data >= 0) * delta

class leakyReLU(ReLULike):

    def __init__(self, alpha=.01):
        
        super().__init__()
        self.data  = None 
        self.name  = 'leaky ReLU'
        self.alpha = alpha 

    def evaluate(self, x):

        self.data = x
        return x.clip(min=0) + self.alpha*x.clip(max=0) 

    def backprop(self, delta):

        return   (self.data >= 0) * delta \
               + (self.data < 0) * self.alpha * delta

class ELU(ReLULike):

    def __init__(self, alpha=1.0):
        
        super().__init__()
        self.data  = None 
        self.name  = 'ELU'
        self.alpha = alpha

    def evaluate(self, x):

        self.data = x 
        return x.clip(min=0) + \
               (np.exp(x.clip(max=0)) - 1.0) * self.alpha

    def backprop(self, delta):

        return   (self.data >= 0) * delta \
               + (self.data < 0) * self.alpha * np.exp(self.data) * delta

class SoftPlus(ReLULike):

    def __init__(self, k=1.0):
        
        super().__init__()
        self.data = None 
        self.name = 'SoftPlus'
        self.k    = k

    def evaluate(self, x):

        self.data = np.log(np.exp(self.k * x) + 1.0)/self.k 
        return self.data 

    def backprop(self, delta):

        return 1.0 / (1.0 + np.exp(-self.data)) * delta

class Swish(ReLULike):

    def __init__(self, k=1.0):
        
        super().__init__()
        self.data = None
        self.name = 'Swish'
        self.k    = k 

    def evaluate(self, x):

        self.data = self.k * x
        y = np.exp(self.data) 
        return x * y/(1.0 + y) 

    def backprop(self, delta):

        y = np.exp(self.data)/(1.0 + np.exp(self.data)) 
        return y * (1.0 + self.data * (1.0 - y)) * delta

# Abs family activation functions
class AbsLike:

    def __init__(self):
      
      self.factor = 1.0

class Abs(AbsLike):

    def __init__(self, alpha=0.0):
        
        super().__init__()
        self.data  = None
        self.name  = 'Abs'
        self.alpha = alpha

    def evaluate(self, x):

        self.data = x
        if(self.alpha == 0.0):
            return np.abs(x)
        else:
            return np.sqrt(x*x + self.alpha**2) - self.alpha

    def backprop(self, delta):

        if(self.alpha == 0.0):
            return np.sign(self.data)*delta
        else: 
            return self.data/np.sqrt(self.data**2 + self.alpha**2) * delta
    
class LOCo(AbsLike):

    def __init__(self, k=1.0):
        
        super().__init__()
        self.data = None 
        self.name = 'LOCo'
        self.k    = k

    def evaluate(self, x):

        self.data = self.k * x
        return np.log(np.cosh(self.data))/self.k

    def backprop(self,delta):

        return np.tanh(self.data) * delta
    
class Twist(AbsLike):

    def __init__(self, k=1.0):
        
        super().__init__()
        self.data = None 
        self.name = 'Twist'
        self.k    = k 

    def evaluate(self, x):

        self.data = self.k * x
        return x * np.tanh(self.data) 

    def backprop(self, delta):

        y = np.tanh(self.data)
        return y * (1.0 + self.data * (1 - y * y)) * delta

class SoftAbs(AbsLike):

    def __init__(self, k=1.0):

        super().__init__()
        self.data = None 
        self.name = 'SoftAbs'
        self.k    = k 

    def evaluate(self, x):

        self.data = self.k * x
        return self.data * x / (1.0 + np.abs(self.data))

    def backprop(self, delta):
        
        y = 1.0 + np.abs(self.data)
        return (1.0 + y)/(y**2) * self.data * delta

    

