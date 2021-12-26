import numpy as np 
from numpy.random import rand, randn
 
"""
Initialization with normal distribution, and standard deviation
depending on the avarage of rows and columns.
The standard deviation depends up to a factor on the activation 
function used. This factor can be applied later.
"""
class RandnAvarage:
  
  def wfun(self, m, n):
    
    avarage = (m + n) / 2.0
    return randn(m, n) * np.sqrt(1.0/avarage)
  
"""
Initialization with uniform distribution on [-r,r] with r 
depending on the avarage of rows and columns.
The radius r depends up to a factor on the activation function
used. This factor can be applied later
"""
class RandAvarage:
  
  def wfun(self, m, n):
    
    avarage = (m + n) / 2.0
    return (rand(m, n) - .5) * np.sqrt(12.0/avarage)
  
  
class RandnOrthonormal:
  
  def wfun(self, m, n):
    
    A = randn(m, n)
    U, _, V = np.linalg.svd(A, full_matrices=False)
    if n >= m:
      return V
    else:
      return U
                              
