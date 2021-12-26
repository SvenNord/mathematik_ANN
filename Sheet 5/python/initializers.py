import numpy as np 
from numpy.random import rand, randn
 
"""
Initialization with normal distribution, and standard deviation
depending on the avarage of rows and columns.
The standard deviation depends up to a factor on the activation 
function used. This factor can be applied later.
"""
class RandnAverage:
  
  def wfun(self, m, n):
    
    average = (m + n) / 2.0
    return randn(m, n) / np.sqrt(average)
  
  def ffun(self, m, c, fh, fw):
    
    fan_in  = c * fh * fw 
    fan_out = m * fh * fw 
    average = (fan_in + fan_out) / 2.0
      
    return randn(m, c, fh, fw) / np.sqrt(average)

  
"""
Initialization with uniform distribution on [-r,r] with r 
depending on the avarage of rows and columns.
The radius r depends up to a factor on the activation function
used. This factor can be applied later
"""
class RandAverage:
  
  def wfun(self, m, n):
    
    average = (m + n) / 2.0
    return (rand(m, n) - .5) * np.sqrt(12.0/average)
  
  def ffun(self, m, c, fh, fw):
    
    fan_in  = c * fh * fw 
    fan_out = m * fh * fw
    average = (fan_in + fan_out) / 2.0
    
    return (rand(m, c, fh, fw) - .5) * np.sqrt(12.0/average)
    
class RandnOrthonormal:
  
  def wfun(self, m, n):
    
    A = randn(m, n)
    U, _, V = np.linalg.svd(A, full_matrices=False)
    if n >= m:
      return V
    else:
      return U
                              
