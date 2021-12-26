import numpy as np 
from numpy.random import rand, randn
 
class RandnAverage:
  
  def wfun(self, ni, no):
    
    average = (ni + no) / 2.0
    return randn(no, ni) * np.sqrt(1.0/average)
  
class RandAverage:
  
  def wfun(self, ni, no):
    
    average = (ni + no) / 2.0
    return (rand(no, ni) - .5) * np.sqrt(12.0/average)
  
  
class RandnOrthonormal:
  
  def wfun(self, ni, no):
    
    A = randn(no, ni)
    U, _, V = np.linalg.svd(A, full_matrices=False)
    if ni >= no:
      return V
    else:
      return U
                              
