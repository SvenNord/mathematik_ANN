"""
quo computes for given f,g the quotient q with 
f = q*g + r for some rest term r

degree returns the degree of a given polynomial 

expand returns a polynomial in standard representation 
expand(f) = \sum_{i = 0}^n f_i x^i
"""
from sympy     import quo, degree, expand
from sympy.abc import u

def extended_euclidean_algorithm(a, b):
  
  r_old, r = a, b 
  c_old, c = 1, 0 
  d_old, d = 0, 1 
  
  while degree(r) > 0:
    
    q = quo(r_old, r)
    r_old, r = r, expand(r_old - q*r)
    c_old, c = c, expand(c_old - q*c)
    d_old, d = d, expand(d_old - q*d)
    
  return r_old, c_old, d_old

if __name__ == '__main__':
  
  a = u**4 + 3*u**3 - 8*u**2 - 12*u + 16 
  b = u**4 + u**3 - 7*u**2 - u + 6 

  r, c, d = extended_euclidean_algorithm(a, b)
  print('GCD =', r)
  print('c =', c)
  print('d =', d) 
  print('Error in Bezout =', expand(a*c + b*d - r))
