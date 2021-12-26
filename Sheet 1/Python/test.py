import numpy as np 

from networks import SequentialNet
from layers   import DenseLayer

layers = [DenseLayer(2, 4),
          DenseLayer(4, 3)]
netz = SequentialNet(2, layers)
    
netz.layers[0].set_weights(np.array([[-1, -1],
                                     [-1,  1],
                                     [ 1, -1],
                                     [ 1,  1]]))

netz.layers[0].set_bias(np.array([[ 1],
                                  [ 0],
                                  [ 0],
                                  [-1]]))

netz.layers[1].set_weights(np.array([[0, 0, 0, 1],
                                     [0, 1, 1, 1],
                                     [0, 1, 1, 0]]))

X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])
                            
Y = netz.evaluate(X)

netz.draw()

print("input:\n", X)
print("output:\n", Y)
