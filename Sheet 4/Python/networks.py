import numpy             as np 
import matplotlib.pyplot as plt


from activations import ReLU
from layers      import DenseLayer 
from optimizers  import SGD

class SequentialNet:

  def __init__(self, n, layers=None):
        
    if layers is None:
      self.layers = [] 
      self.no = n
    else:
      self.layers = layers
      self.no = layers[-1].no 

  def evaluate(self, x):

    for layer in self.layers: 
            
      x = layer.evaluate(x) 
        
    return x

  def add_dense(self, n, afun=None, optim=None):

    self.layers.append(DenseLayer(self.no, n, afun, optim))
    self.no = n

  def backprop(self, x, y):
        
    delta = (self.evaluate(x) - y)/y.shape[1] 

    for layer in reversed(self.layers):
    
      delta = layer.backprop(delta)

  def train(self, x, y, batch_size=16, epochs=10):

    n_data    = y.shape[0]
    n_batches = int(np.ceil(n_data/batch_size)) 

    for e in range(epochs):

      p = np.random.permutation(n_data)
      for j in range(n_batches):
                
        self.backprop(x[p[j*batch_size:(j+1)*batch_size],:], 
                      y[p[j*batch_size:(j+1)*batch_size],:])
                
        for layer in self.layers:
          layer.update()

  def draw(self):
    
    num_layers = len(self.layers) + 1
    max_neurons_per_layer = np.amax([layer.no for layer in self.layers])
    neurons_layers = np.array([self.layers[0].ni] + [layer.no for layer in self.layers])
    dist = 2*max(1,max_neurons_per_layer/num_layers)
    y_shift = neurons_layers/2-.5
    rad = .3

    fig = plt.figure(frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    for i in range(num_layers):
      for j in range(neurons_layers[i]):
        circle = plt.Circle((i*dist, j-y_shift[i]),
                            radius=rad, fill=False)
        ax.add_patch(circle)

    for i in range(num_layers-1):
      for j in range(neurons_layers[i]):
        for k in range(neurons_layers[i+1]):
          angle = np.arctan(float(j-k+y_shift[i+1]-y_shift[i]) / dist)
          x_adjust = rad * np.cos(angle)
          y_adjust = rad * np.sin(angle)
          line = plt.Line2D((i*dist+x_adjust,
                            (i+1)*dist-x_adjust),
                            (j-y_shift[i]-y_adjust,
                            k-y_shift[i+1]+y_adjust),
                            lw=2 / np.sqrt(neurons_layers[i]
                                         + neurons_layers[i+1]),
                            color='b')
          ax.add_line(line)

    ax.axis('scaled')
    plt.show()
