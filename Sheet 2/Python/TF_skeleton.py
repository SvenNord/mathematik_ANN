import numpy             as np 
import tensorflow.keras  as tk
import matplotlib.pyplot as plt 

from random import randrange 

####################################
# Exercise 5, part a)              #
####################################
"""
TODO Create data as provided in Exercise 1b) on sheet 1 
"""
x = None 
y = None
"""
TODO Define batch sizes, number of epochs and learning rate 
"""
bs, ep, eta = None 

"""
TODO Create a list of layers for the network. Use tk.layers.Dense(args)
     (See https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
     for more informations). Compare different numbers of neurons in the 
     hidden layer. 
"""
layers = [] 

"""
TODO Set up a feedforward neural network. Use tk.Sequential(args)
     (See https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
     for more informations).
"""
net = None 

"""
TODO Set up an SGD optimizer for training. Use tk.optimizers.SGD(args)
     (See https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD
     for more informations)
"""
sgd = None 

# Set up optimizer and loss function for training 
net.compile(optimizer=sgd,
            loss='mean_squared_error',
            metrics=['accuracy'])

# Train the network with net.fit()
net.fit(x, y, batch_size=bs, epochs=ep)

# Test network with trained data
y_tilde = net.predict(x)
print("y_train =", y)
print("y_tilde =", y_tilde)
print("error =", np.linalg.norm(y-y_tilde))

####################################
# Exercise 5, part b)              #
####################################

"""
TODO Create training data and set up network as above with 
     100 neurons in the hidden layer
"""
x = None 
y = None 

bs, ep, eta = None 

layers = []

net = None 

sgd = None 

# Set up optimizer and loss function for training 
net.compile(optimizer=sgd,
            loss='mean_squared_error',
            metrics=['accuracy'])

# Train the network with net.fit()
net.fit(x, y, batch_size=bs, epochs=ep)

# Test network
y_tilde = net.predict(x)
print("error =", np.linalg.norm(y-y_tilde))
plt.plot(x, y, '-')
plt.plot(x, y_tilde, '-.')
plt.plot(x, np.log10(np.abs(y-y_tilde)), '--')
plt.legend(['sin', 'cos',
            'appr. sin', 'appr. cos',
            'err. sin', 'err. cos'])
plt.show()

####################################
# Exercise 5, part c)              #
####################################

"""
TODO Set up data and create a network as above
"""

x_train, y_train = None 
x_test, y_test = None 

# Reshape data as needed
x = x_train.reshape(60000, 784)
I = np.eye(10)
y = I[y_train,:]

bs, ep, eta = None

layers = [] 

net = None 

sgd = None 

# Set up optimizer and loss function for training 
net.compile(optimizer=sgd,
            loss='mean_squared_error',
            metrics=['accuracy'])

# Train the network with net.fit()
net.fit(x, y, batch_size=bs, epochs=ep)

# Test network
y_tilde = net.predict(x_test.reshape(10000, 784))
guess   = np.argmax(y_tilde, 1)
print('accuracy =', np.sum(guess == y_test)/100)

for i in range(4):

    k = randrange(y_test.size)
    plt.title('Label is {lb}, guess is {gs}'.format(lb=y_test[k], gs=guess[k]))
    plt.imshow(x_test[k], cmap='gray')
    plt.show()
