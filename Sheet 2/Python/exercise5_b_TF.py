import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# training data
x = np.expand_dims(np.linspace(0, np.pi/2, 1000),0)
y = np.concatenate((np.sin(x), np.cos(x)))/2+.5
x_train = x.T
y_train = y.T

bs, ep, eta = 32, 100, .1
h_neurons = 100

#######################################################
# use FNN with one hidden layer comprising 100 neurons
#######################################################
# TF/Keras model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(h_neurons, input_shape=(1,),
                          activation='relu',
                          dtype='float64'),
    tf.keras.layers.Dense(2, activation='relu',
                          dtype='float64')
])

# TF/Keras training
sgd = tf.keras.optimizers.SGD(learning_rate=eta,
                              momentum=0.0, nesterov=False)
model.compile(optimizer=sgd,
              loss='mean_squared_error',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=ep, batch_size=bs)

y_tilde = model.predict(x_train)
print("error =", np.linalg.norm(y_train-y_tilde))
plt.plot(x_train, y_train, '-')
plt.plot(x_train, y_tilde, '-.')
plt.plot(x_train, np.log10(np.abs(y_train-y_tilde)), '--')
plt.legend(['sin', 'cos',
            'appr. sin', 'appr. cos',
            'err. sin', 'err. cos'])
plt.show()
