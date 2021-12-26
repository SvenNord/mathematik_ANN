import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from time import sleep

# training data for exercise 1b), sheet 1
x_train = np.array( [[0,0], [0,1], [1,0], [1,1]] )
y_train = np.array( [[0,0,0],  # xor, and, or
                     [1,0,1],
                     [1,0,1],
                     [0,1,1]])

# overall parameters
bs, ep, eta = 1, 1000, .1
sleep_time = 2

###########################################################
# use minimal net for exercise 1b), sheet 1 as example
###########################################################
# TF/Keras model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(3, input_shape=(2,),
                          activation='relu',
                          dtype='float64'),
    tf.keras.layers.Dense(3, activation='relu',
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
print("y_train =", y_train)
print("y_tilde =", y_tilde)
print("error =", np.linalg.norm(y_train-y_tilde))
print("n = 3")
sleep(sleep_time)

###########################################################
# use net of exercise 1b), sheet 1 as example
###########################################################
# TF/Keras model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(4, input_shape=(2,),
                          activation='relu',
                          dtype='float64'),
    tf.keras.layers.Dense(3, activation='relu',
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
print("y_train =", y_train)
print("y_tilde =", y_tilde)
print("error =", np.linalg.norm(y_train-y_tilde))
print("n = 4")
sleep(sleep_time)

###########################################################
# use 10 neurons in hidden layer for exercise 1b), sheet 1
###########################################################
# TF/Keras model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=(2,),
                          activation='relu',
                          dtype='float64'),
    tf.keras.layers.Dense(3, activation='relu',
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
print("y_train =", y_train)
print("y_tilde =", y_tilde)
print("error =", np.linalg.norm(y_train-y_tilde))
print("n = 10")
sleep(sleep_time)

###########################################################
# use 100 neurons in hidden layer for exercise 1b), sheet 1
###########################################################
# TF/Keras model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, input_shape=(2,),
                          activation='relu',
                          dtype='float64'),
    tf.keras.layers.Dense(3, activation='relu',
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
print("y_train =", y_train)
print("y_tilde =", y_tilde)
print("error =", np.linalg.norm(y_train-y_tilde))
print("n = 100")
