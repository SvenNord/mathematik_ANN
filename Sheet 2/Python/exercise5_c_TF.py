import numpy             as np 
import tensorflow        as tf
import matplotlib.pyplot as plt
from   random            import randrange

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
x_train, x_test = x_train / 255.0, x_test / 255.0

x = x_train.reshape(60000, 784)
I = np.eye(10)
y = I[y_train,:]

bs, ep, eta = 10, 10, .05

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, input_shape=(784,),
                          activation='relu',
                          dtype='float64'),
    tf.keras.layers.Dense(10, activation='relu',dtype='float64')
])

sgd = tf.keras.optimizers.SGD(learning_rate=eta, momentum=0.0, nesterov=False)

model.compile(optimizer=sgd, loss='mean_squared_error',metrics=['accuracy'])
model.fit(x, y, epochs=ep, batch_size=bs)

y_tilde = model.predict(x_test.reshape(10000, 784))
guess   = np.argmax(y_tilde, 1)
print('accuracy =', np.sum(guess == y_test)/100)

for i in range(4):

    k = randrange(y_test.size)
    plt.title('Label is {lb}, guess is {gs}'.format(lb=y_test[k], gs=guess[k]))
    plt.imshow(x_test[k], cmap='gray')
    plt.show()
