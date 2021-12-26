import numpy             as np 
import matplotlib.pyplot as plt
import tensorflow.keras  as tfk

from   random            import randrange

from networks    import SequentialNet
from layers      import *
from optimizers  import *
from activations import *

DATA = np.load('mnist.npz')
x_train, y_train = DATA['x_train'], DATA['y_train'] 
x_test, y_test   = DATA['x_test'], DATA['y_test']
x_train, x_test = x_train / 255.0, x_test / 255.0

bs, ep, eta = 1000, 10, .001

x = x_train[:,:,:,np.newaxis]
I = np.eye(10)
y = I[y_train,:]

layers = [tfk.layers.Conv2D(32, (3, 3), 
                            input_shape=(28,28,1),
                            activation='relu'),
          tfk.layers.MaxPool2D((2,2)),
          tfk.layers.Flatten(),
          tfk.layers.Dense(100,
                           activation='relu'),
          tfk.layers.Dense(10,
                           activation='softmax')]
          
net = tfk.Sequential(layers)

opt = tfk.optimizers.Adam(learning_rate=eta)
net.compile(optimizer=opt, 
            loss='categorical_crossentropy', 
            metrics=['accuracy'])

net.fit(x, y, batch_size=bs, epochs=ep)

y_tilde = net.predict(x_test[:,:,:,np.newaxis])
guess   = np.argmax(y_tilde, 1).T
print('accuracy =', np.sum(guess == y_test)/100)

for i in range(4):

    k = randrange(y_test.size)
    plt.title('Label is {lb}, guess is {gs}'.format(lb=y_test[k], gs=guess[k]))
    plt.imshow(x_test[k], cmap='gray')
    plt.show()
