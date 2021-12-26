import numpy as np
from time import time

def winograd(X, f):
    AT = np.array(((1,  1,  1,  0),
                   (0,  1, -1, -1)))
    B = np.array(((1,  0, -1,  0),
                  (0,  1,  1,  0),
                  (0, -1,  1,  0),
                  (0,  1,  0, -1)))
    C = np.array((( 1,   0,   0),
                  (.5,  .5,  .5),
                  (.5, -.5,  .5),
                  ( 0,   0,   1)))
    return AT @ ((B @ X) * (C @ f[::-1,:]))

def convolve_np(X, f):
    m = X.shape[1]
    Y = np.zeros((2, m))
    for i in range(m):
        Y[:,i] = np.convolve(X[:,i], f[:,0], 'valid')
    return Y

# test correctness
m = 4
X = np.random.randint(0, 10, (4, m))
f = np.random.randint(0, 10, (3, 1))

Y_winograd = winograd(X, f)
Y_numpy = convolve_np(X, f)

print("Winograd:\n", Y_winograd)
print("NumPy:\n", Y_numpy)

# measure time
m = 10000
X = np.random.randint(0, 10, (4, m))
f = np.random.randint(0, 10, (3, 1))

start = time()
Y_winograd = winograd(X, f)
time_winograd = time() - start
print("Winograd in sec =", time_winograd)

start = time()
Y_numpy = convolve_np(X, f)
time_numpy = time() - start
print("NumPy in sec =", time_numpy)

print("Speedup =", time_numpy / time_winograd)
