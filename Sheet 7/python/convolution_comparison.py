from time import process_time
from layers import *

# set realistic parameters
#n, c, h, w = 100, 3, 28, 28 # MNIST, im2col wins
#n, c, h, w = 100, 3, 100, 100 # larger images, winograd wins
#n, c, h, w = 2001, 50, 28, 28
n, c, h, w = 20, 50, 1000, 1000 # winograd wins, Scipy faster than fft
m, fh, fw = 32, 3, 3
tensor = (c, h, w)
fshape = (m, fh, fw)
conv = Conv2DLayer(tensor, fshape)

# generate random input
x = np.random.randint(0, 10, (n, c, h, w))

wdh = int(input("how many iterations per call?"))

# test winograd
print('---------------winograd----------------')
start = process_time()
for i in range(wdh):
    print(i)
    y = conv.evaluate_winograd(x)
time_winograd = process_time() - start
print('Time used by winograd:', time_winograd, 'seconds')

# test im2col
print('---------------im2col----------------')
start = process_time()
for i in range(wdh):
    print(i)
    y = conv.evaluate_im2col(x)
time_im2col = process_time() - start
print('Time used by im2col:', time_im2col, 'seconds')

# test
print('---------------fft-----------------')
start = process_time()
for i in range(wdh):
    print(i)
    y = conv.evaluate_fft(x)
time_fft = process_time() - start
print('Time used by fft:', time_fft, 'seconds')

# test
print('---------------scipy-----------------')
start = process_time()
for i in range(wdh):
    print(i)
    y = conv.evaluate_scipy(x)
time_scipy = process_time() - start
print('Time used by scipy:', time_scipy, 'seconds')
