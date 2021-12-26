import numpy      as np 

from scipy.signal  import convolve2d
from numpy.fft     import rfft2, irfft2

from activations  import ReLU, Abs
from optimizers   import SGD
from initializers import RandnAverage
from utils        import im2col, col2im

strided = np.lib.stride_tricks.as_strided

class DenseLayer:

    def __init__(self,
                 ni, # Number of inputs
                 no, # Number of outputs
                 afun = None, # Activationfunction for the layer
                 optim = None,
                 initializer = None,
                 kernel_regularizer = None
    ):
        
        self.ni   = ni
        self.no   = no

        if afun is None:
            self.afun = ReLU()
        else:
            self.afun = afun 
        
        if optim is None:
            self.optim = SGD() 
        else:
            self.optim = optim

        self.kernel_regularizer = kernel_regularizer
        
        if initializer is None:
          self.initializer = RandnAverage()
        else:
          self.initializer = initializer
        
        self.W    = self.afun.factor * self.initializer.wfun(ni, no)
        self.b    = np.zeros(no)
        self.__z  = None
        self.__a  = None
        self.dW   = np.zeros_like(self.W)
        self.db   = np.zeros_like(self.b)

    def evaluate(self, a):
        
        self.__a = a
        self.__z = self.__a @ self.W + self.b
        return self.afun.evaluate(self.__z)

    def set_weights(self, W):

        assert(W.shape == (self.ni, self.no))
        self.W = W 

    def set_bias(self, b):
        
        assert(b.size == self.no)
        self.b = b

    def backprop(self, delta):
        

        delta   = self.afun.backprop(delta)
        self.dW = self.__a.T @ delta  
        self.db = np.sum(delta, 0)

        if not self.kernel_regularizer is None:
            self.kernel_regularizer.update(self.W, self.dW)

        return delta @ self.W.T  

    def update(self):

        self.optim.update([self.W, self.b],
                          [self.dW, self.db])

class DropoutLayer:

    def __init__(self, p=.5, trainable=True):

        self.p = p
        self.trainable = trainable
        self.mask = None

    def evaluate(self, a):
        
      if self.trainable:
        
        shape = (1,) + a.shape[1:] # same mask for minibatch

        self.mask = 1/self.p *np.random.binomial(1, self.p, size=shape)
        drop_a = a * self.mask
        return drop_a
      else:
        return a

    def backprop(self, delta):

      delta *= self.mask
      return delta

    def update(self):
        pass


class L2Regularizer:

    def __init__(self, l2):

        self.l2 = l2
    
    def update(self, p, dp):

        dp += self.l2 * p 

class Conv2DLayer:

    def __init__(self, tensor, fshape, afun=None, optim=None, initializer=None, eval_method='scipy'):

        if afun is None:
            self.afun = ReLU() 
        else:
            self.afun = afun 

        if optim is None:
            self.optim = SGD()
        else:
            self.optim = optim
      
        if initializer is None:
          self.initializer = RandnAverage()
        else:
          self.initializer = initializer
          
        self.tensor = tensor 
        self.fshape = fshape 
        
        m, fh, fw = fshape 
        
        c, h, w   = tensor 
        self.f = self.initializer.ffun(m, c, fh, fw) 
        self.f *= self.afun.factor
        self.b = np.zeros(m)

        self.__a = None 
        self.__z = None

        self.df = np.zeros_like(self.f)
        self.db = np.zeros_like(self.b)
        
        # Cache for convolutions with im2col
        self.cache = None
        
        if eval_method == 'scipy':
          
          self.evaluate = self.evaluate_scipy
          self.backprop = self.backprop_scipy 
        
        elif eval_method == 'fft':
          
          self.evaluate = self.evaluate_fft 
          self.backprop = self.backprop_fft 
          
        elif eval_method == 'im2col':
          
          self.evaluate = self.evaluate_im2col
          self.backprop = self.backprop_im2col
        
        else:
          
          print('Evaluation method not found!')

    def evaluate_scipy(self, a):
        
        n, c, h, w = a.shape
        m, fh, fw  = self.fshape  

        self.__a = a 

        self.__z = np.zeros((n, m, h - fh + 1, w - fw + 1))
        for j in range(m):
            for i in range(n):
                self.__z[i,j,:,:] += self.b[j] 
                for k in range(c):
                    self.__z[i,j,:,:] += convolve2d(self.__a[i,k,:,:],\
                                                    self.f[j,k,:,:], 
                                                    mode='valid')

        return self.afun.evaluate(self.__z)
    
    def evaluate_fft(self, a):
      
      self.__a = a
      
      n, c, h, w = a.shape 
      m, fh, fw  = self.fshape
      dh, dw     = fh - 1, fw - 1
      zh, zw     = h - dh, w - dw
      
      # Apply FFT 
      a_hat = rfft2(a, (h + dh, w + dw)) 
      f_hat = rfft2(self.f, (h + dh, w + dw))
      
      self.__z = np.zeros((n, m, zh, zw)) 
      for i in range(n):
        for j in range(m):
          
          """
          Compute the full convolution with applying 
          the componentwise product of a_hat and f_hat.
          z_hat then has the shape (c, h+dh, w+dw) and has 
          to be restricted and summed over all channels.
          """
          z_hat = irfft2(a_hat[i,:,:,:] * \
                         f_hat[j,:,:,:],
                         (h + dh, w + dw)) 
          self.__z[i,j,:,:] = self.b[j] + \
            z_hat[:, dh:-dh, dw:-dw].sum(axis=0) 
          
      return self.afun.evaluate(self.__z)
    
    def evaluate_im2col(self, a):
      
        self.__a = a
        
        n, c, h, w = a.shape
        m, fh, fw = self.fshape
        zh, zw = h - fh + 1, w - fw + 1
        
        # Create im2col matrix
        a_col = im2col(a, fh, fw)
        # reshape filterbank
        f_row = self.f[:,:,::-1,::-1].reshape(m, -1)
        self.cache = a_col, f_row
        
        # BLAS Level 3: GEMM
        z_mat = f_row @ a_col + self.b.reshape(-1, 1)
        
        # reshape
        self.__z = z_mat.reshape(m, n, zh, zw).transpose(1, 0, 2, 3)
        return self.afun.evaluate(self.__z)
    
    def evaluate_winograd(self, a):
        # third implementation, slower than im2col for small images
        self.__a = a
        n, c, h, w = a.shape # need h > 3 and w > 3
        m, fh, fw = self.fshape # only for fh == fw == 3!
        zh, zw = h - fh + 1, w - fw + 1
        AT = np.array([[1, 1,  1,  0],
                       [0, 1, -1, -1]])
        B = np.array([[1,  0, -1,  0],
                      [0,  1,  1,  0],
                      [0, -1,  1,  0],
                      [0,  1,  0, -1]])
        C = np.array([[ 0,   0,  1], # flipped, no need to flip filter
                      [.5,  .5, .5],
                      [.5, -.5, .5],
                      [ 1,   0,  0]])
        hr, wr = h % 2, w % 2 # treat borders
        ph, pw = 4, 4 # patch size
        yh, yw = (h + hr - ph) // 2 + 1, (w + wr - pw) // 2 + 1 # stride 2
        a_pad = np.pad(a, ((0, 0), (0, 0), (0, hr), (0, wr)),
                       mode='constant')
        ns, cs, hs , ws = a_pad.strides
        a_patches = strided(a_pad,
                            (n, c, yh, yw, ph, pw),
                            (ns, cs, 2*hs, 2*ws, hs, ws))
        a_winograd = B @ a_patches @ B.T
        f_winograd = C @ self.f @ C.T
        #path = True # if nothing is known / gives 'greedy'
        ############## to compute the optimal path:
        #path, path_info = np.einsum_path('ik,ncxykl,mckl,jl->nmxiyj',
        #                                 AT, a_winograd,
        #                                 f_winograd, AT, optimize='optimal')
        #print(path_info)
        #print(path)
        path = ['einsum_path', (0, 3), (1, 2), (0, 1)]
        y_winograd = np.einsum('ik,ncxykl,mckl,jl->nmxiyj',
                               AT, a_winograd,
                               f_winograd, AT, optimize=path)
        y = y_winograd.reshape(n, m, zh+hr, zw+wr)\
                      [:,:,:zh,:zw]
        self.__z = y + self.b[np.newaxis, :, np.newaxis, np.newaxis]
        return self.afun.evaluate(self.__z)
      
    def backprop_scipy(self, delta):
        
        n, c, h, w = self.__a.shape
        m, fh, fw  = self.fshape 
        # Compute a'(z)*delta
        delta = self.afun.backprop(delta)
        
        # Compute bias change 
        self.db = np.sum(delta, axis=(0,2,3))

        # Compute df = delta * a^~. df has shape (m, c, fh, fw)
        for k in range(c):
            for j in range(m):
                for i in range(n):
                    self.df[j,k,:,:] += convolve2d(self.__a[i,k,::-1,::-1],
                                                  delta[i,j,:,:],
                                                  mode = 'valid')

        # Update delta 
        # The new delta has the same shape as __a: (n,c,h,w)
        delta2 = np.zeros_like(self.__a)
        for k in range(c):
            for i in range(n):
                for j in range(m):
                                
                    delta2[i, k,:,:] += convolve2d(delta[i,j,:,:],
                                                  self.f[j,k,::-1,::-1])

        return delta2
      
    def backprop_fft(self, delta):
      
      n, c, h, w = self.__a.shape
      m, fh, fw  = self.fshape 
      dh, dw     = h - fh, w - fw
      
      delta = self.afun.backprop(delta)
      
      # Compute bias change 
      self.db = np.sum(delta, axis=(0,2,3)) 
      
      # Compute filter change
      a_hat     = rfft2(self.__a[:,:,::-1,::-1], (h + dh, w + dw))
      delta_hat = rfft2(delta, (h + dh, w + dw)) 
      
      for k in range(c):
        for j in range(m):
          
          df_hat = irfft2(a_hat[:,k,:,:] * \
                          delta_hat[:,j,:,:],
                          (h + dh, w + dw)) 
          self.df[j,k,:,:] = df_hat[:,dh:-dh, dw:-dw].sum(axis=0)
          
      # Compute the next delta
      delta_hat = rfft2(delta, (h, w))
      f_hat     = rfft2(self.f[:,:,::-1,::-1], (h, w))

      da = np.zeros_like(self.__a)
      for i in range(n):
        for k in range(c):
          
          da_hat = irfft2(delta_hat[i,:,:,:] * \
                          f_hat[:,k,:,:], 
                          (h, w))
          da[i,k,:,:] = da_hat.sum(axis=0) 
          
      return da
    
    def backprop_im2col(self, delta):
      
      n, c, h, w = self.__a.shape
      m, fh, fw = self.fshape
      
      a_col, f_row = self.cache
    
      delta = self.afun.backprop(delta)
        
      self.db = np.sum(delta, axis=(0,2,3))
        
      # reshape delta 
      delta_mat = delta.transpose(1, 0, 2, 3).reshape(m, -1)
        
      # BLAS Level 3: GEMM for filter change
      df_row = delta_mat @ a_col.T
      # reshape df_row into df
      self.df = df_row.reshape(m, c, fh, fw)[:,:,::-1,::-1]
        
      # BLAS Level 3: GEMM for da_col
      da_col = f_row.T @ delta_mat
      # reshape back to image (col2im)
      return col2im(da_col, self.__a.shape, fh, fw)
        

    def update(self):

        self.optim.update([self.f, self.b],
                          [self.df, self.db]) 

class Pool2DLayer:

    def __init__(self, area, strict=False):

        self.area   = area
        self.shape  = None 
        self.mask   = None 
        self.strict = strict

    def evaluate(self, a):

        n, c, h, w = self.shape = a.shape
        
        ph, pw = self.area 
        nh, nw = h // ph, w // pw
        oh, ow = h % ph, w % pw 
        
        # Reduce a, if hight and width are odd
        a_reduced  = a[:,:,:h-oh,:w-ow]

        # Reshape reduced a to find maxima
        a_reshaped = a_reduced.reshape(n, c, nh, ph, nw, pw)

        # Find maxima and create mask 
        z = a_reshaped.max(axis=(3,5))
        z_newaxis = z[:,:,:,np.newaxis,:,np.newaxis]
        self.mask = (a_reshaped == z_newaxis)

        return z 
    
    def backprop(self, delta):

        n, c, h, w = self.shape 
        ph, pw     = self.area 

        # New delta should be of shape (n, c, h, w). 
        # It can be recovered from the reshaped form
        da_reshaped = np.zeros((n, c, h//ph, ph, w//pw, pw))

        # Broadcast delta to shape of da_reshaped
        delta_newaxis      = delta[:,:,:,np.newaxis,:,np.newaxis]
        delta_broadcast, _ = np.broadcast_arrays(delta_newaxis, da_reshaped)
        
        # Set points, where maximums where found to 1
        da_reshaped[self.mask]  = delta_broadcast[self.mask] 

        # Strict mode for correction of the subgradients in case of multiple maxima
        if self.strict:
            da_reshaped /= np.sum(self.mask, axis=(3,5), keepdims=True)
        
        # Build da from da_reshaped
        da = np.zeros(self.shape) 
        dah, daw = h - h%ph, w - w%pw 
        da[:,:,:dah,:daw] = da_reshaped.reshape(n, c, dah, daw)

        return da

    def update(self):

        """
         No update is done for Pooling layers. To keep the update method from 
         SequentialNet, it is included here, but nothing is done
        """
        pass 

class FlattenLayer:

    def __init__(self):

        self.shape = None

    def evaluate(self, a):

        self.shape = a.shape 
        # The -1 allows numpy to figure out the second dimension automaticly
        return a.reshape(self.shape[0],-1)

    def backprop(self, delta):

        # Reshape delta from shape (n, c*h*w) to shape (n, c, h, w)
        return delta.reshape(self.shape) 

    def update(self):

        pass 


if __name__ == "__main__":
  
  evaluations = ['scipy', 'fft', 'im2col']
  
  for e in evaluations:
    
    print('----------------------', e, '----------------------')
    # Gradient check for Conv2DLayer()

    # output
    verbose = False

    # structure
    n = 2
    c = 3
    #h = 5
    #w = 6
    h=10
    w=15
    m = 4
    fh = 2
    fw = 3
    zh = h - fh + 1
    zw = w - fw + 1

    # initial data
    f = np.random.randn(m, c, fh, fw)
    b = np.random.randn(m)

    # training data
    x = np.random.randn(n, c, h, w)
    y = np.random.randn(n, m, zh, zw)

    # create the test layer
    test_layer = Conv2DLayer((c,h,w),(m, fh, fw), afun=Abs(), eval_method=e)
    test_layer.f = f
    test_layer.b = b

    # forward step using the method evaluate of the layer
    a=test_layer.evaluate(x)
    z=test_layer._Conv2DLayer__z

    # backprop step using the method backprop of the layer
    # collect all important intermediate results
    dx2 = test_layer.backprop(a - y)
    db2 = test_layer.db
    df2=test_layer.df

    # Directly calculate results that cannot be recovered
    delta = np.sign(z) * (a - y)

    # forward step, used for numerical differentiation
    def conv2d(x, f, b):
        z = np.zeros_like(y)
        for i in range(n):
            for j in range(m):
                z[i, j, :, :] = b[j]
                for k in range(c):
                    z[i, j, :, :] += convolve2d(x[i, k, :, :],
                                                f[j, k, :, :],
                                                mode='valid')
        return z


    C = .5 * np.sum((a - y) * (a - y))

    # parameter for numerical differentiation
    hh = 1e-7

    print('Gradient check for a')

    # numerical differentiation for a
    da = np.zeros_like(a)
    for i in range(n):
        for j in range(m):
            for k in range(zh):
                for l in range(zw):
                    a_tilde = np.copy(a)
                    a_tilde[i, j, k, l] += hh
                    da[i, j, k, l] = \
                        (.5 * np.sum((a_tilde - y) * (a_tilde - y)) - C) / hh

    if verbose:
        print(da)

    # chain rule aka backpropagation for a
    if verbose:
        print(a - y)

    # difference for a
    print(np.linalg.norm(da - (a - y)))

    print('Gradient check for z')

    # numerical differentiation for z
    dz = np.zeros_like(z)
    for i in range(n):
        for j in range(m):
            for k in range(zh):
                for l in range(zw):
                    z_tilde = np.copy(z)
                    z_tilde[i, j, k, l] += hh
                    a = np.abs(z_tilde)
                    dz[i, j, k, l] = (.5 * np.sum((a - y) * (a - y)) - C) / hh

    if verbose:
        print(dz)

    # chain rule aka backpropagation for z
    if verbose:
        print(delta)

    # difference for z
    print(np.linalg.norm(dz - delta))


    print('Gradient check for b')

    # numerical differentiation for b
    db = np.zeros_like(b)
    for i in range(m):
        b_tilde = np.copy(b)
        b_tilde[i] += hh
        z = conv2d(x, f, b_tilde)
        a = np.abs(z)
        db[i] = (.5 * np.sum((a - y) * (a - y)) - C) / hh

    if verbose:
        print(db)

    if verbose:
        print(db2)

    # difference for b
    print(np.linalg.norm(db - db2))


    print('Gradient check for f')

    # numerical differentiation for f
    df = np.zeros_like(f)
    for i in range(m):
        for j in range(c):
            for k in range(fh):
                for l in range(fw):
                    f_tilde = np.copy(f)
                    f_tilde[i, j, k, l] += hh
                    z = conv2d(x, f_tilde, b)
                    a = np.abs(z)
                    df[i, j, k, l] = (.5 * np.sum((a - y) * (a - y)) - C) / hh

    if verbose:
        print(df)

    if verbose:
        print(df2)

    # difference for f
    print(np.linalg.norm(df - df2))


    print('Gradient check for x')

    # numerical differentiation for x
    dx = np.zeros_like(x)
    for i in range(n):
        for j in range(c):
            for k in range(h):
                for l in range(w):
                    x_tilde = np.copy(x)
                    x_tilde[i, j, k, l] += hh
                    z = conv2d(x_tilde, f, b)
                    a = np.abs(z)
                    dx[i, j, k, l] = (.5 * np.sum((a - y) * (a - y)) - C) / hh

    if verbose:
        print(dx)

    if verbose:
        print(dx2)

    # difference for x
    print(np.linalg.norm(dx - dx2))

    # Test a small example for the convolutional layer

    # structure
    n=1
    m=1
    c=1
    fh=2
    fw=2
    h=3
    w=3
    zh = h - fh + 1
    zw = w - fw + 1

    # initial data
    f=np.arange(m*c*fh*fw)
    f=np.reshape(f,(m,c,fh,fw))
    f = f.astype(np.float64)
    b=np.arange(m)
    b=b.astype(np.float64)

    # training data
    x=np.arange(n*c*h*w)
    x = np.reshape(x,(n, c, h, w))
    x = x.astype(np.float64)
    y=np.arange(n*m*zh*zw)
    y = np.reshape(y,(n, m, zh, zw))
    y = y.astype(np.float64)

    # create the test layer
    test_layer2 = Conv2DLayer((c,h,w),(m, fh, fw), afun=ReLU(),optim=SGD(eta=1))
    test_layer2.f = f
    test_layer2.b = b

    print('Check the evaluation of Conv2DLayer()')

    x_conv=test_layer2.evaluate(x)
    x_result=[[[[5,11],[23,29]]]]
    print(np.linalg.norm(x_conv - x_result))

    print('Check the backpropagation of Conv2DLayer()')

    delta_conv=test_layer2.backprop(x_conv-y)
    delta_result=np.array([[[[15,40,20],[68,130,52],[21,26,0]]]])
    print(np.linalg.norm(delta_conv - delta_result))

    # Output check for Flattenlayer

    print('Check the evaluation of FlattenLayer()')

    a = np.arange(16)

    # Reshape using FlattenLayer()

    test_flattenlayer = FlattenLayer()
    a_input = np.reshape(a, (2, 2, 2, 2))
    a_flatten=test_flattenlayer.evaluate(a_input)

    # Compare with the correct result

    a_result = np.reshape(a, (2, 8))
    print(np.linalg.norm(a_flatten - a_result))

    # Output check for Pool2DLayer

    print('Check the evaluation of Pool2DLayer()')

    # Pooling using Pool2DLayer()

    area=(2,2)
    test_pooling=Pool2DLayer(area)
    a_pooling=test_pooling.evaluate(a_input)

    # Compare with the correct result:

    a_result=np.array([[[[3]],[[7]]],[[[11]],[[15]]]])
    print(np.linalg.norm(a_pooling - a_result))

    # Backprop check for Pool2DLayer

    print('Check the backpropagation of Pool2DLayer()')

    delta = np.array([[[[10]], [[20]]], [[[30]], [[40]]]])
    da=back_pooling=test_pooling.backprop(delta)

    # Compare with the correct result:

    da_result = np.array([[[[0,0],[0,10]], [[0,0],[0,20]]], [[[0,0],[0,30]], [[0,0],[0,40]]]])
    print(np.linalg.norm(da_result - da))

    # Test the option strict

    print('Check the option strict in backprop of Pool2DLayer()')

    test_pooling = Pool2DLayer(area,strict=True)
    a[0]=3
    a[1] = 3
    a[2] = 3
    a_input = np.reshape(a, (2, 2, 2, 2))
    a_pooling = test_pooling.evaluate(a_input)
    delta = np.array([[[[10]], [[20]]], [[[30]], [[40]]]])
    da=test_pooling.backprop(delta)

    # Compare with the correct result:

    da_result = np.array([[[[2.5,2.5],[2.5,2.5]], [[0,0],[0,20]]], [[[0,0],[0,30]], [[0,0],[0,40]]]])
    print(np.linalg.norm(da_result - da))
