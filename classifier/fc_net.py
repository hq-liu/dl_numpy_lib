from builtins import range
from builtins import object
import numpy as np

from layers import *
from layers_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.
    The architecure should be affine - relu - affine - softmax.
    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.
    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        self.params['W1'] = np.random.randn(input_dim, hidden_dim) * weight_scale
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b2'] = np.zeros(num_classes)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.
        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        W1, b1, W2, b2 =self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2']
        hidden_layer, first_cache = affine_relu_forward(x=X, w=W1, b=b1)
        scores, second_cache = affine_forward(x=hidden_layer, w=W2, b=b2)

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        loss, dscores = softmax_loss(scores, y)
        dhidden, grads['W2'], grads['b2'] = affine_backward(dscores, second_cache)
        dx, grads['W1'], grads['b1'] = affine_relu_backward(dhidden, first_cache)

        # reg
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        grads['W1'] += self.reg * W1
        grads['W2'] += self.reg * W2


        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be
    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.
    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.
        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        self.params['W1'] = np.random.randn(input_dim, hidden_dims[0]) * weight_scale
        self.params['b1'] = np.zeros(hidden_dims[0])
        for k in range(0, self.num_layers-2):
            self.params['W' + str(k+2)] = np.random.randn(hidden_dims[k], hidden_dims[k+1]) * weight_scale
            self.params['b' + str(k+2)] = np.zeros(hidden_dims[k+1])
            if use_batchnorm:
                self.params['gamma' + str(k+1)] = np.ones(hidden_dims[k])
                self.params['beta' + str(k+1)] = np.zeros(hidden_dims[k])
        self.params['W' + str(self.num_layers)] = \
            np.random.randn(hidden_dims[-1], num_classes) * weight_scale
        self.params['b' + str(self.num_layers)] = np.zeros(num_classes)
        if use_batchnorm:
            self.params['gamma' + str(self.num_layers-1)] = np.ones(hidden_dims[-1])
            self.params['beta' + str(self.num_layers-1)] = np.zeros(hidden_dims[-1])

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.
        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        hidden_layers, caches = {}, {}
        dp_caches = {}
        hidden_layers[0] = X
        W = self.params['W1']
        for i in range(self.num_layers):
            W, b = self.params['W' + str(i+1)], self.params['b' + str(i+1)]
            if i == self.num_layers-1:
                hidden_layers[i+1], caches[i] = affine_forward(hidden_layers[i], W, b)
            else:
                if self.use_batchnorm:
                    gamma, beta = self.params['gamma' + str(i+1)], self.params['beta' + str(i+1)]
                    hidden_layers[i+1], caches[i] = affine_bn_relu_forward(hidden_layers[i], W, b, gamma, beta, self.bn_params[i])
                else:
                    hidden_layers[i+1], caches[i] = affine_relu_forward(hidden_layers[i], W, b)
                if self.use_dropout:
                    hidden_layers[i+1], dp_caches[i] = dropout_forward(hidden_layers[i+1], self.dropout_param)

        scores = hidden_layers[self.num_layers]

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}

        loss, dscore = softmax_loss(scores, y)
        dhiddens = {}
        dhiddens[self.num_layers] = dscore
        for i in range(self.num_layers, 0, -1):
            if i == self.num_layers:
                dhiddens[i-1], grads['W'+str(i)], grads['b'+str(i)] = \
                    affine_backward(dhiddens[i], caches[i-1])
            else:
                if self.use_dropout:
                    dhiddens[i] = dropout_backward(dhiddens[i], dp_caches[i-1])
                if self.use_batchnorm:
                    dhiddens[i-1], grads['W' + str(i)], grads['b' + str(i)], grads['gamma' + str(i)], grads['beta' + str(i)] = \
                        affine_bn_relu_backward(dhiddens[i], caches[i-1])
                else:
                    dhiddens[i-1], grads['W' + str(i)], grads['b' + str(i)] = \
                        affine_relu_backward(dhiddens[i], caches[i - 1])
            loss += 0.5 * self.reg * np.sum(self.params['W'+str(i)] ** 2)
            grads['W'+str(i)] += self.reg * self.params['W'+str(i)]

        return loss, grads