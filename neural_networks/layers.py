from __future__ import print_function, division
import math
import numpy as np
import copy

from .activation_functions import Sigmoid, Softmax, TanH
import ravop as R

class Layer(object):

    def set_input_shape(self, shape):
        """ Sets the shape that the layer expects of the input in the forward
        pass method """
        self.input_shape = shape

    def layer_name(self):
        """ The name of the layer. Used in model summary. """
        return self.__class__.__name__

    def parameters(self):
        """ The number of trainable parameters used by the layer """
        return R.t(0)

    def forward_pass(self, X, training):
        """ Propogates the signal forward in the network """
        raise NotImplementedError()

    def backward_pass(self, accum_grad):
        """ Propogates the accumulated gradient backwards in the network.
        If the has trainable weights then these weights are also tuned in this method.
        As input (accum_grad) it receives the gradient with respect to the output of the layer and
        returns the gradient with respect to the output of the previous layer. """
        raise NotImplementedError()

    def output_shape(self):
        """ The shape of the output produced by forward_pass """
        raise NotImplementedError()

class Dense(Layer):
    """A fully-connected NN layer.
    Parameters:
    -----------
    n_units: int
        The number of neurons in the layer.
    input_shape: tuple
        The expected input shape of the layer. For dense layers a single digit specifying
        the number of features of the input. Must be specified if it is the first layer in
        the network.
    """
    def __init__(self, n_units, input_shape=None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.W = None
        self.w0 = None

    def initialize(self, optimizer):
        # Initialize the weights
        limit = R.div(R.t(1), R.square_root(R.t(self.input_shape[0])))
        limit_value = limit()
        self.W = R.t(np.random.uniform(-limit_value, limit_value, (self.input_shape[0], self.n_units)))
        self.w0 = R.t(np.zeros((1,self.n_units)))
        # Weight optimizers
        self.W_opt  = copy.copy(optimizer)
        self.w0_opt = copy.copy(optimizer)

    def parameters(self):
        return R.t(int(np.prod(self.W().shape))) + R.t(int(np.prod(self.w0().shape)))

    def forward_pass(self, X, training=True):
        self.layer_input = X
        return X.dot(self.W).add(self.w0)

    def backward_pass(self, accum_grad):
        # Save weights used during forwards pass
        W = self.W

        if self.trainable:
            # Calculate gradient w.r.t layer weights
            grad_w = R.transpose(self.layer_input).dot(accum_grad)
            grad_w0 = R.t(np.sum(accum_grad(), axis=0, keepdims=True))

            # Update the layer weights
            self.W = self.W_opt.update(self.W, grad_w)
            self.w0 = self.w0_opt.update(self.w0, grad_w0)

        # Return accumulated gradient for next layer
        # Calculated based on the weights used during the forward pass
        accum_grad = accum_grad.dot(R.transpose(W))
        return accum_grad

    def output_shape(self):
        return (self.n_units, )

class BatchNormalization(Layer):
    """Batch normalization.
    """
    def __init__(self, momentum=0.99):
        self.momentum = R.t(momentum)
        self.trainable = True
        self.eps = R.t(0.01)
        self.running_mean = None
        self.running_var = None

    def initialize(self, optimizer):
        # Initialize the parameters
        self.gamma  = R.t(np.ones(self.input_shape))
        self.beta = R.t(np.zeros(self.input_shape))
        # parameter optimizers
        self.gamma_opt  = copy.copy(optimizer)
        self.beta_opt = copy.copy(optimizer)

    def parameters(self):
        return R.t(int(np.prod(self.gamma().shape))) + R.t(int(np.prod(self.beta().shape)))

    def forward_pass(self, X, training=True):

        # Initialize running mean and variance if first run
        if self.running_mean is None:
            self.running_mean = R.mean(X, axis=0)
            self.running_var = R.variance(X, axis=0)

        if training and self.trainable:
            mean = R.mean(X, axis=0)
            var = R.variance(X, axis=0)
            self.running_mean = self.momentum * self.running_mean + (R.t(1) - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (R.t(1) - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var

        # Statistics saved for backward pass
        self.X_centered = X - mean
        self.stddev_inv = R.div(R.t(1), R.square_root(var + self.eps))

        X_norm = self.X_centered * self.stddev_inv
        output = self.gamma * X_norm + self.beta

        return output

    def backward_pass(self, accum_grad):

        # Save parameters used during the forward pass
        gamma = self.gamma

        # If the layer is trainable the parameters are updated
        if self.trainable:
            X_norm = self.X_centered * self.stddev_inv
            grad_gamma = R.sum(accum_grad * X_norm, axis=0)
            grad_beta = R.sum(accum_grad, axis=0)

            self.gamma = self.gamma_opt.update(self.gamma, grad_gamma)
            self.beta = self.beta_opt.update(self.beta, grad_beta)

        batch_size = R.t(accum_grad().shape[0])

        # The gradient of the loss with respect to the layer inputs (use weights and statistics from forward pass)
        
        accum_grad = R.div(R.t(1),batch_size) * gamma * self.stddev_inv * (batch_size * accum_grad - R.sum(accum_grad, axis=0) 
                                                                            - self.X_centered * R.square(self.stddev_inv) * R.sum(accum_grad * self.X_centered, axis=0))
        
        return accum_grad

    def output_shape(self):
        return self.input_shape


class Dropout(Layer):
    """A layer that randomly sets a fraction p of the output units of the previous layer
    to zero.

    Parameters:
    -----------
    p: float
        The probability that unit x is set to zero.
    """
    def __init__(self, p=0.2):
        self.p = p
        self._mask = None
        self.input_shape = None
        self.n_units = None
        self.pass_through = True
        self.trainable = True

    def forward_pass(self, X, training=True):
        c = R.t(1) - R.t(self.p)
        if training:
            self._mask = np.random.uniform(size=X().shape) > self.p
            c = R.t(self._mask)
        return X * c

    def backward_pass(self, accum_grad):
        return accum_grad * R.t(self._mask)

    def output_shape(self):
        return self.input_shape

activation_functions = {
    'sigmoid': Sigmoid,
    'softmax': Softmax,
    'tanh': TanH
}

class Activation(Layer):
    """A layer that applies an activation operation to the input.

    Parameters:
    -----------
    name: string
        The name of the activation function that will be used.
    """

    def __init__(self, name):
        self.activation_name = name
        self.activation_func = activation_functions[name]()
        self.trainable = True

    def layer_name(self):
        return "Activation (%s)" % (self.activation_func.__class__.__name__)

    def forward_pass(self, X, training=True):
        self.layer_input = X
        return self.activation_func(X)

    def backward_pass(self, accum_grad):
        return accum_grad * self.activation_func.gradient(self.layer_input)

    def output_shape(self):
        return self.input_shape
