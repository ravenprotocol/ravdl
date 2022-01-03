from __future__ import print_function, division
import math
import numpy as np
import copy
import random
from .utils import Activation,LeakyReLU,Softmax

import ravop.core.c as R

import sys
act=Activation()
activation_functions = {
    'leaky_relu': act.LeakyReLU,
    'softmax': act.Softmax
}




class Layer(object):
    def __init__(self):
        self.input_shape = None

    def set_input_shape(self, shape):
        self.input_shape = shape

    def layer_name(self):
        return self.__class__.__name__

    def parameters(self):
        return 0

    def forward_pass(self, X, training):
        raise NotImplementedError()

    def backward_pass(self, accum_grad):
        raise NotImplementedError()

    def output_shape(self):
        raise NotImplementedError()


class Dense(Layer):

    def __init__(self, n_units, input_shape=None,activation=None):
        super().__init__()
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.activation= activation
        if activation is not None:
            self.activation_func=activation_functions[self.activation]
        self.W = None
        self.w0 = None

    def initialize(self, optimizer):
        #print(optimizer)
        limit = R.div(R.Scalar(1) , R.square_root(R.Scalar(self.input_shape[0])) )
        limit.wait_till_computed()

        self.W = R.Tensor(np.random.uniform(-limit(), limit(), (self.input_shape[0], self.n_units)))
        self.w0 = R.Tensor(np.zeros((1, self.n_units)) )
        self.W_opt = copy.copy(optimizer)
        self.w0_opt = copy.copy(optimizer)

    def parameters(self):
        pass
        #print(np.prod(self.W.shape) + np.prod(self.w0.shape))
        #return np.prod(self.W.shape) + np.prod(self.w0.shape)

    def forward_pass(self, X, training=True):
        self.layer_input = X
        #print(self.W, self.w0,np.shape(X.output), np.shape(self.W.output), np.shape(self.w0.output),np.shape(temp.output))
        z= X.dot(self.W)
        #z=z.foreach(operation="add",params=self.w0)
        z= z.add(self.w0)
        if self.activation is not None:
            return self.activation_func(z)
        return z

    def backward_pass(self, accum_grad):
        W = self.W
        inp=self.layer_input
        if self.trainable:
            if self.activation is not None:
                activ=self.activation_func(inp)
            grad_w=R.transpose(inp).dot(accum_grad)
            grad_w0 = R.sum(accum_grad)

            # Update the layer weights
            self.W = self.W_opt.update(self.W, grad_w)
            self.w0 = self.w0_opt.update(self.w0, grad_w0)
        #print("_______________________________________________________backpass")
        # Return accumulated gradient for next layer
        # Calculated based on the weights used during the forward pass
        accum_grad = accum_grad.dot(R.transpose(W))
        return accum_grad

    def output_shape(self):
        return (self.n_units,)



class Activation(Layer):

    def __init__(self, name):
        super().__init__()
        self.activation_name = name
        self.activation_func = activation_functions[name]
        self.trainable = True

    def forward_pass(self, X, training=True):
        self.layer_input = X
        act= self.activation_func(X)
        return act

    def backward_pass(self, accum_grad):
        #print("qqqqwwwww")
        ac=self.activation_func(self.layer_input)
        z=accum_grad * ac
        z.wait_till_computed()
        #print(self.activation_name)
        return z

    def output_shape(self):
        return self.input_shape



class Dropout(Layer):
    def __init__(self,rate,noise_shape=None,input_shape=None):
        super().__init__()
        self.rate=rate
        self.noise_shape=noise_shape
        self.mask=None

    def forward_pass(self, input_layer, training):
        self.mask=R.Tensor([random.random() for _ in range(self.input_shape[0])]) < R.Scalar(self.rate)
        output=self.mask.multiply(input_layer)
        return output

    def backward_pass(self, accum_grad):
        return accum_grad*self.mask

    def output_shape(self):
        return self.input_shape





class BatchNormalization(Layer):
    def __init__(self, momentum=0.99):
        super().__init__()
        self.momentum = R.Scalar(momentum)
        self.eps = 0.01
        self.running_mean = None
        self.running_var = None
        self.trainable=True

    def initialize(self, optimizer):
        # Initialize the parameters
        self.gamma  = R.Tensor(np.ones(self.input_shape))
        self.beta = R.Tensor(np.zeros(self.input_shape))
        # parameter optimizers
        self.gamma_opt  = copy.copy(optimizer)
        self.beta_opt = copy.copy(optimizer)


    def forward_pass(self, X, training=True):

        # Initialize running mean and variance if first run
        if self.running_mean is None:
            self.running_mean = R.mean(X, axis=0)
            self.running_var = R.variance(X, axis=0)

        if training and self.trainable:
            mean = R.mean(X, axis=0)
            var = R.variance(X, axis=0)
            self.running_mean = self.momentum * self.running_mean + (R.Scalar(1) - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (R.Scalar(1) - self.momentum)* var
        else:
            mean = self.running_mean
            var = self.running_var

        # Statistics saved for backward pass
        self.X_centered = X - mean
        self.stddev_inv = R.Scalar(1).div( R.square_root (var + R.Scalar(self.eps)))

        X_norm = self.X_centered * self.stddev_inv
        output = self.gamma * X_norm + self.beta

        return output
    def backward_pass(self, accum_grad):

        # Save parameters used during the forward pass
        gamma = self.gamma

        # If the layer is trainable the parameters are updated
        if self.trainable:
            X_norm = self.X_centered * self.stddev_inv
            grad_gamma = np.sum(accum_grad * X_norm, axis=0)
            grad_beta = np.sum(accum_grad, axis=0)

            self.gamma = self.gamma_opt.update(self.gamma, grad_gamma)
            self.beta = self.beta_opt.update(self.beta, grad_beta)
        accum_grad.wait_till_computed()
        batch_size = accum_grad().shape[0]

        # The gradient of the loss with respect to the layer inputs (use weights and statistics from forward pass)
        accum_grad = R.Scalar(1).div(R.Scalar(batch_size)) * gamma * self.stddev_inv * (
            R.Scalar(batch_size) * accum_grad- R.sum(accum_grad, axis=0) - self.X_centered * R.pow(self.stddev_inv,R.Scalar(2)) * R.sum(accum_grad * self.X_centered, axis=0))

        return accum_grad

    def output_shape(self):
        return self.input_shape