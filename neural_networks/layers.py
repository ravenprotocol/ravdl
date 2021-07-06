from __future__ import print_function, division
import math
import numpy as np
import copy
from ravdl.activations import Activation

import ravop.core as R

import sys
act=Activation()
activation_functions = {
    'leaky_relu': act.LeakyRelu,
    'softmax':act.softmax
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

    def __init__(self, n_units, input_shape=None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.W = None
        self.w0 = None

    def initialize(self, optimizer):
        print(optimizer)
        limit = R.div(R.Scalar(1) , R.square_root(R.Scalar(self.input_shape[0])) )
        while limit.status!="computed":
            pass
        self.W = R.Tensor(np.random.uniform(-limit.output, limit.output, (self.input_shape[0], self.n_units)))
        self.w0 = R.Tensor(np.zeros((1, self.n_units)) )
        self.W_opt = copy.copy(optimizer)
        self.w0_opt = copy.copy(optimizer)

    def parameters(self):
        print(np.prod(self.W.shape) + np.prod(self.w0.shape))
        return np.prod(self.W.shape) + np.prod(self.w0.shape)

    def forward_pass(self, X, training=True):
        self.layer_input = X
        return R.add( X.dot(self.W) , self.w0)

    def backward_pass(self, accum_grad):
        W = self.W
        if self.trainable:
            grad_w = R.transpose(self.layer_input).dot(accum_grad)
            grad_w0 = R.sum(accum_grad ,axis=0)

            grad_w=R.transpose(self.layer_input).dot(accum_grad)
            grad_w0 = R.sum(accum_grad)

            # Update the layer weights
            self.W = self.W_opt.update(self.W, grad_w)
            self.w0 = self.w0_opt.update(self.w0, grad_w0)

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
        return accum_grad * self.activation_func(self.layer_input)

    def output_shape(self):
        return self.input_shape


