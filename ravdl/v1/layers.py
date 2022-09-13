from __future__ import print_function, division
import math
import numpy as np
import copy
from ..globals import globals as g
from .activation_functions import Sigmoid, Softmax, TanH, ReLU
import ravop as R
import onnx
from ..utils import create_initializer_tensor

class Layer(object):

    def set_input_shape(self, shape):
        """ Sets the shape that the layer expects of the input in the forward
        pass method """
        self.input_shape = shape

    def set_layer_name(self, name=None):
        """ Sets the name of the layer. """
        if name is not None:
            self.layer_name = name
        else:
            self.layer_name = self.__class__.__name__

    def get_layer_name(self):
        """ The name of the layer. Used in model summary. """
        return self.layer_name

    def parameters(self):
        """ The number of trainable parameters used by the layer """
        return 0

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
        limit = R.div(g.one, R.square_root(R.t(int(self.input_shape[0]))))
        self.W = R.random_uniform(R.neg(limit), limit, size=(int(self.input_shape[0]), self.n_units))
        self.w0 = R.t(np.zeros((1,self.n_units)))

        # np equivalent for summary
        np_limit = 1 / math.sqrt(self.input_shape[0])
        self.np_W  = np.random.uniform(-np_limit, np_limit, (self.input_shape[0], self.n_units))
        self.np_w0 = np.zeros((1, self.n_units))

        # Weight optimizers
        self.W_opt  = copy.copy(optimizer)
        self.w0_opt = copy.copy(optimizer)

    def parameters(self): 
        return np.prod(self.np_W.shape) + np.prod(self.np_w0.shape)

    def forward_pass(self, X, training=True):
        self.layer_input = X
        return X.dot(self.W).add(self.w0)

    def backward_pass(self, accum_grad):
        # Save weights used during forwards pass
        W = self.W

        if self.trainable:
            # Calculate gradient w.r.t layer weights
            grad_w = R.transpose(self.layer_input).dot(accum_grad)
            grad_w0 = R.sum(accum_grad, axis=0, keepdims="True")

            # Update the layer weights
            self.W = self.W_opt.update(self.W, grad_w)
            self.w0 = self.w0_opt.update(self.w0, grad_w0)

        # Return accumulated gradient for next layer
        # Calculated based on the weights used during the forward pass
        accum_grad = accum_grad.dot(R.transpose(W))
        return accum_grad

    def output_shape(self):
        return (self.n_units, )

    def save_weights(self):
        self.W.persist_op(self.get_layer_name() + "_W")
        self.w0.persist_op(self.get_layer_name() + "_w0")

    def fetch_onnx_params(self,input_initializer, output_initializer = None):
        W_fetched = R.fetch_persisting_op(op_name=self.get_layer_name() + "_W")
        w0_fetched = R.fetch_persisting_op(op_name=self.get_layer_name() + "_w0")

        if isinstance(W_fetched, list):
            W_fetched = np.array(W_fetched)
        if isinstance(w0_fetched, list):
            w0_fetched = np.array(w0_fetched)

        dense_W_initializer_name = self.layer_name + "_W"
        W_initializer_tensor = create_initializer_tensor(name=dense_W_initializer_name, tensor_array= W_fetched, data_type=onnx.TensorProto.FLOAT)
        dense_w0_initializer_name = self.layer_name + "_w0"
        w0_initializer_tensor = create_initializer_tensor(name=dense_w0_initializer_name, tensor_array= w0_fetched, data_type=onnx.TensorProto.FLOAT)

        if output_initializer:
            dense_output_node_name = output_initializer
        else:
            dense_output_node_name = self.layer_name + "_output"

        dense_node = onnx.helper.make_node(
            name=self.layer_name,
            op_type="Gemm",
            inputs=[input_initializer, dense_W_initializer_name, dense_w0_initializer_name],
            outputs=[dense_output_node_name]
        )

        return dense_node, dense_output_node_name, [W_initializer_tensor, w0_initializer_tensor]

class BatchNormalization(Layer):
    """Batch normalization.
    """
    def __init__(self, momentum=0.99, epsilon=1e-2):
        self.momentum = R.t(momentum)
        self.float_momentum = momentum
        self.trainable = True
        self.eps = R.t(epsilon)
        self.float_eps = epsilon
        self.running_mean = None
        self.running_var = None

    def initialize(self, optimizer):
        # Initialize the parameters
        if len(self.input_shape) == 1:
            shape = (1, self.input_shape[0])
        else:
            shape = (1,self.input_shape[0], 1,1)

        self.gamma  = R.t(np.ones(shape))
        self.beta = R.t(np.zeros(shape))

        # np equivalent for summary params
        self.np_gamma = np.ones(shape)
        self.np_beta = np.zeros(shape)
        

        # parameter optimizers
        self.gamma_opt  = copy.copy(optimizer)
        self.beta_opt = copy.copy(optimizer)

        self.running_mean = R.t(np.zeros(shape))
        self.running_var = R.t(np.ones(shape))

    def parameters(self):
        return np.prod(self.np_gamma.shape) + np.prod(self.np_beta.shape)        

    def forward_pass(self, X, training=True):

        # Initialize running mean and variance if first run
        # if self.running_mean is None:
        #     self.running_mean = R.mean(X, axis=0)
        #     self.running_var = R.variance(X, axis=0)

        if training and self.trainable:
            if len(self.input_shape) == 1:
                mean = R.mean(X, axis=0)
                var = R.variance(X, axis=0)
            else:
                mean = R.mean(X, axis=(0,2,3), keepdims="True")
                var = R.variance(X, axis=(0,2,3), keepdims="True")
            self.running_mean = self.momentum * self.running_mean + (g.one - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (g.one - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var

        # Statistics saved for backward pass
        self.X_centered = X - mean
        self.stddev_inv = R.div(g.one, R.square_root(var + self.eps))

        X_norm = self.X_centered * self.stddev_inv
        output = self.gamma * X_norm + self.beta

        return output

    def backward_pass(self, accum_grad):

        # Save parameters used during the forward pass
        gamma = self.gamma

        # If the layer is trainable the parameters are updated
        if self.trainable:
            X_norm = self.X_centered * self.stddev_inv
            if len(self.input_shape) == 1:
                grad_gamma = R.sum(accum_grad * X_norm, axis=0, keepdims="True")
                grad_beta = R.sum(accum_grad, axis=0, keepdims="True")
            else:
                grad_gamma = R.sum(accum_grad * X_norm, axis=(0,2,3), keepdims="True")
                grad_beta = R.sum(accum_grad, axis=(0,2,3), keepdims="True")

            self.gamma = self.gamma_opt.update(self.gamma, grad_gamma)
            self.beta = self.beta_opt.update(self.beta, grad_beta)

        batch_size = accum_grad.shape(index=0)

        # The gradient of the loss with respect to the layer inputs (use weights and statistics from forward pass)

        if len(self.input_shape) == 1:
            accum_grad = R.div(g.one,batch_size) * gamma * self.stddev_inv * (batch_size * accum_grad - R.sum(accum_grad, axis=0,keepdims="True") 
                                                                                - self.X_centered * R.square(self.stddev_inv) * R.sum(accum_grad * self.X_centered, axis=0, keepdims="True"))
        else:
            accum_grad = R.div(g.one,batch_size) * gamma * self.stddev_inv * (batch_size * accum_grad - R.sum(accum_grad, axis=(0,2,3),keepdims="True") 
                                                                                - self.X_centered * R.square(self.stddev_inv) * R.sum(accum_grad * self.X_centered, axis=(0,2,3), keepdims="True"))
            
        return accum_grad

    def output_shape(self):
        return self.input_shape

    def save_weights(self):
        self.gamma.persist_op(self.get_layer_name() + "_gamma")
        self.beta.persist_op(self.get_layer_name() + "_beta")
        self.running_mean.persist_op(self.get_layer_name() + "_running_mean")
        self.running_var.persist_op(self.get_layer_name() + "_running_var")

    def fetch_onnx_params(self,input_initializer, output_initializer = None):
        gamma_fetched = R.fetch_persisting_op(op_name=self.get_layer_name() + "_gamma")
        beta_fetched = R.fetch_persisting_op(op_name=self.get_layer_name() + "_beta")
        running_mean_fetched = R.fetch_persisting_op(op_name=self.get_layer_name() + "_running_mean")
        running_var_fetched = R.fetch_persisting_op(op_name=self.get_layer_name() + "_running_var")

        if isinstance(gamma_fetched, list):
            gamma_fetched = np.array(gamma_fetched)
        if isinstance(beta_fetched, list):
            beta_fetched = np.array(beta_fetched)
        if isinstance(running_mean_fetched, list):
            running_mean_fetched = np.array(running_mean_fetched)
        if isinstance(running_var_fetched, list):
            running_var_fetched = np.array(running_var_fetched)

        bn_gamma_initializer_name = self.layer_name + "_gamma"
        gamma_initializer_tensor = create_initializer_tensor(name=bn_gamma_initializer_name, tensor_array= gamma_fetched.reshape((-1)), data_type=onnx.TensorProto.FLOAT)
        bn_beta_initializer_name = self.layer_name + "_beta"
        beta_initializer_tensor = create_initializer_tensor(name=bn_beta_initializer_name, tensor_array= beta_fetched.reshape((-1)), data_type=onnx.TensorProto.FLOAT)
        bn_mean_initializer_name = self.layer_name + "_mean"
        mean_initializer_tensor = create_initializer_tensor(name=bn_mean_initializer_name, tensor_array= running_mean_fetched.reshape((-1)), data_type=onnx.TensorProto.FLOAT)
        bn_var_initializer_name = self.layer_name + "_var"
        var_initializer_tensor = create_initializer_tensor(name=bn_var_initializer_name, tensor_array= running_var_fetched.reshape((-1)), data_type=onnx.TensorProto.FLOAT)
        
        if output_initializer:
            bn_output_node_name = output_initializer
        else:
            bn_output_node_name = self.layer_name + "_output"

        bn_node = onnx.helper.make_node(
            name=self.layer_name,  # Name is optional.
            op_type="BatchNormalization",
            inputs=[
                input_initializer, bn_gamma_initializer_name,
                bn_beta_initializer_name, bn_mean_initializer_name,
                bn_var_initializer_name
            ],
            momentum=self.float_momentum,
            epsilon=self.float_eps,
            outputs=[bn_output_node_name],
        )

        return bn_node, bn_output_node_name, [gamma_initializer_tensor, beta_initializer_tensor, mean_initializer_tensor, var_initializer_tensor]


class Dropout(Layer):
    """A layer that randomly sets a fraction p of the output units of the previous layer
    to zero.

    Parameters:
    -----------
    p: float
        The probability that unit x is set to zero.
    """
    def __init__(self, p=0.2):
        self.p = R.t(p)
        self._mask = None
        self.input_shape = None
        self.n_units = None
        self.pass_through = True
        self.trainable = True

    def forward_pass(self, X, training=True):
        # c = g.one - self.p
        if training:
            self._mask = R.greater(R.random_uniform(g.zero,g.one,size=R.shape(X)), self.p)#X().shape
            c = self._mask * R.div(g.one,g.one-self.p)
            return X * c
        return X

    def backward_pass(self, accum_grad):
        return accum_grad * self._mask

    def output_shape(self):
        return self.input_shape

    def save_weights(self):
        self.p.persist_op(self.get_layer_name() + "_p")

    def fetch_onnx_params(self,input_initializer, output_initializer = None):
        p_fetched = R.fetch_persisting_op(op_name=self.get_layer_name() + "_p")

        if not isinstance(p_fetched, np.ndarray):
            p_fetched = np.array(p_fetched)

        dropout_p_initializer_name = self.layer_name + "_p"
        p_initializer_tensor = create_initializer_tensor(name=dropout_p_initializer_name, tensor_array= p_fetched, data_type=onnx.TensorProto.FLOAT)
    
        if output_initializer:
            dropout_output_node_name = output_initializer
        else:
            dropout_output_node_name = self.layer_name + "_output"

        dropout_node = onnx.helper.make_node(
            name=self.layer_name,
            op_type="Dropout",
            inputs=[input_initializer, dropout_p_initializer_name],
            outputs=[dropout_output_node_name]
        )

        return dropout_node, dropout_output_node_name, [p_initializer_tensor]

activation_functions = {
    'sigmoid': Sigmoid,
    'softmax': Softmax,
    'tanh': TanH,
    'relu': ReLU
}

onnx_activation_functions = {
    'relu': 'Relu',
    'sigmoid': 'Sigmoid',
    'softmax': 'Softmax',
    'tanh': 'Tanh'
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

    # def layer_name(self):
    #     return "Activation (%s)" % (self.activation_func.__class__.__name__)

    def forward_pass(self, X, training=True):
        self.layer_input = X
        return self.activation_func(X)

    def backward_pass(self, accum_grad):
        return accum_grad * self.activation_func.gradient(self.layer_input)

    def output_shape(self):
        return self.input_shape

    def save_weights(self):
        pass

    def fetch_onnx_params(self,input_initializer, output_initializer = None):
        if output_initializer:
            activation_output_node_name = output_initializer
        else:
            activation_output_node_name = self.layer_name + "_output"

        op_type = onnx_activation_functions[self.activation_name]

        activation_node = onnx.helper.make_node(
            name=self.layer_name,
            op_type=op_type,
            inputs=[input_initializer],
            outputs=[activation_output_node_name]
        )

        return activation_node, activation_output_node_name, []


class Conv2D(Layer):

    def __init__(self, n_filters, filter_shape, input_shape=None, padding='same', stride=1):
        self.n_filters = n_filters
        self.filter_shape = filter_shape
        self.padding = padding
        self.stride = stride
        self.input_shape = input_shape
        self.trainable = True
        self.W = None
        self.w0 = None

    def initialize(self, optimizer):
        # Initialize the weights
        filter_height, filter_width = self.filter_shape
        channels = self.input_shape[0]
        limit = R.div(g.one, R.square_root(R.prod(R.t(self.filter_shape))))

        self.W = R.random_uniform(R.neg(limit), limit, size=(self.n_filters, channels, filter_height, filter_width))
        self.w0 = R.t(np.zeros((self.n_filters, 1)))

        # equivalent for summary params
        np_limit = 1 / math.sqrt(np.prod(self.filter_shape))
        self.np_W  = np.random.uniform(-np_limit, np_limit, size=(self.n_filters, channels, filter_height, filter_width))
        self.np_w0 = np.zeros((self.n_filters, 1))

        # Weight optimizers
        self.W_opt  = copy.copy(optimizer)
        self.w0_opt = copy.copy(optimizer)

    def parameters(self):
        return np.prod(self.np_W.shape) + np.prod(self.np_w0.shape)

    def forward_pass(self, X, training=True):
        batch_size = R.shape(X).index(indices='[0]')#X().shape
        self.layer_input = X
        # Turn image shape into column shape
        # (enables dot product between input and weights)
        self.X_col = image_to_column(X, self.filter_shape, stride=self.stride, output_shape=self.padding)
        # Turn weights into column shape
        self.W_col = self.W.reshape(shape=(self.n_filters, -1))
        # Calculate output
        output = self.W_col.dot(self.X_col) + self.w0
        # Reshape into (n_filters, out_height, out_width, batch_size)
        output = output.reshape(shape=R.join_to_list(R.t(self.output_shape()),batch_size))
        # output = output.reshape(shape=(self.output_shape() + (batch_size, )))
        # Redistribute axises so that batch size comes first
        return output.transpose(axes=(3,0,1,2))    


    def backward_pass(self, accum_grad):
        # Reshape accumulated gradient into column shape
        accum_grad = accum_grad.transpose(axes=(1, 2, 3, 0)).reshape(shape=(self.n_filters, -1))

        if self.trainable:
            # Take dot product between column shaped accum. gradient and column shape
            # layer input to determine the gradient at the layer with respect to layer weights
            # grad_w = accum_grad.dot(R.transpose(self.X_col)).reshape(shape=list(self.W().shape))
            # grad_w = R.t(accum_grad().dot(self.X_col().T).reshape(self.W().shape))
            grad_w = accum_grad.dot(self.X_col.transpose()).reshape(shape=R.shape(self.W)) #self.W().shape)


            # The gradient with respect to bias terms is the sum similarly to in Dense layer
            grad_w0 = R.sum(accum_grad, axis=1, keepdims="True")

            # Update the layers weights
            self.W = self.W_opt.update(self.W, grad_w)
            self.w0 = self.w0_opt.update(self.w0, grad_w0)

        # Recalculate the gradient which will be propogated back to prev. layer
        accum_grad = R.transpose(self.W_col).dot(accum_grad)
        # Reshape from column shape to image shape
        accum_grad = column_to_image(accum_grad,
                                # self.layer_input().shape,
                                R.shape(self.layer_input),
                                self.filter_shape,
                                stride=self.stride,
                                output_shape=self.padding)

        return accum_grad

    def output_shape(self):
        channels, height, width = self.input_shape
        pad_h, pad_w = determine_padding(self.filter_shape, output_shape=self.padding)
        output_height = (height + np.sum(pad_h) - self.filter_shape[0]) / self.stride + 1
        output_width = (width + np.sum(pad_w) - self.filter_shape[1]) / self.stride + 1
        return self.n_filters, int(output_height), int(output_width)

    def save_weights(self):
        self.W.persist_op(self.get_layer_name() + "_W")
        self.w0.persist_op(self.get_layer_name() + "_w0")

    def fetch_onnx_params(self,input_initializer, output_initializer = None):
        W_fetched = R.fetch_persisting_op(op_name=self.get_layer_name() + "_W")
        w0_fetched = R.fetch_persisting_op(op_name=self.get_layer_name() + "_w0")

        if isinstance(W_fetched, list):
            W_fetched = np.array(W_fetched)
        if isinstance(w0_fetched, list):
            w0_fetched = np.array(w0_fetched)

        conv_W_initializer_name = self.layer_name + "_W"
        W_initializer_tensor = create_initializer_tensor(name=conv_W_initializer_name, tensor_array= W_fetched, data_type=onnx.TensorProto.FLOAT)
        conv_w0_initializer_name = self.layer_name + "_w0"
        w0_initializer_tensor = create_initializer_tensor(name=conv_w0_initializer_name, tensor_array= w0_fetched, data_type=onnx.TensorProto.FLOAT)

        if output_initializer:
            conv_output_node_name = output_initializer
        else:
            conv_output_node_name = self.layer_name + "_output"

        if self.padding == 'same':
            pad = 'SAME_UPPER'
        else:
            pad = 'VALID'

        if isinstance(self.stride, int):
            stride = [self.stride, self.stride]
        else:
            stride = np.asarray(self.stride).tolist()

        conv_node = onnx.helper.make_node(
            name=self.layer_name,
            op_type="Conv",
            inputs=[
                input_initializer, conv_W_initializer_name,
                conv_w0_initializer_name
            ],
            outputs=[conv_output_node_name],
            kernel_shape=np.asarray(self.filter_shape).tolist(),
            pads=(1,1,1,1),
            strides=stride
        )

        return conv_node, conv_output_node_name, [W_initializer_tensor, w0_initializer_tensor]


class Flatten(Layer):
    """ Turns a multidimensional matrix into two-dimensional """
    def __init__(self, input_shape=None):
        self.prev_shape = None
        self.trainable = True
        self.input_shape = input_shape

    def forward_pass(self, X, training=True):
        # self.prev_shape = X().shape
        self.prev_shape = R.shape(X)
        zeroth_index = self.prev_shape.index(indices='[0]')
        new_shape = R.join_to_list(zeroth_index,R.t(-1))
        return X.reshape(shape=new_shape)
        # return X.reshape(shape=[self.prev_shape[0], -1])

    def backward_pass(self, accum_grad):
        return accum_grad.reshape(shape=self.prev_shape)

    def output_shape(self):
        return (int(np.prod(self.input_shape)),)

    def save_weights(self):
        pass

    def fetch_onnx_params(self,input_initializer, output_initializer = None):
        if output_initializer:
            flatten_output_node_name = output_initializer
        else:
            flatten_output_node_name = self.layer_name + "_output"

        flatten_node = onnx.helper.make_node(
            name=self.layer_name,  # Name is optional.
            op_type="Flatten",
            inputs=[
                input_initializer
            ],
            outputs=[flatten_output_node_name],
        )

        return flatten_node, flatten_output_node_name, []

class PoolingLayer(Layer):
    """A parent class of MaxPooling2D and AveragePooling2D
    """
    def __init__(self, pool_shape=(2, 2), stride=1, padding="same"):
        self.pool_shape = pool_shape
        self.stride = stride
        self.padding = padding
        self.trainable = True

    def forward_pass(self, X, training=True):
        self.layer_input = X

        # batch_size, channels, height, width = X().shape
        X_shape = R.shape(X)
        batch_size = X_shape.index(indices='[0]')
        channels = X_shape.index(indices='[1]')
        height = X_shape.index(indices='[2]')
        width = X_shape.index(indices='[3]')

        _, out_height, out_width = self.output_shape()

        X_reshape = R.join_to_list(batch_size*channels,R.t(1))
        X_reshape = R.join_to_list(X_reshape,height)
        X_reshape = R.join_to_list(X_reshape,width)

        X = X.reshape(shape=X_reshape)
        X_col = image_to_column(X, self.pool_shape, stride=self.stride, output_shape=self.padding)

        # MaxPool or AveragePool specific method
        output = self._pool_forward(X_col)

        output_reshape = R.t((out_height,out_width))
        output_reshape = R.join_to_list(output_reshape,batch_size)
        output_reshape = R.join_to_list(output_reshape, channels)

        output = output.reshape(shape=output_reshape)
        output = output.transpose(axes=(2, 3, 0, 1))

        return output

    def backward_pass(self, accum_grad):
        # batch_size, _, _, _ = accum_grad().shape
        accum_grad_shape = R.shape(accum_grad)
        batch_size = accum_grad_shape.index(indices='[0]')

        channels, height, width = self.input_shape
        accum_grad = accum_grad.transpose(axes=(2, 3, 0, 1)).ravel()

        # MaxPool or AveragePool specific method
        accum_grad_col = self._pool_backward(accum_grad)

        images_shape = R.join_to_list(batch_size*R.t(channels),R.t(1))
        images_shape = R.join_to_list(images_shape,R.t(height))
        images_shape = R.join_to_list(images_shape,R.t(width))

        accum_grad = column_to_image(accum_grad_col, images_shape, self.pool_shape, self.stride, 'same')
        accum_grad_shape1 = R.join_to_list(batch_size,R.t(self.input_shape))
        # accum_grad = accum_grad.reshape(shape=((batch_size,) + self.input_shape))
        accum_grad = accum_grad.reshape(shape=accum_grad_shape1)

        return accum_grad

    def output_shape(self):
        channels, height, width = self.input_shape
        out_height = (height - self.pool_shape[0]) / self.stride + 1
        out_width = (width - self.pool_shape[1]) / self.stride + 1
        assert out_height % 1 == 0
        assert out_width % 1 == 0
        return channels, int(out_height), int(out_width)


class MaxPooling2D(PoolingLayer):
    def _pool_forward(self, X_col):
        arg_max = R.argmax(X_col, axis=0).flatten()
        
        index = R.combine_to_list(arg_max, R.arange(R.size(arg_max)))
        output = R.index(X_col, indices=index)     

        # output = X_col()[arg_max, range(arg_max.size)]
        self.cache = arg_max
        return output #R.t(output)

    def _pool_backward(self, accum_grad):
        zeros_shape = R.join_to_list(R.prod(R.t(self.pool_shape)), R.size(accum_grad))
        accum_grad_col = R.zeros(zeros_shape)
        # accum_grad_col = np.zeros((np.prod(self.pool_shape), accum_grad().size))
        arg_max = self.cache
        index = R.combine_to_list(arg_max, R.arange(R.size(accum_grad)))
        accum_grad_col = R.set_value(accum_grad_col,accum_grad,indices=index)
        # accum_grad_col[arg_max, range(accum_grad().size)] = accum_grad()
        return accum_grad_col #R.t(accum_grad_col)

    def save_weights(self):
        pass

    def fetch_onnx_params(self,input_initializer, output_initializer = None):
        if output_initializer:
            maxpool2d_output_node_name = output_initializer
        else:
            maxpool2d_output_node_name = self.layer_name + "_output"

        if isinstance(self.stride, int):
            stride = [self.stride, self.stride]
        else:
            stride = np.asarray(self.stride).tolist()

        maxpool2d_node = onnx.helper.make_node(
            name=self.layer_name,  # Name is optional.
            op_type="MaxPool",
            inputs=[
                input_initializer
            ],
            outputs=[maxpool2d_output_node_name],
            kernel_shape=np.asarray(self.pool_shape).tolist(),
            strides=stride,
            # pads=self.padding,
        )

        return maxpool2d_node, maxpool2d_output_node_name, []


def image_to_column(images, filter_shape, stride, output_shape='same'):
    filter_height, filter_width = filter_shape

    pad_h, pad_w = determine_padding(filter_shape, output_shape)

    # Add padding to the image
    # images_padded = np.pad(images(), ((0, 0), (0, 0), pad_h, pad_w), mode='constant')
    images_padded = R.pad(images, sequence=((0, 0), (0, 0), pad_h, pad_w), mode='constant')

    # Calculate the indices where the dot products are to be applied between weights
    # and the image
    # k, i, j = get_im2col_indices(images().shape, filter_shape, (pad_h, pad_w), stride)
    k, i, j = get_im2col_indices(R.shape(images), filter_shape, (pad_h, pad_w), stride)

    # Get content from image at those indices
    # cols = images_padded[:, k, i, j]
    # cols = R.index(images_padded, indices="[:, {}, {}, {}]".format(k.tolist(),i.tolist(),j.tolist()))
    cols = R.cnn_index(images_padded,index1=k,index2=i,index3=j)
    # channels = images().shape[1]
    channels = R.shape(images).index(indices='[1]')
    product = R.t(filter_height * filter_width) * channels
    cols_shape = R.join_to_list(product,R.t(-1))
    # Reshape content into column shape
    cols = cols.transpose(axes=(1, 2, 0)).reshape(shape=cols_shape)
    return cols

def determine_padding(filter_shape, output_shape="same"):

    # No padding
    if output_shape == "valid":
        return (0, 0), (0, 0)
    # Pad so that the output shape is the same as input shape (given that stride=1)
    elif output_shape == "same":
        filter_height, filter_width = filter_shape
        # Derived from:
        # output_height = (height + pad_h - filter_height) / stride + 1
        # In this case output_height = height and stride = 1. This gives the
        # expression for the padding below.
        pad_h1 = int(math.floor((filter_height - 1)/2))
        pad_h2 = int(math.ceil((filter_height - 1)/2))
        pad_w1 = int(math.floor((filter_width - 1)/2))
        pad_w2 = int(math.ceil((filter_width - 1)/2))

        return (pad_h1, pad_h2), (pad_w1, pad_w2)


# Reference: CS231n Stanford
def get_im2col_indices(images_shape, filter_shape, padding, stride=1):
    # First figure out what the size of the output should be
    # batch_size, channels, height, width = images_shape
    batch_size = R.index(images_shape, indices="[0]")
    channels = R.index(images_shape, indices="[1]")
    height = R.index(images_shape, indices="[2]")
    width = R.index(images_shape, indices="[3]")

    filter_height, filter_width = filter_shape
    filter_height = R.t(filter_height)
    filter_width = R.t(filter_width)

    pad_h, pad_w = padding
    # out_height = int((height() + np.sum(pad_h) - filter_height) / stride + 1)
    out_height = R.ravint(R.div(height + R.t(np.sum(pad_h)) - filter_height,R.t(stride))+R.t(1))
    # out_width = int((width() + np.sum(pad_w) - filter_width) / stride + 1)
    out_width = R.ravint(R.div(width + R.t(np.sum(pad_w)) - filter_width,R.t(stride))+R.t(1))

    # i0 = np.repeat(np.arange(filter_height), filter_width)
    i0 = R.repeat(R.arange(filter_height), repeats=filter_width)
    # i0 = np.tile(i0, channels)
    i0 = R.tile(i0, reps=channels) #int(channels()))
    # i1 = stride * np.repeat(np.arange(out_height), out_width)
    i1 = R.t(stride) * R.repeat(R.arange(out_height), repeats=out_width)
    # j0 = np.tile(np.arange(filter_width), filter_height * channels)
    j0 = R.tile(R.arange(filter_width), reps= filter_height * channels) #int(channels()))
    # j1 = stride * np.tile(np.arange(out_width), out_height)
    j1 = R.t(stride) * R.tile(R.arange(out_width), reps = out_height)
    # i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    i = R.reshape(i0, shape=(-1,1)) + R.reshape(i1, shape=(1,-1))
    # j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    j = R.reshape(j0, shape=(-1,1)) + R.reshape(j1, shape=(1,-1))

    # k = np.repeat(np.arange(channels), filter_height * filter_width).reshape(-1, 1)
    k = R.reshape(R.repeat(R.arange(channels), repeats = filter_height * filter_width), shape=(-1,1))

    return (k, i, j)

def column_to_image(cols, images_shape, filter_shape, stride, output_shape='same'):
    # batch_size, channels, height, width = images_shape()
    batch_size = R.index(images_shape, indices="[0]")
    channels = R.index(images_shape, indices="[1]")
    height = R.index(images_shape, indices="[2]")
    width = R.index(images_shape, indices="[3]")

    pad_h, pad_w = determine_padding(filter_shape, output_shape)
    height_padded = height + R.t(np.sum(pad_h))
    width_padded = width + R.t(np.sum(pad_w))

    zeros_param = R.join_to_list(batch_size,channels)
    zeros_param = R.join_to_list(zeros_param,height_padded)
    zeros_param = R.join_to_list(zeros_param,width_padded)

    # images_padded = np.zeros((batch_size, channels, height_padded, width_padded))
    images_padded = R.zeros(zeros_param)

    # Calculate the indices where the dot products are applied between weights
    # and the image
    k, i, j = get_im2col_indices(images_shape, filter_shape, (pad_h, pad_w), stride)

    cols_reshape = R.join_to_list(channels * R.t(np.prod(filter_shape)), R.t(-1))
    cols_reshape = R.join_to_list(cols_reshape,batch_size)

    cols = cols.reshape(shape=cols_reshape)

    # cols = cols.reshape(shape=(channels * np.prod(filter_shape), -1, batch_size))
    cols = cols.transpose(axes=(2, 0, 1))
    # Add column content to the images at the indices
    
    images_padded = R.cnn_add_at(images_padded, cols, index1=k, index2=i, index3=j)
    
    # np.add.at(images_padded, (slice(None), k, i, j), cols())

    return R.cnn_index_2(images_padded, pad_h=R.t(pad_h[0]), height=height, pad_w=R.t(pad_w[0]), width=width)

    # Return image without padding
    # return R.t(images_padded[:, :, pad_h[0]:height+pad_h[0], pad_w[0]:width+pad_w[0]])
