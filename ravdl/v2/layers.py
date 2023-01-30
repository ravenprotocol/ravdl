import ravop as R
import numpy as np
import math
from .utils.data_operations import determine_padding


class CustomLayer():
    def __init__(self) -> None:
        self.graph = {}
        self.layer_type = []
        self.layer_name = None
        self.graph_params = {}
        self.forward_pass = None
        self.backward_pass = None
        self.grads_list = []
        self.last_layer_name = None
        self.grad_dict = {}

    def initialize(self, optimizer):
        for attr in self.__dict__:
            if isinstance(self.__dict__[attr], Layer) or isinstance(self.__dict__[attr], CustomLayer):
                layer = self.__dict__[attr]
                layer_name=layer.__class__.__name__
                self.layer_type.append(layer_name)
                if layer_name in self.layer_type:
                    layer_name+="_"+str(self.layer_type.count(layer.__class__.__name__)) 
                if "Activation" in layer_name:
                    layer_name += "_"+layer.activation_name                
                layer_name = self.layer_name + "_" + layer_name
                layer.set_layer_name(layer_name)
                if hasattr(layer, 'initialize'):
                    layer.initialize(optimizer)
            elif isinstance(self.__dict__[attr],list):
                for item in self.__dict__[attr]:
                    if isinstance(item,CustomLayer) or isinstance(item,Layer):
                        layer = item
                        layer_name=layer.__class__.__name__
                        self.layer_type.append(layer_name)
                        if layer_name in self.layer_type:
                            layer_name+="_"+str(self.layer_type.count(layer.__class__.__name__)) 
                        if "Activation" in layer_name:
                            layer_name += "_"+layer.activation_name
                        layer_name = self.layer_name + "_" + layer_name
                        layer.set_layer_name(layer_name)
                        if hasattr(layer, 'initialize'):
                            layer.initialize(optimizer)

    def print_layer_names(self):
        for attr in self.__dict__:
            if "ravdl.v2.layers" in str(type(self.__dict__[attr])):
                print(self.__dict__[attr].layer_name)

    def set_layer_name(self, layer_name):
        self.layer_name = layer_name

    def set_graph(self):
        for attr in self.__dict__:
            if isinstance(self.__dict__[attr], Layer) or isinstance(self.__dict__[attr], CustomLayer):
                self.graph = {**self.graph,**self.__dict__[attr].graph_params}
                if hasattr(self.__dict__[attr], 'set_graph'):
                    self.__dict__[attr].set_graph()

            elif isinstance(self.__dict__[attr],list):
                for item in self.__dict__[attr]:
                    if isinstance(item,CustomLayer) or isinstance(item,Layer):
                        self.graph = {**self.graph,**item.graph_params}
                        if hasattr(item, 'set_graph'):
                            item.set_graph()

    def _forward_pass(self,*args, training=True, **kwargs):
        inputs_list = []
        for X in args:
            if isinstance(X, dict):
                prev_layer_name = X['name']
                X = X['output']
                if self.backward_pass is None:
                    inputs_list.append(prev_layer_name)
            else:
                if self.backward_pass is None:
                    inputs_list.append(X)
        
        if self.backward_pass is None:
            self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': inputs_list}
        
        self.forward_pass = self._forward_pass_call(*args, training=training, **kwargs)
        self.last_layer_name = self.forward_pass['name']
        return {'output':self.forward_pass['output'], 'name':self.layer_name}
        

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

    def output_shape(self):
        """ The shape of the output produced by forward_pass """
        raise NotImplementedError()


class Dense(Layer):
    def __init__(self, n_units, input_shape=None, initial_W=None, initial_w0=None, use_bias='True'):
        self.forward_pass = None
        self.backward_pass = None
        self.n_units = n_units
        self.layer_input = None
        self.input_shape = input_shape
        self.trainable = True
        self.graph_params = {}
        self.initial_W = initial_W
        self.initial_w0 = initial_w0
        self.use_bias = use_bias

    def initialize(self, optimizer):
        self.optimizer = optimizer
        if self.input_shape is not None:
            np_limit = 1 / math.sqrt(self.input_shape[0])
            self.np_W  = np.random.uniform(-np_limit, np_limit, (self.input_shape[0], self.n_units))
            self.np_w0 = np.zeros((1, self.n_units))

    def parameters(self): 
        return np.prod(self.np_W.shape) + np.prod(self.np_w0.shape)

    def _forward_pass(self, X, input_layer = "False", training=True):
        if isinstance(X, dict):
            prev_layer_name = X['name']
            X = X['output']
            if self.backward_pass is None:
                self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': [prev_layer_name]}
        else:
            if self.backward_pass is None:
                self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': [X]}
        self.layer_input = X
        if self.initial_W is not None or self.initial_w0 is not None:
            self.forward_pass = R.forward_pass_dense(
                                X, n_units=self.n_units,
                                input_layer = input_layer,
                                optimizer_dict = self.optimizer.data_dict(),
                                initial_W = self.initial_W,
                                initial_w0 = self.initial_w0, 
                                )    
        else:
            self.forward_pass = R.forward_pass_dense(
                                X, n_units=self.n_units,
                                input_layer = input_layer,
                                optimizer_dict = self.optimizer.data_dict(),
                                previous_forward_pass = self.forward_pass
                                ) 
        return {'output':self.forward_pass, 'name':self.layer_name}

    def output_shape(self):
        return (self.n_units, )

    def persist_weights(self):
        self.forward_pass.persist_op("{}_forward_pass".format(self.layer_name))
        self.backward_pass.persist_op("{}_backward_pass".format(self.layer_name))

class BatchNormalization1D(Layer):
    """Batch Normalization 1D
    """
    def __init__(self, momentum=0.99, epsilon=0.01, affine=True, input_shape=None, initial_gamma=None, initial_beta=None, initial_running_mean=None, initial_running_var=None):
        self.momentum = momentum
        self.float_momentum = momentum
        self.trainable = True
        self.eps = epsilon
        self.affine = str(affine)
        self.float_eps = epsilon
        self.running_mean = None
        self.running_var = None
        self.forward_pass = None
        self.backward_pass = None
        self.input_shape = input_shape
        self.graph_params = {}
        self.initial_gamma = initial_gamma
        self.initial_beta = initial_beta
        self.initial_running_mean = initial_running_mean
        self.initial_running_var = initial_running_var

    def initialize(self, optimizer):
        self.optimizer = optimizer
        if self.input_shape is not None:
            if len(self.input_shape) == 1:
                shape = (1, self.input_shape[0])
            else:
                shape = (1,self.input_shape[0], 1,1)
            self.np_gamma = np.ones(shape)
            self.np_beta = np.zeros(shape)

    def parameters(self):
        return np.prod(self.np_gamma.shape) + np.prod(self.np_beta.shape)        

    def _forward_pass(self, X, input_layer="False", training=True):
        if training:
            training="True"
        else:
            training = "False"
        if isinstance(X, dict):
            prev_layer_name = X['name']
            X = X['output']
            if self.backward_pass is None:
                self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': [prev_layer_name]}
        else:
            if self.backward_pass is None:
                self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': [X]}
        if self.initial_gamma is not None or self.initial_beta is not None or self.initial_running_mean is not None or self.initial_running_var is not None:
            self.forward_pass = R.forward_pass_batchnorm1d(
                                X, 
                                momentum = self.momentum, eps = self.eps,
                                affine=self.affine,
                                training=training,
                                optimizer_dict = self.optimizer.data_dict(),
                                initial_gamma = self.initial_gamma,
                                initial_beta = self.initial_beta,
                                initial_running_mean = self.initial_running_mean,
                                initial_running_var = self.initial_running_var,
                                )
        else:
            self.forward_pass = R.forward_pass_batchnorm1d(
                                X, 
                                momentum = self.momentum, eps = self.eps,
                                affine=self.affine,
                                training=training,
                                optimizer_dict = self.optimizer.data_dict(),
                                previous_forward_pass = self.forward_pass
                                )       
        
        return {'output':self.forward_pass, 'name':self.layer_name}

    def output_shape(self):
        return self.input_shape

    def persist_weights(self):
        self.forward_pass.persist_op("{}_forward_pass".format(self.layer_name))
        self.backward_pass.persist_op("{}_backward_pass".format(self.layer_name))

class BatchNormalization2D(Layer):
    """Batch Normalization 2D
    """
    def __init__(self, num_features, momentum=0.99, epsilon=0.01, affine=True, input_shape=None, initial_gamma=None, initial_beta=None, initial_running_mean=None, initial_running_var=None):
        self.num_features = num_features
        self.momentum = momentum
        self.float_momentum = momentum
        self.trainable = True
        self.eps = epsilon
        self.affine = str(affine)
        self.float_eps = epsilon
        self.running_mean = None
        self.running_var = None
        self.forward_pass = None
        self.backward_pass = None
        self.input_shape = input_shape
        self.graph_params = {}
        self.initial_gamma = initial_gamma
        self.initial_beta = initial_beta
        self.initial_running_mean = initial_running_mean
        self.initial_running_var = initial_running_var

    def initialize(self, optimizer):
        self.optimizer = optimizer
        if self.input_shape is not None:
            if len(self.input_shape) == 1:
                shape = (1, self.input_shape[0])
            else:
                shape = (1,self.input_shape[0], 1,1)
            self.np_gamma = np.ones(shape)
            self.np_beta = np.zeros(shape)

    def parameters(self):
        return np.prod(self.np_gamma.shape) + np.prod(self.np_beta.shape)        

    def _forward_pass(self, X, input_layer="False", training=True):
        if training:
            training="True"
        else:
            training = "False"
        if isinstance(X, dict):
            prev_layer_name = X['name']
            X = X['output']
            if self.backward_pass is None:
                self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': [prev_layer_name]}
        else:
            if self.backward_pass is None:
                self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': [X]}
        if self.initial_gamma is not None or self.initial_beta is not None or self.initial_running_mean is not None or self.initial_running_var is not None:
            self.forward_pass = R.forward_pass_batchnorm2d(
                                X,
                                num_features = self.num_features, 
                                momentum = self.momentum, eps = self.eps,
                                affine=self.affine,
                                training=training,
                                optimizer_dict = self.optimizer.data_dict(),
                                initial_gamma = self.initial_gamma,
                                initial_beta = self.initial_beta,
                                initial_running_mean = self.initial_running_mean,
                                initial_running_var = self.initial_running_var,
                                )    
        else:
            self.forward_pass = R.forward_pass_batchnorm2d(
                                X,
                                num_features = self.num_features,
                                momentum = self.momentum, eps = self.eps,
                                affine=self.affine,
                                training=training, 
                                optimizer_dict = self.optimizer.data_dict(),
                                previous_forward_pass = self.forward_pass
                                )
        return {'output':self.forward_pass, 'name':self.layer_name}

    def output_shape(self):
        return self.input_shape

    def persist_weights(self):
        self.forward_pass.persist_op("{}_forward_pass".format(self.layer_name))
        self.backward_pass.persist_op("{}_backward_pass".format(self.layer_name))


class LayerNormalization(Layer):
    """Batch normalization.
    """
    def __init__(self, normalized_shape=None, epsilon=1e-5, input_shape=None, initial_W=None, initial_w0=None):
        self.normalized_shape = normalized_shape
        self.trainable = True
        self.eps = epsilon
        self.float_eps = epsilon
        self.running_mean = None
        self.running_var = None
        self.forward_pass = None
        self.backward_pass = None
        self.input_shape = input_shape
        self.graph_params = {}
        self.initial_W = initial_W
        self.initial_w0 = initial_w0

    def initialize(self, optimizer):
        self.optimizer = optimizer
        if self.input_shape is not None:
            if len(self.input_shape) == 1:
                shape = (1, self.input_shape[0])
            else:
                shape = (1,self.input_shape[0], 1,1)
            self.np_gamma = np.ones(shape)
            self.np_beta = np.zeros(shape)

    def parameters(self):
        return np.prod(self.np_gamma.shape) + np.prod(self.np_beta.shape)        

    def _forward_pass(self, X, input_layer="False", training=True):
        if training:
            training="True"
        else:
            training = "False"
        if isinstance(X, dict):
            prev_layer_name = X['name']
            X = X['output']
            if self.backward_pass is None:
                self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': [prev_layer_name]}
        else:
            if self.backward_pass is None:
                self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': [X]}
        if self.initial_W is not None or self.initial_w0 is not None:
            self.forward_pass = R.forward_pass_layernorm(
                                X, normalized_shape=self.normalized_shape,
                                input_shape=self.input_shape,
                                eps = self.eps,
                                training=training, 
                                optimizer_dict = self.optimizer.data_dict(),
                                initial_W = self.initial_W,
                                initial_w0 = self.initial_w0,
                                )
        else:
            self.forward_pass = R.forward_pass_layernorm(
                                X, normalized_shape=self.normalized_shape,
                                input_shape=self.input_shape,
                                eps = self.eps,
                                training=training, 
                                optimizer_dict = self.optimizer.data_dict(),
                                previous_forward_pass = self.forward_pass
                                )
        return {'output':self.forward_pass, 'name':self.layer_name}

    def output_shape(self):
        return self.input_shape

    def persist_weights(self):
        self.forward_pass.persist_op("{}_forward_pass".format(self.layer_name))
        self.backward_pass.persist_op("{}_backward_pass".format(self.layer_name))


class Dropout(Layer):
    """A layer that randomly sets a fraction p of the output units of the previous layer
    to zero.

    Parameters:
    -----------
    p: float
        The probability that unit x is set to zero.
    """
    def __init__(self, p=0.5):
        self.p = p
        self._mask = None
        self.input_shape = None
        self.n_units = None
        self.pass_through = True
        self.trainable = True
        self.forward_pass = None
        self.backward_pass = None
        self.graph_params = {}

    def _forward_pass(self, X, training=True, input_layer="False"):
        if training:
            training = "True"
        else:
            training = "False"
        if isinstance(X, dict):
            prev_layer_name = X['name']
            X = X['output']
            if self.backward_pass is None:
                self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': [prev_layer_name]}
        else:
            if self.backward_pass is None:
                self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': [X]}
        self.forward_pass = R.forward_pass_dropout(X, p = self.p, training=training)
        return {'output':self.forward_pass, 'name':self.layer_name}

    def output_shape(self):
        return self.input_shape

    def persist_weights(self):
        pass

    def save_weights(self):
        self.p.persist_op(self.get_layer_name() + "_p")


class Activation(Layer):
    """A layer that applies an activation operation to the input.

    Parameters:
    -----------
    name: string
        The name of the activation function that will be used.
    """

    def __init__(self, name):
        self.activation_name = name
        self.trainable = True
        self.forward_pass = None
        self.backward_pass = None
        self.graph_params = {}
        
    def _forward_pass(self, X, training=True, input_layer = "False"):
        if isinstance(X, dict):
            prev_layer_name = X['name']
            X = X['output']
            if self.backward_pass is None:
                self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': [prev_layer_name]}
        else:
            if self.backward_pass is None:
                self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': [X]}
        self.layer_input = X
        self.forward_pass = R.forward_pass_activation(X, act_data = str({'name':self.activation_name}))
        return {'output':self.forward_pass, 'name':self.layer_name}

    def output_shape(self):
        return self.input_shape

    def persist_weights(self):
        pass

class Conv2D(Layer):
    """A 2D convolutional layer.

    Parameters:
    -----------
    n_filters: int
        The number of filters that the layer will learn.
    filter_shape: tuple
        The shape of the filters that the layer will learn.
    stride: int
        The stride of the convolution operation.
    padding: string
        The padding that will be applied to the input.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', initial_W=None, initial_w0=None, input_shape=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = str(bias)
        self.padding_mode = padding_mode
        self.initial_W = initial_W
        self.initial_w0 = initial_w0
        self.trainable = True
        self.forward_pass = None
        self.backward_pass = None
        self.input_shape = input_shape
        self.graph_params = {}

    def initialize(self, optimizer):
        self.optimizer = optimizer
        if self.input_shape is not None:
            filter_height, filter_width = self.kernel_size
            channels = self.input_shape[0]
            limit = 1 / math.sqrt(np.prod(self.kernel_size))
            self.W  = np.random.uniform(-limit, limit, size=(self.out_channels, channels, filter_height, filter_width))
            self.w0 = np.zeros((self.out_channels, 1))

    def parameters(self):
        return  np.prod(self.W.shape) + np.prod(self.w0.shape)

    def _forward_pass(self, X, input_layer="False", training=True):
        if training:
            training="True"
        else:
            training = "False"
        if isinstance(X, dict):
            prev_layer_name = X['name']
            X = X['output']
            if self.backward_pass is None:
                self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': [prev_layer_name]}
        else:
            if self.backward_pass is None:
                self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': [X]}
        if self.initial_W is not None or self.initial_w0 is not None:
            self.forward_pass = R.forward_pass_conv2d(
                                X, 
                                in_channels = self.in_channels,
                                out_channels = self.out_channels,
                                kernel_size = self.kernel_size,
                                stride = self.stride,
                                padding = self.padding,
                                dilation = self.dilation,
                                groups = self.groups,
                                bias = self.bias,
                                padding_mode = self.padding_mode,
                                training=training,
                                optimizer_dict = self.optimizer.data_dict(),
                                initial_W = self.initial_W,
                                initial_w0 = self.initial_w0,
                                )    
        else:
            self.forward_pass = R.forward_pass_conv2d(
                                X, 
                                in_channels = self.in_channels,
                                out_channels = self.out_channels,
                                kernel_size = self.kernel_size,
                                stride = self.stride,
                                padding = self.padding,
                                dilation = self.dilation,
                                groups = self.groups,
                                bias = self.bias,
                                padding_mode = self.padding_mode,
                                training=training,
                                optimizer_dict = self.optimizer.data_dict(),
                                previous_forward_pass = self.forward_pass
                                ) 
        return {'output':self.forward_pass, 'name':self.layer_name}

    def output_shape(self):
        channels, height, width = self.input_shape
        pad_h, pad_w = determine_padding(self.kernel_size, output_shape=self.padding)
        output_height = (height + np.sum(pad_h) - self.kernel_size[0]) / self.stride + 1
        output_width = (width + np.sum(pad_w) - self.kernel_size[1]) / self.stride + 1
        return self.out_channels, int(output_height), int(output_width)

    
class Flatten(Layer):
    def __init__(self, start_dim=1, end_dim=-1):
        self.start_dim = start_dim
        self.end_dim = end_dim
        self.trainable = True
        self.forward_pass = None
        self.backward_pass = None
        self.graph_params = {}

    def _forward_pass(self, X, training=True, input_layer = "False"):
        if isinstance(X, dict):
            prev_layer_name = X['name']
            X = X['output']
            if self.backward_pass is None:
                self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': [prev_layer_name]}
        else:
            if self.backward_pass is None:
                self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': [X]}
        self.forward_pass = R.forward_pass_flatten(X, start_dim = self.start_dim, end_dim = self.end_dim)
        return {'output':self.forward_pass, 'name':self.layer_name}
    
    def output_shape(self):
        return (np.prod(self.input_shape),)

class MaxPooling2D(Layer):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = str(return_indices)
        self.ceil_mode = str(ceil_mode)
        self.trainable = True
        self.forward_pass = None
        self.backward_pass = None
        self.graph_params = {}

    def initialize(self, optimizer):
        self.optimizer = optimizer

    
    def _forward_pass(self, X, input_layer="False", training=True):
        if training:
            training="True"
        else:
            training = "False"
        if isinstance(X, dict):
            prev_layer_name = X['name']
            X = X['output']
            if self.backward_pass is None:
                self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': [prev_layer_name]}
        else:
            if self.backward_pass is None:
                self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': [X]}
        
        self.forward_pass = R.forward_pass_maxpool2d(
                            X, 
                            kernel_size = self.kernel_size,
                            stride = self.stride,
                            padding = self.padding,
                            dilation = self.dilation,
                            return_indices = self.return_indices,
                            ceil_mode = self.ceil_mode,
                            training=training,
                            optimizer_dict = self.optimizer.data_dict(),
                            # previous_forward_pass = self.forward_pass
                            )    

        return {'output':self.forward_pass, 'name':self.layer_name}

    def output_shape(self):
        channels, height, width = self.input_shape
        out_height = (height - self.kernel_size[0]) / self.stride + 1
        out_width = (width - self.kernel_size[1]) / self.stride + 1
        assert out_height % 1 == 0
        assert out_width % 1 == 0
        return channels, int(out_height), int(out_width)


'''------------------- Math Op Layers -----------------'''

class Concat(Layer):
    def __init__(self):
        self.forward_pass = None
        self.backward_pass = None
        self.graph_params = {}

    def _forward_pass(self, x1, x2, axis=-1, input_layer="False", training=True):
        inputs_list = []
        if isinstance(x1, dict):
            prev_layer_name = x1['name']
            x1 = x1['output']
            if self.backward_pass is None:
                inputs_list.append(prev_layer_name)
        else:
            if self.backward_pass is None:
                inputs_list.append(x1)

        if isinstance(x2, dict):
            prev_layer_name = x2['name']
            x2 = x2['output']
            if self.backward_pass is None:
                inputs_list.append(prev_layer_name)
        else:
            if self.backward_pass is None:
                inputs_list.append(x2)
        if self.backward_pass is None:
            self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': inputs_list}
        self.x1 = x1
        self.x2 = x2
        self.axis = axis
        self.forward_pass = R.forward_pass_concat(x1,x2,axis=axis,input_layer = input_layer)
        return {'output':self.forward_pass, 'name':self.layer_name}


class Add(Layer):
    def __init__(self):
        self.forward_pass = None
        self.backward_pass = None
        self.graph_params = {}

    def _forward_pass(self, x1, x2, input_layer="False", training=True):
        inputs_list = []
        if isinstance(x1, dict):
            prev_layer_name = x1['name']
            x1 = x1['output']
            if self.backward_pass is None:
                inputs_list.append(prev_layer_name)
        else:
            if self.backward_pass is None:
                inputs_list.append(x1)

        if isinstance(x2, dict):
            prev_layer_name = x2['name']
            x2 = x2['output']
            if self.backward_pass is None:
                inputs_list.append(prev_layer_name)
        else:
            if self.backward_pass is None:
                inputs_list.append(x2)
        if self.backward_pass is None:
            self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': inputs_list}
        
        self.forward_pass = R.forward_pass_add(x1,x2,input_layer = input_layer)
        return {'output':self.forward_pass, 'name':self.layer_name}


class Subtract(Layer):
    def __init__(self):
        self.forward_pass = None
        self.backward_pass = None
        self.graph_params = {}

    def _forward_pass(self, x1, x2, input_layer="False", training=True):
        inputs_list = []
        if isinstance(x1, dict):
            prev_layer_name = x1['name']
            x1 = x1['output']
            if self.backward_pass is None:
                inputs_list.append(prev_layer_name)
        else:
            if self.backward_pass is None:
                inputs_list.append(x1)

        if isinstance(x2, dict):
            prev_layer_name = x2['name']
            x2 = x2['output']
            if self.backward_pass is None:
                inputs_list.append(prev_layer_name)
        else:
            if self.backward_pass is None:
                inputs_list.append(x2)
        if self.backward_pass is None:
            self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': inputs_list}
        
        self.forward_pass = R.forward_pass_subtract(x1,x2,input_layer = input_layer)
        return {'output':self.forward_pass, 'name':self.layer_name}


class Dot(Layer):
    def __init__(self):
        self.forward_pass = None
        self.backward_pass = None
        self.graph_params = {}

    def _forward_pass(self, x1, x2, input_layer="False", training=True):
        inputs_list = []
        if isinstance(x1, dict):
            prev_layer_name = x1['name']
            x1 = x1['output']
            if self.backward_pass is None:
                inputs_list.append(prev_layer_name)
        else:
            if self.backward_pass is None:
                inputs_list.append(x1)
        if isinstance(x2, dict):
            prev_layer_name = x2['name']
            x2 = x2['output']
            if self.backward_pass is None:
                inputs_list.append(prev_layer_name)
        else:
            if self.backward_pass is None:
                inputs_list.append(x2)
        if self.backward_pass is None:
            self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': inputs_list}
        self.x1 = x1
        self.x2 = x2
        self.forward_pass = R.forward_pass_dot(x1,x2,input_layer = input_layer)
        return {'output':self.forward_pass, 'name':self.layer_name}


class Reshape(Layer):
    def __init__(self, contiguous="False"):
        self.forward_pass = None
        self.backward_pass = None
        self.graph_params = {}
        self.contiguous = contiguous

    def _forward_pass(self, X, shape=None, input_layer="False", training=True):
        if isinstance(X, dict):
            prev_layer_name = X['name']
            X = X['output']
            if self.backward_pass is None:
                self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': [prev_layer_name]}
        else:
            if self.backward_pass is None:
                self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': [X]}
        self.layer_input = X
        self.forward_pass = R.forward_pass_reshape(
            X, contiguous=self.contiguous, shape=shape, input_layer = input_layer
        )
        return {'output':self.forward_pass, 'name':self.layer_name}


class Transpose(Layer):
    def __init__(self):
        self.forward_pass = None
        self.backward_pass = None
        self.graph_params = {}

    def _forward_pass(self, X, axes=None, input_layer="False", training=True):
        if isinstance(X, dict):
            prev_layer_name = X['name']
            X = X['output']
            if self.backward_pass is None:
                self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': [prev_layer_name]}
        else:
            if self.backward_pass is None:
                self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': [X]}
        self.layer_input = X
        self.axes = axes
        self.forward_pass = R.forward_pass_transpose(
            X, axes=axes, input_layer = input_layer
        )
        return {'output':self.forward_pass, 'name':self.layer_name}


class Power(Layer):
    def __init__(self):
        self.forward_pass = None
        self.backward_pass = None
        self.graph_params = {}

    def _forward_pass(self, X, power=None, input_layer="False", training=True):
        if isinstance(X, dict):
            prev_layer_name = X['name']
            X = X['output']
            if self.backward_pass is None:
                self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': [prev_layer_name]}
        else:
            if self.backward_pass is None:
                self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': [X]}
        self.power = power
        self.layer_input = X
        self.forward_pass = R.forward_pass_power(
            X, power=power, input_layer = input_layer
        )
        return {'output':self.forward_pass, 'name':self.layer_name}


class Multiply(Layer):
    def __init__(self):
        self.forward_pass = None
        self.backward_pass = None
        self.graph_params = {}

    def _forward_pass(self, x1, x2, input_layer="False", training=True):
        inputs_list = []
        if isinstance(x1, dict):
            prev_layer_name = x1['name']
            x1 = x1['output']
            if self.backward_pass is None:
                inputs_list.append(prev_layer_name)
        else:
            if self.backward_pass is None:
                inputs_list.append(x1)
        if isinstance(x2, dict):
            prev_layer_name = x2['name']
            x2 = x2['output']
            if self.backward_pass is None:
                inputs_list.append(prev_layer_name)
        else:
            if self.backward_pass is None:
                inputs_list.append(x2)
        if self.backward_pass is None:
            self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': inputs_list}
        self.x1 = x1
        self.x2 = x2
        self.forward_pass = R.forward_pass_multiply(x1,x2,input_layer = input_layer)
        return {'output':self.forward_pass, 'name':self.layer_name}


class Division(Layer):
    def __init__(self):
        self.forward_pass = None
        self.backward_pass = None
        self.graph_params = {}

    def _forward_pass(self, x1, x2, input_layer="False", training=True):
        inputs_list = []
        if isinstance(x1, dict):
            prev_layer_name = x1['name']
            x1 = x1['output']
            if self.backward_pass is None:
                inputs_list.append(prev_layer_name)
        else:
            if self.backward_pass is None:
                inputs_list.append(x1)
        if isinstance(x2, dict):
            prev_layer_name = x2['name']
            x2 = x2['output']
            if self.backward_pass is None:
                inputs_list.append(prev_layer_name)
        else:
            if self.backward_pass is None:
                inputs_list.append(x2)
        if self.backward_pass is None:
            self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': inputs_list}
        self.x1 = x1
        self.x2 = x2
        self.forward_pass = R.forward_pass_division(x1,x2,input_layer = input_layer)
        return {'output':self.forward_pass, 'name':self.layer_name}


class Embedding(Layer):
    def __init__(self, vocab_size, embed_dim, input_shape=None, initial_W=None):
        self.forward_pass = None
        self.backward_pass = None
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.layer_input = None
        self.input_shape = input_shape
        self.trainable = True
        self.graph_params = {}
        self.initial_W = initial_W

    def initialize(self, optimizer):
        self.optimizer = optimizer
        if self.input_shape is not None:
            self.np_wrd_embed = np.random.randn(self.vocab_size, self.embed_dim) * 0.01
    
            assert(self.np_wrd_embed.shape == (self.vocab_size, self.embed_dim))

    def parameters(self): 
        return np.prod(self.np_wrd_embed.shape)

    def _forward_pass(self, X, input_layer = "False", training=True):
        if isinstance(X, dict):
            prev_layer_name = X['name']
            X = X['output']
            if self.backward_pass is None:
                self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': [prev_layer_name]}
        else:
            if self.backward_pass is None:
                self.graph_params[self.layer_name] = {'layer_instance':self, 'inputs': [X]}
        self.layer_input = X
        if self.initial_W is not None:
            self.forward_pass = R.forward_pass_embedding(
                                X, vocab_size = self.vocab_size, embed_dim = self.embed_dim, 
                                input_shape = self.input_shape, 
                                initial_weights = self.initial_W,
                                optimizer_dict = self.optimizer.data_dict(),
                                )            
        else:
            self.forward_pass = R.forward_pass_embedding(
                                X, vocab_size = self.vocab_size, embed_dim = self.embed_dim, 
                                input_shape = self.input_shape, 
                                optimizer_dict = self.optimizer.data_dict(),
                                previous_forward_pass = self.forward_pass
                                )          
        return {'output':self.forward_pass, 'name':self.layer_name}

    def output_shape(self):
        return (self.n_units, )

    def persist_weights(self):
        self.forward_pass.persist_op("{}_forward_pass".format(self.layer_name))
        self.backward_pass.persist_op("{}_backward_pass".format(self.layer_name))
