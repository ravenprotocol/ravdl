import ravop as R
import onnx
from terminaltables import AsciiTable
from .utils.data_manipulation import batch_iterator
from .layers import CustomLayer, Layer

loss_op_mapping = {
    'SquareLoss': {
        'loss': R.square_loss,
        # 'accuracy': R.square_loss_accuracy,
        'gradient': R.square_loss_gradient
    },
    'CrossEntropy': {
        'loss': R.cross_entropy_loss,
        'accuracy': R.cross_entropy_accuracy,
        'gradient': R.cross_entropy_gradient
    }
}

class NeuralNetwork():
    """Neural Network. Deep Learning base model.

    Parameters:
    -----------
    optimizer: class
        The weight optimizer that will be used to tune the weights in order of minimizing
        the loss.
    loss: class
        Loss function used to measure the model's performance. SquareLoss or CrossEntropy.
    validation: tuple
        A tuple containing validation data and labels (X, y)
    """
    def __init__(self, optimizer, loss, validation_data=None):
        self.optimizer = optimizer
        self.layers = []
        self.errors = {"training": [], "validation": []}
        self.loss_function = loss_op_mapping[loss]#loss()

        self.input = None
        self.output = None
        self.layer_type=[]

        self.val_set = None
        if validation_data:
            X, y = validation_data
            self.val_set = {"X": R.t(X), "y": R.t(y)}

    def set_trainable(self, trainable):
        """ Method which enables freezing of the weights of the network's layers. """
        for layer in self.layers:
            layer.trainable = trainable

    def add(self, layer):
        """ Method which adds a layer to the neural network """
        # If this is not the first layer added then set the input shape
        # to the output shape of the last added layer
        if self.layers:
            layer.set_input_shape(shape=self.layers[-1].output_shape())

        # If the layer has weights that needs to be initialized 
        if hasattr(layer, 'initialize'):
            layer.initialize(optimizer=self.optimizer)

        layer_name=layer.__class__.__name__
        
        self.layer_type.append(layer_name)
        if layer_name in self.layer_type:
            layer_name+="_"+str(self.layer_type.count(layer.__class__.__name__)) 
        if "Activation" in layer_name:
            layer_name += "_"+layer.activation_name
        layer.set_layer_name(layer_name)

        # Add layer to the network
        self.layers.append(layer)

    def test_on_batch(self, X):
        """ Evaluates the model over a single batch of samples """
        y_pred = self._forward_pass(X, training=False)
        return y_pred

    def train_on_batch(self, X, y):
        """ Single gradient update over one batch of samples """
        y_pred = self._forward_pass(X)
        loss = self.loss_function['loss'](y, y_pred)
        self._backward_pass(loss_grad=loss)
        return loss

    def fit(self, X, y, n_epochs, batch_size, save_model = False, persist_weights=False):
        """ Trains the model for a fixed number of epochs """
        for epoch in range(1, n_epochs + 1):
            print('\nEpoch: ', epoch)
            batch_error = []
            batch_num = 1
            for X_batch, y_batch in batch_iterator(X, y, batch_size=batch_size):
                loss = self.train_on_batch(R.t(X_batch), R.t(y_batch))
                loss.persist_op(name = "training_loss_epoch_{}_batch_{}".format(epoch,batch_num))
                batch_num += 1

            # if self.val_set is not None:
            #     val_loss, _ = self.test_on_batch(self.val_set["X"], self.val_set["y"])
            #     val_loss.persist_op(name="val_loss_epoch_{}".format(epoch))
        
        if persist_weights:
            self.persist_model()

        # if save_model:
        #     self.save_model()
            
    def _forward_pass(self, X, training=True):
        """ Calculate the output of the NN """
        layer_output = X
        for layer in self.layers:
            layer_output = layer._forward_pass(layer_output, training = training)
            if isinstance(layer_output, dict):
                layer_output = layer_output['output']

        return layer_output

    def _backward_pass(self, loss_grad):
        """ Propagate the gradient 'backwards' and update the weights in each layer """
        self.start_backward_marker = R.start_backward_marker(loss_grad)        

    def summary(self, name="Model Summary"):
        # Print model name
        print (AsciiTable([[name]]).table)
        # Network input shape (first layer's input shape)
        print ("Input Shape: %s" % str(self.layers[0].input_shape))
        # Iterate through network and get each layer's configuration
        table_data = [["Layer Type", "Parameters", "Output Shape"]]
        tot_params = 0
        for layer in self.layers:
            layer_name = layer.get_layer_name()
            params = layer.parameters()
            out_shape = layer.output_shape()
            table_data.append([layer_name, str(params), str(out_shape)])
            tot_params += params
        # Print network configuration table
        print (AsciiTable(table_data).table)
        print ("Total Parameters: %d\n" % tot_params)

    def predict(self, X):
        """ Use the trained model to predict labels of X """
        X = R.t(X)
        return self._forward_pass(X, training=False)

    def persist_model(self):
        """ Save the model's weights to a file """
        for layer in self.layers:
            layer.persist_weights()
    
    def save_model(self):
        """ Save the model's weights to a file """
        for layer in self.layers:
            layer.save_weights()

    def get_onnx_model(self, model_name):
        """ Get an ONNX model of the network """
        model_input_name = "X"
        input_shape = [None]
        input_shape.extend(list(self.layers[0].input_shape))
        
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                           onnx.TensorProto.FLOAT,
                                           input_shape)

        model_output_name = "Y"
        output_shape = [None]
        output_shape.extend(list(self.layers[-1].output_shape()))

        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                            onnx.TensorProto.FLOAT,
                                            output_shape)

        layer_nodes = []
        layer_tensors = []
        layer_input_initializer = model_input_name
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == len(self.layers)-1:
                layer_node, layer_input_initializer, layer_tensor = layer.fetch_onnx_params(layer_input_initializer, model_output_name)
                layer_nodes.append(layer_node)
                layer_tensors.extend(layer_tensor)
            else:
                layer_node, layer_input_initializer, layer_tensor = layer.fetch_onnx_params(layer_input_initializer)
                layer_nodes.append(layer_node)
                layer_tensors.extend(layer_tensor)

        graph_def = onnx.helper.make_graph(
            nodes=layer_nodes,
            name="Model",
            inputs=[X],  # Graph input
            outputs=[Y],  # Graph output
            initializer=layer_tensors,
        )

        # Create the model (ModelProto)
        model_def = onnx.helper.make_model(graph_def, producer_name="raven-protocol")
        model_def.opset_import[0].version = 14

        model_def = onnx.shape_inference.infer_shapes(model_def)

        onnx.checker.check_model(model_def)

        onnx.save(model_def, "{}.onnx".format(model_name))


class Functional():
    def __init__(self):
        self.graph = {}
        self.layer_type = []
        self.is_training_init = False
        self.last_layer_name = None
        self.final_grad = None
        self.start_backward_marker = None

    def initialize_params(self, optimizer):
        for attr in self.__dict__:
            if isinstance(self.__dict__[attr],CustomLayer) or isinstance(self.__dict__[attr],Layer):
                layer = self.__dict__[attr]
                layer_name=layer.__class__.__name__
                self.layer_type.append(layer_name)
                if layer_name in self.layer_type:
                    layer_name+="_"+str(self.layer_type.count(layer.__class__.__name__)) 
                if "Activation" in layer_name:
                    layer_name += "_"+layer.activation_name
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
                        layer.set_layer_name(layer_name)
                        if hasattr(layer, 'initialize'):
                            layer.initialize(optimizer)

    def set_graph(self):
        if not self.is_training_init:
            for attr in self.__dict__:
                if isinstance(self.__dict__[attr], Layer) or isinstance(self.__dict__[attr], CustomLayer):#if "layer" in str(type(self.__dict__[attr])).lower():
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
        self.forward_pass = self._forward_pass_call(*args, training=training, **kwargs)
        self.set_graph()
        self.last_layer_name = self.forward_pass['name']
        self.forward_pass = self.forward_pass['output']
        return self.forward_pass
            
    def _backward_pass(self,loss_grad):
        self.start_backward_marker = R.start_backward_marker(loss_grad)        
        self.is_training_init = True
        
    def predict(self, *args):
        output = self._forward_pass_call(*args, training=False)['output']
        return output

    def save_backprops(self):
        for attr in self.__dict__:
            if isinstance(self.__dict__[attr], Layer) or isinstance(self.__dict__[attr], CustomLayer):#if "layer" in str(type(self.__dict__[attr])).lower():
                layer = self.__dict__[attr]
                if hasattr(layer, 'save_backprops'):
                    layer.save_backprops()
