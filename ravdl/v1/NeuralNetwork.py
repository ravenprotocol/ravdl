from __future__ import print_function, division
from terminaltables import AsciiTable
from ..utils import batch_iterator
from ..globals import globals as g
import ravop as R
import onnx


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
        self.loss_function = loss()
        g.zero = R.t(0)
        g.half = R.t(0.5)
        g.one = R.t(1)
        g.two = R.t(2)

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
            layer_name += "_"+layer.activation_func.__class__.__name__
        layer.set_layer_name(layer_name)

        # Add layer to the network
        self.layers.append(layer)

    def test_on_batch(self, X, y):
        """ Evaluates the model over a single batch of samples """
        y_pred = self._forward_pass(X, training=False)
        loss = R.mean(self.loss_function.loss(y, y_pred))
        acc = self.loss_function.acc(y, y_pred)

        return loss, acc

    def train_on_batch(self, X, y):
        """ Single gradient update over one batch of samples """
        y_pred = self._forward_pass(X)
        loss = R.mean(self.loss_function.loss(y, y_pred))
        # print('   Loss: ', loss())
        acc = self.loss_function.acc(y, y_pred)
        # Calculate the gradient of the loss function wrt y_pred
        loss_grad = self.loss_function.gradient(y, y_pred)
        # print('   Loss Gradient: ', loss_grad())
        # Backpropagate. Update weights
        self._backward_pass(loss_grad=loss_grad)

        return loss, acc

    def fit(self, X, y, n_epochs, batch_size, save_model = False):
        """ Trains the model for a fixed number of epochs """
        for epoch in range(1, n_epochs + 1):
            print('\nEpoch: ', epoch)
            batch_error = []
            batch_num = 1
            for X_batch, y_batch in batch_iterator(X, y, batch_size=batch_size):
                loss, _ = self.train_on_batch(X_batch, y_batch)
                loss.persist_op(name = "training_loss_epoch_{}_batch_{}".format(epoch,batch_num))
                batch_num += 1

            if self.val_set is not None:
                val_loss, _ = self.test_on_batch(self.val_set["X"], self.val_set["y"])
                val_loss.persist_op(name="val_loss_epoch_{}".format(epoch))
        
        if save_model:
            self.save_model()
            
    def _forward_pass(self, X, training=True):
        """ Calculate the output of the NN """
        layer_output = X
        for layer in self.layers:
            layer_output = layer.forward_pass(layer_output, training)

        return layer_output

    def _backward_pass(self, loss_grad):
        """ Propagate the gradient 'backwards' and update the weights in each layer """
        for layer in reversed(self.layers):
            loss_grad = layer.backward_pass(loss_grad)

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
