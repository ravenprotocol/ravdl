from __future__ import print_function, division
from terminaltables import AsciiTable
import numpy as np
# import progressbar
from ..utils import batch_iterator
from ..utils.misc import bar_widgets
from .callbacks import Callback

import ravop as R
import json


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
    def __init__(self, optimizer, loss, validation_data=None,save_weight=None,load_weights=None):
        self.optimizer = optimizer
        self.layers = []
        self.errors = {"training": [], "validation": []}
        self.loss_function = loss()
        self.save_weight=save_weight
        self.load_weights=load_weights
        self.loaded_weights=None
        if self.load_weights is not None:
            with open(self.load_weights, "rb") as f:
                weights = json.load(f)
            self.loaded_weights=weights
        self.layer_type=[]
        self.loss=None
        # print(weights)
        # print(self.save_weight)
        # self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)

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
        if hasattr(layer, 'initialize') :
            layer.initialize(optimizer=self.optimizer)   

        layer_name=layer.__class__.__name__
        self.layer_type.append(layer_name)
        if layer_name in self.layer_type:
            layer_name+=str(self.layer_type.count(layer.__class__.__name__)) 
        layer.layer_name=layer_name


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

    def fit(self, X, y, n_epochs, batch_size,training=True, callbacks=None): 
        """ Trains the model for a fixed number of epochs """
        X = R.t(X)
        y = R.t(y)
        
        # for _ in self.progressbar(range(n_epochs)):
        if self.loaded_weights is not None:
            self.load_weights_file()
        self.callbacks=callbacks
        cb=Callback(self.callbacks,model=self)

        if training is True:
        #callback func on begining training
            cb.on_train_begin()
            for epoch in range(1, n_epochs + 1):
                #callback func on epoch training
                cb.on_epoch_begin()

                print('\nEpoch: ', epoch)
                batch_error = []
                for X_batch, y_batch in batch_iterator(X, y, batch_size=batch_size):
                    cb.on_batch_begin() #callbacks on batch begin
                    loss, _ = self.train_on_batch(X_batch, y_batch)
                    batch_error.append(loss())
                    self.loss=loss()
                    cb.on_batch_end()#Callback on batch ending

                print("Batch Error: ", batch_error)
                self.errors["training"].append(np.mean(batch_error))


                if self.val_set is not None:
                    val_loss, _ = self.test_on_batch(self.val_set["X"], self.val_set["y"])
                    self.errors["validation"].append(val_loss())
                
                #callback func on epoch end
                cb.on_epoch_end()

                if self.save_weight is True:
                    self.save_model()
            return self.errors["training"], self.errors["validation"]
            
            #callback func on ending training
            cb.on_train_end()
    
    def load_weights_file(self):
        for i in self.layers:
            l_name=i.get_layer_name()
            layer_w=self.loaded_weights[l_name]
            if layer_w is not None:
                i.W=R.t(layer_w[0])
                i.w0=R.t(layer_w[1])
        

    def save_model(self,filename="weights.json"):
        layer_weights={}
        layer_type=[]
        for layer in self.layers:
            weights=layer.get_weights()
            layer_weights[layer.layer_name]=weights
        with open(filename, "w") as outfile:
            json.dump(layer_weights, outfile)
        pass

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
            params = layer.parameters()()
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

