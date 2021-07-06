import ravop.core as R
import numpy as np

def batch_iterator(X, y=None, batch_size=64):
    while X.status != "computed":
        pass
    n_samples = len(X.output)
    for i in range(0, n_samples, batch_size):
        begin, end = i, min(i + batch_size, n_samples)
        if y is not None:
            return [X.slice(begin=begin, size=end-begin), y.slice(begin=begin, size=end-begin)]
        else:
            yield X.slice(begin=begin, size=end-begin)


class NeuralNetwork():

    def __init__(self, optimizer, loss):
        self.optimizer = optimizer
        self.layers = []
        self.errors = {"training": [], "validation": []}
        self.loss_function = loss()

    def set_trainable(self, trainable):
        for layer in self.layers:
            layer.trainable = trainable

    def add(self, layer):
        if self.layers:
            layer.set_input_shape(shape=self.layers[-1].output_shape())
        if hasattr(layer, 'initialize'):
            layer.initialize(optimizer=self.optimizer)

        # Add layer to the network
        self.layers.append(layer)

    def train_on_batch(self, X, y):
        y_pred = self._forward_pass(X)
        loss = R.mean(self.loss_function.loss(y, y_pred))
        acc = self.loss_function.acc(y, y_pred)
        # Calculate the gradient of the loss function wrt y_pred
        loss_grad = self.loss_function.gradient(y, y_pred)
        # Backpropagate. Update weights
        self._backward_pass(loss_grad=loss_grad)

        return loss, acc

    def fit(self, X, y, n_epochs, batch_size):
        X = R.Tensor(X)
        y = R.Tensor(y)
        while X.status != "computed":
            pass
        n_samples = len(X.output)
        for _ in range(n_epochs):
            batch_error = []
            for batch in range(0, n_samples, batch_size):
                begin, end = batch, min(batch + batch_size, n_samples)
                [batch_x,batch_y]=X.slice(begin=begin, size=end - begin), y.slice(begin=begin, size=end - begin)
                loss, _ = self.train_on_batch(batch_x, batch_y)
                batch_error.append(loss)

            self.errors["training"].append(np.mean(batch_error))

        return self.errors["training"]

    def _forward_pass(self, X, training=True):
        layer_output = X
        for layer in self.layers:
            layer_output = layer.forward_pass(layer_output, training)

        return layer_output

    def _backward_pass(self, loss_grad):
        for layer in reversed(self.layers):
            loss_grad = layer.backward_pass(loss_grad)

    def predict(self, X):
        return self._forward_pass(X, training=False)
