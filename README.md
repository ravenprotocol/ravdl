# RavDL - Deep Learning Library

Introducing Raven Protocol's Distributed Deep Learning tool that allows developers to easily build, train and test their neural networks by leveraging the compute power of participating nodes across the globe.


## Working
RavDL can be thought of as a high level wrapper (written in Python) that defines the mathematical backend for building layers of neural networks by utilizing the fundamental operations from RavOP library to provide essential abstractions for training complex DL architectures in the Ravenverse.  

This framework seemlessly integrates with the Ravenverse where the models get divided into optimized subgraphs, which get assigned to the participating nodes for computation in a secure manner. Once all subgraphs have been computed, the saved model will be returned to the developer.

In this manner, a developer can securely train complex models without dedicating his or her own system for this heavy and time consuming task.

There is something in it for the contributers too! The nodes that contribute their processing power will be rewarded with tokens proportionate to the capabilities of their systems and duration of participation. More information is available here.

## Installation

Make sure RavOP is installed and working properly. <Link>

### Clone
```bash
git clone https://github.com/ravenprotocol/ravdl.git
```

## Features
### Layers
```python
Dense(n_units, input_shape) 
Activation(name='relu')
BatchNormalization(momentum=0.99)
Dropout(p=0.2)
Conv2D(n_filters, filter_shape, padding='same', stride=1)
Flatten()
MaxPooling2D(pool_shape, stride=1, padding='same')
```
> **Note:** That the input_shape parameter needs to be given only for the 1st layer of the model.

### Optimizers

```python
RMSprop(learning_rate=0.01, rho=0.9)
Adam(learning_rate=0.001, b1=0.9, b2=0.999)
```
### Activation Functions
- Sigmoid
- Softmax
- Tanh
- Relu

### Losses
- MSE
- CrossEntropy

## Usage

This section gives a more detailed walkthrough on how a developer can define their ML/DL architectures in Python by using RavDL and RavOP functionalities.

### Authentication and Graph Definition

The developer must connect to the Ravenverse using a unique token that they can generate by logging in on Raven's Website using their MetaMask wallet credentials.   

```python
import ravop as R
R.initialize('<TOKEN>')
```

In the Ravenverse, each script executed by a developer is treated as a collection of RavOP Operations called Graph. The next step involves the creation of a Graph... 

```python
algo = R.Graph(name='cnn', algorithm='convolutional_neural_network', approach='distributed')
```
Note: ```name``` and ```algorithm``` parameters can be set to any string. However, the ```approach``` needs to be set to either "distributed" or "federated". 


### Setting Model Parameters

```python
from ravdl.neural_networks import NeuralNetwork
from ravdl.neural_networks.optimizers import RMSprop
from ravdl.neural_networks.loss_functions import SquareLoss

optimizer = RMSprop()
model = NeuralNetwork(optimizer=optimizer,loss=SquareLoss)
```

### Adding Layers to Model

```python
from ravdl.neural_networks.layers import Dense, Dropout, BatchNormalization, Activation

model.add(Dense(n_hidden, input_shape=(n_features,)))
model.add(BatchNormalization())
model.add(Dense(30))
model.add(Dropout(0.9))
model.add(Dense(3))
model.add(Activation('softmax'))
```

You can view the summary of model in tabular format...

```python
model.summary()
```

### Training the Model

```python
train_err = model.fit(X, y, n_epochs=5, batch_size=25)
```

### Testing the Model

```python 
import numpy as np
from sklearn.metrics import accuracy_score

y_pred = np.argmax(model.predict(X_test)(),axis=1)
accuracy = accuracy_score(y_test, y_pred)
print ("Accuracy:", accuracy)
```

### Terminating the Graph

We now wrap up the developer script by explicitly calling the end() method of the Graph.

```python
algo.end()
```


## Examples

### Training a Convolutional Neural Network
A sample CNN network to be trained on Sklearn's ```load_digits``` dataset is shown in CNN_example.py.

![Screenshot 2022-05-25 at 4 54 22 PM](https://user-images.githubusercontent.com/36446402/170251625-fe875e22-082e-48a1-b2ed-0bfc4315071e.png)

<!-- ## How to Contribute -->
