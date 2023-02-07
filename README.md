<div align="center">
  <img src="https://user-images.githubusercontent.com/36446402/217170090-b3090798-bc0c-4ead-aa3b-7b4ced07e3ec.svg" width="200" height="100">
<h1> RavDL - Deep Learning Library </h1>
</div>

Introducing Raven Protocol's Distributed Deep Learning tool that allows Requesters to easily build, train and test their neural networks by leveraging the compute power of participating nodes across the globe.

RavDL can be thought of as a high level wrapper (written in Python) that defines the mathematical backend for building layers of neural networks by utilizing the fundamental operations from Ravop library to provide essential abstractions for training complex DL architectures in the Ravenverse.  

This framework seemlessly integrates with the [Ravenverse](https://www.ravenverse.ai/) where the models get divided into optimized subgraphs, which get assigned to the participating nodes for computation in a secure manner. Once all subgraphs have been computed, the saved model will be returned to the requester.

In this manner, a requester can securely train complex models without dedicating his or her own system for this heavy and time-consuming task.

There is something in it for the providers too! The nodes that contribute their processing power will be rewarded with tokens proportionate to the capabilities of their systems and duration of participation. More information is available [here](https://github.com/ravenprotocol/ravpy).

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## Table of Contents

- [Installation](#installation)
- [Layers](#layers)
  - [Dense](#dense)
  - [BatchNormalization1D](#batchnormalization1d)
  - [BatchNormalization2D](#batchnormalization2d)
  - [LayerNormalization](#layernormalization)
  - [Dropout](#dropout)
  - [Activation](#activation)
  - [Conv2D](#conv2d)
  - [Flatten](#flatten)
  - [MaxPooling2D](#maxpooling2d)
  - [Embedding](#embedding)

- [Optimizers](#optimizers)
- [Loss Functions](#losses)
- [Usage](#usage)
- [Functional Model Definition](#functional-model-definition)
- [Sequential Model Definition](#sequential-model-definition)
- [Activate Graph](#activating-the-graph)
- [Execute Graph](#executing-the-graph)
- [Retrieving Persisting Ops](#retrieving-persisting-ops)
- [License](#license)

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## Installation

Make sure [Ravop](https://github.com/ravenprotocol/ravop) is installed and working properly. 

### With PIP
```bash
pip install ravdl
```

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## Layers


### Dense
```python
Dense(n_units, initial_W=None, initial_w0=None, use_bias='True') 
```
#### Parameters
* ```n_units```: Output dimension of the layer
* ```initial_W```: Initial weights of the layer
* ```initial_w0```: Initial bias of the layer
* ```use_bias```: Whether to use bias or not

#### Shape
* Input: (batch_size, ..., input_dim)
* Output: (batch_size, ..., n_units)



### BatchNormalization1D

```python
BatchNormalization1D(momentum=0.99, epsilon=0.01, affine=True, initial_gamma=None, initial_beta=None, initial_running_mean=None, initial_running_var=None)
```

#### Parameters
* ```momentum```: Momentum for the moving average and variance
* ```epsilon```: Small value to avoid division by zero
* ```affine```: Whether to learn the scaling and shifting parameters
* ```initial_gamma```: Initial scaling parameter
* ```initial_beta```: Initial shifting parameter
* ```initial_running_mean```: Initial running mean
* ```initial_running_var```: Initial running variance

#### Shape
* Input: (batch_size, channels) or (batch_size, channels, length)
* Output: same as input


### BatchNormalization2D

```python
BatchNormalization2D(num_features, momentum=0.99, epsilon=0.01, affine=True, initial_gamma=None, initial_beta=None, initial_running_mean=None, initial_running_var=None)
```

#### Parameters
* ```num_features```: Number of channels in the input
* ```momentum```: Momentum for the moving average and variance
* ```epsilon```: Small value to avoid division by zero
* ```affine```: Whether to learn the scaling and shifting parameters
* ```initial_gamma```: Initial scaling parameter
* ```initial_beta```: Initial shifting parameter
* ```initial_running_mean```: Initial running mean
* ```initial_running_var```: Initial running variance

#### Shape
* Input: (batch_size, channels, height, width)
* Output: same as input


### LayerNormalization

```python
LayerNormalization(normalized_shape=None, epsilon=1e-5, initial_W=None, initial_w0=None)
```

#### Parameters
* ```normalized_shape```: Shape of the input or integer representing the last dimension of the input
* ```epsilon```: Small value to avoid division by zero
* ```initial_W```: Initial weights of the layer
* ```initial_w0```: Initial bias of the layer

#### Shape
* Input: (batch_size, ...)
* Output: same as input


### Dropout

```python
Dropout(p=0.5)
```

#### Parameters
* ```p```: Probability of dropping out a unit

#### Shape
* Input: any shape
* Output: same as input

### Activation

```python
Activation(name='relu')
```

#### Parameters
* ```name```: Name of the activation function

> **Currently Supported:** 'relu', 'sigmoid', 'tanh', 'softmax', 'leaky_relu','elu', 'selu', 'softplus', 'softsign', 'tanhshrink', 'logsigmoid', 'hardshrink', 'hardtanh', 'softmin', 'softshrink', 'threshold',


#### Shape
* Input: any shape
* Output: same as input

### Conv2D

```python
Conv2D(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', initial_W=None, initial_w0=None)
```

#### Parameters
* ```in_channels```: Number of channels in the input image
* ```out_channels```: Number of channels produced by the convolution
* ```kernel_size```: Size of the convolving kernel
* ```stride```: Stride of the convolution
* ```padding```: Padding added to all 4 sides of the input (int, tuple or string)
* ```dilation```: Spacing between kernel elements
* ```groups```: Number of blocked connections from input channels to output channels
* ```bias```: If True, adds a learnable bias to the output
* ```padding_mode```: 'zeros', 'reflect', 'replicate' or 'circular'
* ```initial_W```: Initial weights of the layer
* ```initial_w0```: Initial bias of the layer

#### Shape
* Input: (batch_size, in_channels, height, width)
* Output: (batch_size, out_channels, new_height, new_width)


### Flatten

```python
Flatten(start_dim=1, end_dim=-1)
```

#### Parameters
* ```start_dim```: First dimension to flatten
* ```end_dim```: Last dimension to flatten

#### Shape
* Input: (batch_size, ...)
* Output: (batch_size, flattened_dimension)


### MaxPooling2D

```python
MaxPooling2D(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
```

#### Parameters
* ```kernel_size```: Size of the max pooling window
* ```stride```: Stride of the max pooling window
* ```padding```: Zero-padding added to both sides of the input
* ```dilation```: Spacing between kernel elements
* ```return_indices```: If True, will return the max indices along with the outputs
* ```ceil_mode```: If True, will use ceil instead of floor to compute the output shape

#### Shape
* Input: (batch_size, channels, height, width)
* Output: (batch_size, channels, new_height, new_width)


### Embedding
```python
Embedding(vocab_size, embed_dim, initial_W=None)
```

#### Parameters
* ```vocab_size```: Size of the vocabulary
* ```embed_dim```: Dimension of the embedding
* ```initial_W```: Initial weights of the layer

#### Shape
* Input: (batch_size, sequence_length)
* Output: (batch_size, sequence_length, embed_dim)


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## Optimizers

### RMSprop

```python
RMSprop(lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
```

#### Parameters
* ```lr```: Learning rate
* ```alpha```: Smoothing constant
* ```eps```: Term added to the denominator to improve numerical stability
* ```weight_decay```: Weight decay (L2 penalty)
* ```momentum```: Momentum factor
* ```centered```: If True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance

### Adam

```python
Adam(lr=0.001, betas=(0.9,0.999), eps=1e-08, weight_decay=0, amsgrad=False)
```

#### Parameters
* ```lr```: Learning rate
* ```betas```: Coefficients used for computing running averages of gradient and its square
* ```eps```: Term added to the denominator to improve numerical stability
* ```weight_decay```: Weight decay (L2 penalty)
* ```amsgrad```: If True, use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## Losses
* Mean Squared Error
```python    
ravop.square_loss(y_true, y_pred)
```
* Cross Entropy
```python        
ravop.cross_entropy_loss(y_true, y_pred, ignore_index=None, reshape_target=None, reshape_label=None)
```

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## Usage

This section gives a more detailed walkthrough on how a requester can define their ML/DL architectures in Python by using RavDL and Ravop functionalities.

>**Note:** The complete scripts of the functionalities demonstrated in this document are available in the [Ravenverse Repository](https://github.com/ravenprotocol/ravenverse).   

### Authentication and Graph Definition

The Requester must connect to the Ravenverse using a unique token that they can generate by logging in on Raven's Website using their MetaMask wallet credentials.   

```python
import ravop as R
R.initialize('<TOKEN>')
```

In the Ravenverse, each script executed by a requester is treated as a collection of Ravop Operations called Graph.<br> 
> **Note:** In the current release, the requester can execute only 1 graph with their unique token. Therefore, to clear any previous/existing graphs, the requester must use ```R.flush()``` method. <br>

The next step involves the creation of a Graph... 

```python
R.flush()

algo = R.Graph(name='cnn', algorithm='convolutional_neural_network', approach='distributed')
```
> **Note:** ```name``` and ```algorithm``` parameters can be set to any string. However, the ```approach``` needs to be set to either "distributed" or "federated". 

The Current Release of RavDL supports Functional and Sequential Model Definitions.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## Functional Model Definition

### Define Custom Layers

The latest release of RavDL supports the definition of custom layers by the requester allowing them to write their own application-specific layers either from scratch or as the composition of existing layers.

The custom layer can be defined by inheriting the ```CustomLayer``` class from ```ravdl.v2.layers``` module. The class defined by the requester must implement certain methods shown as follows:

```python
class CustomLayer1(CustomLayer):
    def __init__(self) -> None:
        super().__init__()
        self.d1 = Dense(n_hidden, input_shape=(n_features,))
        self.bn1 = BatchNormalization1D(momentum=0.99, epsilon=0.01)

    def _forward_pass_call(self, input, training=True):
        o = self.d1._forward_pass(input)
        o = self.bn1._forward_pass(o, training=training)
        return o

class CustomLayer2(CustomLayer):
    def __init__(self) -> None:
        super().__init__()
        self.d1 = Dense(30)
        self.dropout = Dropout(0.9)
        self.d2 = Dense(3)

    def _forward_pass_call(self, input, training=True):
        o = self.d1._forward_pass(input)
        o = self.dropout._forward_pass(o, training=training)
        o = self.d2._forward_pass(o)
        return 
```
### Defining Custom Model Class

The custom model class can be defined by inheriting the ```Functional``` class from ```ravdl.v2``` module. This feature allows the requester to define their own model class by composing the custom and existing layers.

The class defined by the requester must implement certain methods shown as follows:

```python
class ANNModel(Functional):
    def __init__(self, optimizer):
        super().__init__()
        self.custom_layer1 = CustomLayer1()
        self.custom_layer2 = CustomLayer2()
        self.act = Activation('softmax')
        self.initialize_params(optimizer)

    def _forward_pass_call(self, input, training=True):
        o = self.custom_layer1._forward_pass(input, training=training)
        o = self.custom_layer2._forward_pass(o, training=training)
        o = self.act._forward_pass(o)
        return o
```

> **Note:** The ```initialize_params``` method must be called in the ```__init__``` method of the custom model class. This method initializes the parameters of the model and sets the optimizer for the model. 

### Defining the Training Loop

The requester can now define their training loop by using the ```batch_iterator``` function from ```ravdl.v2.utils``` module. This function takes the input and target data as arguments and returns a generator that yields a batch of data at each iteration. 

Note that the ```_forward_pass()``` and ```_backward_pass()``` methods of the custom model class must be called in the training loop.

```python
optimizer = Adam()
model = ANNModel(optimizer)

epochs = 100

for i in range(epochs):
    for X_batch, y_batch in batch_iterator(X, y, batch_size=25):
        X_t = R.t(X_batch)
        y_t = R.t(y_batch)

        out = model._forward_pass(X_t)
        loss = R.square_loss(y_t, out)
        model._backward_pass(loss)
```

### Make a Prediction

```python
out = model._forward_pass(R.t(X_test), training=False)
out.persist_op(name="prediction")
```

> **Note:** The ```_forward_pass()``` method takes an additional argument ```training``` which is set to ```True``` by default. This argument is used to determine whether the model is in training mode or not. The ```_forward_pass()``` method must be called with ```training=False``` when making predictions.


Complete example scripts of Functional Model can be found here:
- [ANN](https://github.com/ravenprotocol/ravenverse/blob/master/Requester/ann_functional.py)
- [CNN](https://github.com/ravenprotocol/ravenverse/blob/master/Requester/cnn_functional.py)


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## Sequential Model Definition

### Setting Model Parameters

```python
from ravdl.v2 import NeuralNetwork
from ravdl.v2.optimizers import RMSprop, Adam
from ravdl.v2.layers import Activation, Dense, BatchNormalization1D, Dropout, Conv2D, Flatten, MaxPooling2D

model = NeuralNetwork(optimizer=RMSprop(),loss='SquareLoss')
```

### Adding Layers to Model

```python
model.add(Dense(n_hidden, input_shape=(n_features,)))
model.add(BatchNormalization1D())
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
By default, the batch losses for each epoch are made to persist in the Ravenverse and can be retrieved later on as and when the computations of those losses are completed. 

### Testing the Model on Ravenverse

If required, model inference can be tested by using the ```predict``` function. The output is stored as an Op and should be made to persist in order to view it later on.

```python 
y_test_pred = model.predict(X_test)
y_test_pred.persist_op(name='test_prediction')
```

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## Activating the Graph

Once the model has been defined (Functional/Sequential) and all required Ops for the Graph have been defined, then Graph can be activated and made ready for execution as follows: 

```python
R.activate()
```
Here is what should happen on activating the Graph (the script executed below is available [here](https://github.com/ravenprotocol/ravenverse/blob/master/ANN_example/ANN_compile.py)):
![ANN_compile](https://user-images.githubusercontent.com/36445587/178669352-03758cbd-85ae-4ccf-bdc8-a7a99001a065.gif)

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## Executing the Graph
Once the Graph has been activated, no more Ops can be added to it. The Graph is now ready for execution. Once Ravop has been initialized with the token, the graph can be executed and tracked as follows:

```python
R.execute()
R.track_progress()
```
Here is what should happen on executing the Graph (the script executed below is available [here](https://github.com/ravenprotocol/ravenverse/blob/master/Requester/ann_sequential.py)):

![ANN_execute](https://user-images.githubusercontent.com/36445587/178670666-0b98a36b-12f9-4d4b-a956-2d8bafbe6728.gif)

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## Retrieving Persisting Ops
As mentioned above, the batch losses for each epoch can be retrieved as and when they have been computed. The entire Graph need not be computed in order to view a persisting Op that has been computed. Any other Ops that have been made to persist, such as ```y_test_pred``` in the example above, can be retrieved as well.

```python
batch_loss = R.fetch_persisting_op(op_name="training_loss_epoch_{}_batch_{}".format(epoch_no, batch_no))
print("training_loss_epoch_1_batch_1: ", batch_loss)

y_test_pred = R.fetch_persisting_op(op_name="test_prediction")
print("Test prediction: ", y_test_pred)
```
> **Note:** The Ops that have been fetched are of type **torch.Tensor**.


<!-- ## How to Contribute -->

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## License

<a href="https://github.com/ravenprotocol/ravdl/blob/master/LICENSE"><img src="https://img.shields.io/github/license/ravenprotocol/ravdl"></a>

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
