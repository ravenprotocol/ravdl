# RavDL - Deep Learning Library

Introducing Raven Protocol's Distributed Deep Learning tool that allows Requesters to easily build, train and test their neural networks by leveraging the compute power of participating nodes across the globe.


## Working
RavDL can be thought of as a high level wrapper (written in Python) that defines the mathematical backend for building layers of neural networks by utilizing the fundamental operations from Ravop library to provide essential abstractions for training complex DL architectures in the Ravenverse.  

This framework seemlessly integrates with the Ravenverse where the models get divided into optimized subgraphs, which get assigned to the participating nodes for computation in a secure manner. Once all subgraphs have been computed, the saved model will be returned to the requester.

In this manner, a requester can securely train complex models without dedicating his or her own system for this heavy and time-consuming task.

There is something in it for the providers too! The nodes that contribute their processing power will be rewarded with tokens proportionate to the capabilities of their systems and duration of participation. More information is available here.

## Installation

Make sure Ravop is installed and working properly. <Link>

### With PIP
```bash
pip install ravdl
```

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

This section gives a more detailed walkthrough on how a requester can define their ML/DL architectures in Python by using RavDL and Ravop functionalities.

>**Note:** The complete scripts of the functionalities demonstrated in this document are available in the [Ravenverse Repository](https://github.com/ravenprotocol/ravenverse) in the *```ANN_example```* and *```CNN_example```* folders.   

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
train_err = model.fit(X, y, n_epochs=5, batch_size=25, save_model=True)
```
By default, the batch losses for each epoch are made to persist in the Ravenverse and can be retrieved later on as and when the computations of those losses are completed. <br>
The ```save_model``` parameter can be set to true if the trained model needs to be retrieved later for inference or further training. 
> **Note:** It is recommended that the model object be saved as a pickle file in order to load the saved weights later on.
```python
import pickle as pkl
pkl.dump(model, open("model.pkl", "wb"))
```
### Testing the Model on Ravenverse

If required, model inference can be tested by using the ```predict``` function. The output is stored as an Op and should be made to persist in order to view it later on.

```python 
y_test_pred = model.predict(X_test)
y_test_pred.persist_op(name='test_prediction')
```

### Activating the Graph

Once the aforementioned steps have been completed and all required Ops for the Graph have been defined, then Graph can be activated and made ready for execution as follows: 

```python
R.activate()
```
Here is what should happen on activating the Graph (the script executed below is available [here](https://github.com/ravenprotocol/ravenverse/blob/master/ANN_example/ANN_compile.py)):
![ANN_compile](https://user-images.githubusercontent.com/36445587/178669352-03758cbd-85ae-4ccf-bdc8-a7a99001a065.gif)

### Executing the Graph
Once the Graph has been activated, no more Ops can be added to it. The Graph is now ready for execution. Once Ravop has been initialized with the token, the graph can be executed and tracked as follows:

```python
R.execute()
R.track_progress()
```
Here is what should happen on executing the Graph (the script executed below is available [here](https://github.com/ravenprotocol/ravenverse/blob/master/ANN_example/ANN_execute.py)):

![ANN_execute](https://user-images.githubusercontent.com/36445587/178670666-0b98a36b-12f9-4d4b-a956-2d8bafbe6728.gif)

### Retrieving Persisting Ops
As mentioned above, the batch losses for each epoch can be retrieved as and when they have been computed. The entire Graph need not be computed in order to view a persisting Op that has been computed. Any other Ops that have been made to persist, such as ```y_test_pred``` in the example above, can be retrieved as well.

```python
batch_loss = R.fetch_persisting_op(op_name="training_loss_epoch_{}_batch_{}".format(epoch_no, batch_no))
print("training_loss_epoch_1_batch_1: ", batch_loss)

y_test_pred = R.fetch_persisting_op(op_name="test_prediction")
print("Test prediction: ", y_test_pred)
```
> **Note:** The Ops that have been fetched are of type *int*, *float* or *list*.

### Saving the Model from RavDL to Onnx
If the ```save_model``` parameter has been set to *True* in the ```model.fit``` function, the model can be loaded as an Onnx model after Ravop has been initialized with the token.<br>
The persisting weights and biases of the trained model will be loaded onto the Onnx model.

```python
test_model = pkl.load(open("model.pkl", "rb"))
print("\n\n Pickle loaded model: \n")
test_model.summary()

test_model.get_onnx_model("test_ann")
```
The ```get_onnx_model``` function takes the name of the onnx file, in which the model must be saved, as parameter. In this case it will be saved as *```"test_ann.onnx"```*.

> **Note:** As mentioned above, a ```ravdl.neural_networks.NeuralNetwork``` instance with the same architecture as the saved model is required for loading the model. Hence it is recommended to save the model object as a pickle file.

The script executed below, to load the Onnx model, is available [here](https://github.com/ravenprotocol/ravenverse/blob/master/ANN_example/ANN_get_onnx.py).

![get_onnx](https://user-images.githubusercontent.com/36445587/178671018-b5f8d1c6-a5a8-4426-8466-811c69997755.png)

Now the Onnx model can be tested and fine tuned locally. An example inference made with a loaded Onnx model is shown below.

```python
import onnx
import numpy as np
import onnxruntime as rt

model_file_path = "test_ann.onnx"

# Test onnx model
sess = rt.InferenceSession(model_file_path)
input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape
batch_size = 1
dummy_input = np.random.random(
    (batch_size, *input_shape[1:])).astype(np.float32)
prediction = sess.run(None, {input_name: dummy_input})[0]
print(prediction)
```

### Loading an Onnx Model to RavDL
RavDL now supports loading a pre-existing Onnx model into RavDL (for supported layers). 

```python
from ravdl.neural_networks.load_onnx_model import load_onnx

model = load_onnx(model_file_path="test_cnn.onnx", optimizer=Adam(), loss=CrossEntropy)
model.summary()
```

In the above code, the ```model``` obtained from the <i>"test_cnn.onnx"</i> file is an ```ravdl.neural_networks.NeuralNetwork``` instance that can further be trained or used for making predictions in the Ravenverse.

> **Note:** As of now, Onnx files exported from ```.pt``` / ```.pth``` **(Pytorch)** or from **RavDL** are supported.


## Examples

### Training a Convolutional Neural Network
A sample CNN network to be trained on Sklearn's ```load_digits``` dataset is available [here](https://github.com/ravenprotocol/ravenverse/tree/master/CNN_example).

![Screenshot 2022-05-25 at 4 54 22 PM](https://user-images.githubusercontent.com/36446402/170251625-fe875e22-082e-48a1-b2ed-0bfc4315071e.png)

<!-- ## How to Contribute -->

## License

<a href="https://github.com/ravenprotocol/ravdl/blob/master/LICENSE"><img src="https://img.shields.io/github/license/ravenprotocol/ravdl"></a>

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details