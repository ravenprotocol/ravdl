
from sklearn import datasets
import numpy as np
from neural_networks import NeuralNetwork
from neural_networks.layers import Dense,Activation
from neural_networks.optimizers import RMS_prop
from neural_networks.utils import SquareLoss

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data = datasets.load_iris()
X = data.data[0:90]
y = data.target[0:90]

X, X_test, y , y_test = train_test_split(X, y, test_size=0.33)


def to_categorical(x, n_col=None):
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot

y = to_categorical(y.astype("int"))
n_samples, n_features = X.shape
n_hidden = 15
print("no of samples:",n_samples)

optimizer = RMS_prop()
clf = NeuralNetwork(optimizer=optimizer,
                        loss=SquareLoss)

clf.add(Dense(n_hidden, input_shape=(n_features,)))
clf.add(Activation('leaky_relu'))
clf.add(Dense(n_hidden))
clf.add(Dense(2))
clf.add(Activation('softmax'))

train_err = clf.fit(X, y, n_epochs=50, batch_size=32)


# print(train_err)

#n = len(train_err)
#training, = plt.plot(range(n), train_err, label="Training Error")
#plt.legend(handles=[training])
#plt.show()


y_pred = np.argmax(clf.predict(X_test),axis=1)


accuracy = accuracy_score(y_test, y_pred)

print ("Accuracy:", accuracy)