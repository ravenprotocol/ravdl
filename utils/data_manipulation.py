import numpy as np
import ravop as R

def batch_iterator(X, y=None, batch_size=64):
    """ Simple batch generator """
    n_samples = X.shape[0]
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i+batch_size, n_samples)
        if y is not None:
            yield R.t(X[begin:end]), R.t(y[begin:end])
        else:
            yield R.t(X[begin:end])