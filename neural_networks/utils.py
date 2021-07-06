import numpy as np
import ravop.core as R




class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def acc(self, y, y_pred):
        return 0

def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy

class SquareLoss(Loss):
    def __init__(self): pass

    def loss(self, y, y_pred):

        return R.mul(R.Scalar(0.5), R.pow(R.sub(y , y_pred), R.Scalar(2)))

    def gradient(self, y, y_pred):
        return R.neg(R.sub(y , y_pred))

