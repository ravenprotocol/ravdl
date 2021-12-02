import numpy as np
import ravop.core.c as R


class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()


def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = np.sum(y_true == np.round(y_pred), axis=0) / len(y_true)
    return accuracy


class SquareLoss(Loss):
    def __init__(self): pass

    def loss(self, y, y_pred):
        loss= R.mul(R.Scalar(0.5), R.pow(R.sub(y, y_pred), R.Scalar(2)))
        return loss

    def gradient(self, y, y_pred):
        return R.neg(R.sub(y, y_pred))

    def acc(self, y, y_pred):
        y_pred.wait_till_computed()
        return accuracy_score(y(), y_pred())


class Softmax():
    def __call__(self, x):
        #print("softmax")
        e_x = R.exp(R.sub(x, R.max(x, axis=-1)))
        return R.div(e_x, R.sum(e_x, axis=-1))

    def gradient(self, x):
        p = self.__call__(x)
        return R.multiply(p, R.sub(1, p))


class LeakyReLU():
    def __init__(self, alpha=0.2):
        self.alpha = R.Scalar(0.2)

    def __call__(self, X):
        return X.where(X.multiply(self.alpha), condition=X > R.Scalar(0))

    def gradient(self, X):
        return X.where(X.multiply(self.alpha), condition=X > R.Scalar(0))


class Activation():
    def LeakyReLU(self, X, alpha=0.2):
        alpha = R.Scalar(alpha)
        #print("leakyrelu")
        return X.where(X.multiply(alpha), condition=X > R.Scalar(0))

    def Softmax(self,x):
        #print("softmax")
        e_x = R.exp(R.sub(x, R.max(x)))
        return R.div(e_x, R.sum(e_x))
