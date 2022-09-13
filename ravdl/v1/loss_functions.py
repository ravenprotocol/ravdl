from __future__ import division
import ravop as R
from ..utils import accuracy_score
from ..globals import globals as g

class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def acc(self, y, y_pred):
        return g.zero

class SquareLoss(Loss):
    def init(self): pass

    def loss(self, y, y_pred):
        return g.half * R.pow(y - y_pred, g.two)

    def gradient(self, y, y_pred):
        # return R.neg(y - y_pred)
        return y_pred - y

class CrossEntropy(Loss):
    def __init__(self): pass

    def loss(self, y, p):
        # Avoid division by zero
        p = R.clip(p,lower_limit=1e-15,upper_limit=1 - 1e-15)
        return R.neg(y) * R.natlog(p) - (g.one - y) * R.natlog(g.one - p)

    def acc(self, y, p):
        return accuracy_score(R.argmax(y, axis=1), R.argmax(p, axis=1))

    def gradient(self, y, p):
        # Avoid division by zero
        p = R.clip(p,lower_limit=1e-15,upper_limit=1 - 1e-15)
        return R.neg(R.div(y,p)) + R.div(g.one - y, g.one - p)