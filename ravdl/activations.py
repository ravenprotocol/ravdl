import ravcom.core as R

class Activation:
    
    def __init__(self):
        self.one = R.Tensor([1])
        self.zero = R.Tensor([0])
        pass
    
    def linear(self, X):
        return X
    
    def sigmoid(self, X):
        den = self.one + R.exp(R.neg(X))
        return R.div(one, den)
    
    def ReLU(self, X):
        return R.max(R.concat(self.zero, X))
    
    def softmax(self, X):
        return R.div(R.exp(X), R.sum(R.exp(X)))
    
    def tanh(self, X):
        num = R.exp(X) - R.exp(R.neg(X))
        den = R.exp(X) + R.exp(R.neg(X))
        return R.div(num, den)
    
    def softsign(self, X):
        den = self.one + R.abs(X)
        return R.div(X, den)