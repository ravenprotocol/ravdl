class Activation:
    
    def __init__(self):
        self.one = R.Scalar(1)
        self.zero = R.Scalar(0)
    
    def linear(self, X):
        return X
    
    def sigmoid(self, X):
        den = self.one + R.exp(R.neg(X))
        return R.div(one, den)
    
    def ReLU(self, X):
        return X.where(X * self.zero, condition = X > self.zero)
    
    def softmax(self, X):
        return R.div(R.exp(X), R.sum(R.exp(X)))
    
    def tanh(self, X):
        num = R.exp(X) - R.exp(R.neg(X))
        den = R.exp(X) + R.exp(R.neg(X))
        return R.div(num, den)
    
    def softsign(self, X):
        den = self.one + R.abs(X)
        return R.div(X, den)
    
    def LeakyRelu(self, alpha =  0.2, X):
        alpha = R.Scalar(alpha)
        return X.where(X.multiply(alpha), condition = X > self.zero)
    
    def elu(self, alpha = 0.2, X):
        alpha = R.Scalar(alpha)
        return X.where((R.exp(X).sub(self.one)).multiply(alpha), condition=x > self.zero)
    
    def selu(self, alpha = 1.67326324, scale = 1.05070098 , X):
        alpha = R.Scalar(alpha)
        scale = R.Scalar(scale)
        return ((X.multiply(scale)).where((R.exp(X).sub(self.one)).multiply(scale).multiply(alpha), condition = X > self.zero))