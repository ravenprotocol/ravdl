import ravop.core as R

class Loss():
    def __init__(self):
        self.one = R.Scalar(1)
        self.zero = R.Scalar(0)

    def cross_entropy(self, y_true, y_pred):
        return R.Scalar(-1).multiply(R.sum(y_true.multiply(R.natlog(y_pred)))).div(R.Scalar(y_pred.shape[0]))

    def mean_squared_error(self, y_true, y_pred):
        return R.square(y_true.sub(y_pred)).mean()

    def mean_absolute_error(self, y_true, y_pred):
        return R.abs(y_pred.sub(y_true)).mean()

    def KL_divergence(self, y_true, y_pred):
        return R.sum((y_true.multiply(R.natlog(y_true.div(y_pred)))).where(self.zero,condition = y_true!=self.zero))

    def cosine_similarity(self, y_true, y_pred):
        return (y_true.dot(y_pred)).div((R.square_root(R.sum(R.square(y_true)))).multiply(R.square_root(R.sum(R.square(y_pred)))))

    def poisson(self, y_true, y_pred):
        return y_pred - y_true * R.natlog(y_pred)

    def huber(self, y_true, y_pred, delta=0.1):
        a = R.Scalar(0.5).multiply(R.square(y_true.sub(y_pred)))
        b = (R.Scalar(delta).multiply(R.abs(y_true.sub(y_pred)))).sub(R.Scalar(0.5).multiply(R.square(R.Scalar(delta))))
        return R.sum(a.where(b,condition=R.abs(y_true.sub(y_pred))<R.Scalar(delta)))

    def logcosh(self, y_true, y_pred):
        x = y_pred - y_true
        return R.natlog(R.div(R.exp(x) + R.exp(R.neg(X)), 2))

    def hinge(self, y_true, y_pred):
        x = self.one - (y_true * y_pred)
        return x.where(x * self.zero, condition=x > self.zero)
