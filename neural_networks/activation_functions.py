import ravop as R
from ..globals import globals as g

class Sigmoid():
    def __call__(self, x):
        return R.div(g.one, R.add(g.one, R.exp(R.neg(x))))

    def gradient(self, x):
        return self.__call__(x) * (g.one - self.__call__(x))

class Softmax():
    def __call__(self, x):
        e_x = R.exp(x - R.max(x, axis=-1, keepdims="True"))
        return R.div(e_x, R.sum(e_x, axis=-1, keepdims="True"))

    def gradient(self, x):
        return self.__call__(x) * (g.one - self.__call__(x))

class TanH():
    def __call__(self, x):
        return R.div(g.two, g.one + R.exp(R.neg(g.two) * x)) - g.one

    def gradient(self, x):
        return g.one - R.pow(self.__call__(x), g.two)

class ReLU():
    def __call__(self, x):
        condition = R.greater_equal(x, g.zero)
        return R.where(x,g.zero,condition=condition)

    def gradient(self, x):
        condition = R.greater_equal(x, g.zero)
        return R.where(g.one,g.zero,condition=condition)