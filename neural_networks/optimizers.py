import ravop as R
from ..globals import globals as g

class RMSprop():
    def __init__(self, learning_rate=0.01, rho=0.9):
        self.learning_rate = R.t(learning_rate)
        self.Eg = None # Running average of the square gradients at w
        self.eps = R.t(1e-8)
        self.rho = R.t(rho)

    def update(self, w, grad_wrt_w):
        # If not initialized
        if self.Eg is None:
            self.Eg = R.zeros(R.shape(grad_wrt_w))

        self.Eg = self.rho * self.Eg + (g.one - self.rho) * R.pow(grad_wrt_w, g.two)
        # Divide the learning rate for a weight by a running average of the magnitudes of recent
        # gradients for that weight
        return w - self.learning_rate * R.div(grad_wrt_w, R.square_root(self.Eg + self.eps))

class Adam():
    def __init__(self, learning_rate=0.001, b1=0.9, b2=0.999):
        self.learning_rate = R.t(learning_rate)
        self.eps = R.t(1e-8)
        self.m = None
        self.v = None
        # Decay rates
        self.b1 = R.t(b1)
        self.b2 = R.t(b2)

    def update(self, w, grad_wrt_w):
        # If not initialized
        if self.m is None:
            self.m = R.zeros(R.shape(grad_wrt_w))
            self.v = R.zeros(R.shape(grad_wrt_w))
        
        self.m = self.b1 * self.m + (g.one - self.b1) * grad_wrt_w
        self.v = self.b2 * self.v + (g.one - self.b2) * R.pow(grad_wrt_w, g.two)

        m_hat = R.div(self.m, g.one - self.b1)
        v_hat = R.div(self.v, g.one - self.b2)

        self.w_updt = R.div(self.learning_rate * m_hat, R.square_root(v_hat) + self.eps)

        return w - self.w_updt
