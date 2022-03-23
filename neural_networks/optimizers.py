import ravop as R
import numpy as np

class RMSprop():
    def __init__(self, learning_rate=0.01, rho=0.9):
        self.learning_rate = R.t(learning_rate)
        self.Eg = None # Running average of the square gradients at w
        self.eps = R.t(1e-8)
        self.rho = R.t(rho)

    def update(self, w, grad_wrt_w):
        # If not initialized
        if self.Eg is None:
            self.Eg = R.t(np.zeros(np.shape(grad_wrt_w())))

        self.Eg = self.rho * self.Eg + (R.t(1) - self.rho) * R.pow(grad_wrt_w, R.t(2))
        # Divide the learning rate for a weight by a running average of the magnitudes of recent
        # gradients for that weight
        return w - self.learning_rate * R.div(grad_wrt_w, R.square_root(self.Eg + self.eps))