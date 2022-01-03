import ravop.core.c as R
import numpy as np
import sys,time

class Adam():
    def __init__(self, learning_rate=0.001, b1=0.9, b2=0.999):
        self.learning_rate = learning_rate
        self.eps = R.Scalar(sys.float_info.epsilon)
        self.m = None
        self.v = None
        # Decay rates
        self.b1 = R.Scalar(b1)
        self.b2 = R.Scalar(b2)

    def update(self, w, grad_wrt_w):
        # If not initialized
        if self.m is None:
            grad_wrt_w.wait_till_computed()
            self.m = R.Tensor(np.zeros(np.shape(grad_wrt_w())))
            self.v = R.Tensor(np.zeros(np.shape(grad_wrt_w())))

        temp=R.mul( R.sub(R.Scalar(1) , self.b1) ,  grad_wrt_w)
        self.m = R.multiply(self.b1 , R.add(self.m , temp ))
        temp1=R.mul( R.sub(R.Scalar(1) , self.b2) , R.pow(grad_wrt_w, R.Scalar(2)))
        self.v = R.add( R.mul(self.b2 , self.v ) , temp1)
        m_hat = R.div(self.m , R.sub(R.Scalar(1) , self.b1))
        v_hat = R.div(self.v , R.sub(R.Scalar(1) , self.b2))
        temp2=R.add(R.square_root(v_hat) , self.eps)
        self.w_updt = R.add(R.mul(R.Scalar(self.learning_rate) , m_hat) , temp2)

        return R.sub(w , self.w_updt)


class RMS_prop():
    def __init__(self, learning_rate=0.01, rho=0.9):
        self.learning_rate = R.Scalar(learning_rate)
        self.Eg = None  # Running average of the square gradients at w
        self.eps = R.Scalar(1e-8)
        self.rho =R.Scalar(rho)

    def update(self, w, grad_wrt_w):
        # If not initialized
        grad_wrt_w.wait_till_computed()
        shape_grad=grad_wrt_w.shape()
        shape_grad.wait_till_computed()
        #print("update rms prop")
        if self.Eg is None:
            self.Eg = R.Tensor(np.zeros(shape_grad()))

        #self.Eg = self.rho * self.Eg + (1 - self.rho) * np.power(grad_wrt_w, 2)
        self.Eg=R.mul(self.rho,self.Eg).add(R.sub(R.Scalar(1),self.rho)).mul(R.square(grad_wrt_w))
        # Divide the learning rate for a weight by a running average of the magnitudes of recent
        # gradients for that weight
        #return w - self.learning_rate * grad_wrt_w / np.sqrt(self.Eg + self.eps)
        return w.sub(self.learning_rate).mul(grad_wrt_w).div(R.square_root(self.Eg + self.eps))