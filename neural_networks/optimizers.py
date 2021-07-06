import ravop.core as R
import numpy as np

class Adam():
    def __init__(self, learning_rate=0.001, b1=0.9, b2=0.999):
        self.learning_rate = learning_rate
        self.eps = 1e-8
        self.m = None
        self.v = None
        # Decay rates
        self.b1 = R.Scalar(b1)
        self.b2 = R.Scalar(b2)

    def update(self, w, grad_wrt_w):
        # If not initialized
        if self.m is None:
            while grad_wrt_w.status!="computed":
                pass
            self.m = R.Tensor(np.zeros(grad_wrt_w.shape()))
            self.v = R.Tensor(np.zeros(grad_wrt_w.shape()))
        temp=R.mul( R.sub(R.Scalar(1) , self.b1) ,  grad_wrt_w)
        self.m = R.multiply(self.b1 , R.sum(self.m , temp ))
        temp1=R.mul( R.sub(R.Scalar(1) , self.b2) , R.pow(grad_wrt_w, R.Scalar(2)))
        self.v = R.sum( R.mul(self.b2 , self.v ) , temp1)
        m_hat = R.div(self.m , R.sub(R.Scalar(1) , self.b1))
        v_hat = R.div(self.v , R.sub(R.Scalar(1) , self.b2))
        temp2=R.sum(R.square_root(v_hat) , self.eps)
        self.w_updt = R.sum(R.mul(self.learning_rate , m_hat) , temp2)

        return R.div(w , self.w_updt)