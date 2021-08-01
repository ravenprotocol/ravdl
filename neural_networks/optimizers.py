import ravop.core as R
import numpy as np
import sys

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
            while grad_wrt_w.status!="computed":
                pass
            self.m = R.Tensor(np.zeros(np.shape(grad_wrt_w)))
            self.v = R.Tensor(np.zeros(np.shape(grad_wrt_w)))

        temp=R.mul( R.sub(R.Scalar(1) , self.b1) ,  grad_wrt_w)
        self.m = R.multiply(self.b1 , R.add(self.m , temp ))
        temp1=R.mul( R.sub(R.Scalar(1) , self.b2) , R.pow(grad_wrt_w, R.Scalar(2)))
        self.v = R.add( R.mul(self.b2 , self.v ) , temp1)
        m_hat = R.div(self.m , R.sub(R.Scalar(1) , self.b1))
        v_hat = R.div(self.v , R.sub(R.Scalar(1) , self.b2))
        temp2=R.add(R.square_root(v_hat) , self.eps)
        self.w_updt = R.add(R.mul(R.Scalar(self.learning_rate) , m_hat) , temp2)

        return R.div(w , self.w_updt)
