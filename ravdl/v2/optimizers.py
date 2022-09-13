import ravop as R

class RMSprop():
    def __init__(self, learning_rate=0.01, rho=0.9):
        self.learning_rate = learning_rate
        self.Eg = None # Running average of the square gradients at w
        self.eps = 1e-8
        self.rho = rho

    def data_dict(self):
        return str({
            "name": "RMSprop",
            "learning_rate" : self.learning_rate,
            # "Eg": self.Eg,
            # "eps": self.eps,
            "rho": self.rho
        })

class Adam():
    def __init__(self, learning_rate=0.001, b1=0.9, b2=0.999):
        self.learning_rate = learning_rate
        self.eps = 1e-8
        self.m = None
        self.v = None
        # Decay rates
        self.b1 = b1
        self.b2 = b2

    def data_dict(self):
        return str({
            "name": "Adam",
            "learning_rate" : self.learning_rate,
            "b1":self.b1,
            "b2": self.b2
        })

