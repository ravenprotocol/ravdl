import ravop as R

class RMSprop():
    def __init__(self, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False):
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered


    def data_dict(self):
        return str({
            "name": "rmsprop",
            "lr" : self.lr,
            "alpha":self.alpha,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
            "centered": self.centered
        })

class Adam():
    def __init__(self, lr=0.001, betas=(0.9,0.999), eps=1e-08, weight_decay=0, amsgrad=False):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad


    def data_dict(self):
        return str({
            "name": "adam",
            "lr" : self.lr,
            "betas":self.betas,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "amsgrad": self.amsgrad
        })