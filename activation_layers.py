from layer import *

class Sigmoid_Layer(Layer):
    def __init__(self, prev_dim, layer_dim, max_epochs, base_lr=0.5):
        super().__init__(prev_dim, layer_dim, max_epochs, base_lr)

    def act_forward(self, inter):
        return np.exp(inter) / sum(np.exp(inter))

    def act_backwards(self, errors):
        return errors * (1 - errors)
    
class Tanh_Layer(Layer):
    def __init__(self, prev_dim, layer_dim, max_epochs, base_lr=0.1):
        super().__init__(prev_dim, layer_dim, max_epochs, base_lr)

    def act_forward(self, inter):
        return np.tanh(inter)

    def act_backwards(self, errors):
        return 1 - errors * errors
    
class ReLU_Layer(Layer):
    def __init__(self, prev_dim, layer_dim, max_epochs, base_lr=0.1):
        super().__init__(prev_dim, layer_dim, max_epochs, base_lr)

    def act_forward(self, inter):
        return np.maximum(0, inter)

    def act_backwards(self, errors):
        return np.maximum(0, errors)