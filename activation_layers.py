from layer import *

class Sigmoid_Layer(Layer):
    def __init__(self, prev_dim, layer_dim, max_epochs, base_lr=0.1):
        super().__init__(prev_dim, layer_dim, max_epochs, base_lr)

    def forward(self, inputs):
        self.inputs = inputs
        inter = np.matmul(self.weights, self.inputs) + self.biases
        self.outputs = np.exp(inter) / sum(np.exp(inter))
        return self.outputs
    
    def backward(self, errors):
        inter_loss = errors * (1 - errors)
        weight_loss = np.outer(inter_loss, self.inputs)
        bias_loss = errors
        prev_loss = np.matmul(self.weights.T, errors)
        self.weights -= self.lr * weight_loss
        self.biases -= self.lr * bias_loss
        return prev_loss
    
class ReLU_Layer(Layer):
    def __init__(self, prev_dim, layer_dim, max_epochs, base_lr=0.1):
        super().__init__(prev_dim, layer_dim, max_epochs, base_lr)

    def forward(self, inputs):
        self.inputs = inputs
        inter = np.matmul(self.weights, self.inputs) + self.biases
        self.outputs = np.maximum(0, inter)
        return self.outputs
    
    def backward(self, errors):
        inter_loss = np.maximum(0, errors)
        weight_loss = np.outer(inter_loss, self.inputs)
        bias_loss = errors
        prev_loss = np.matmul(self.weights.T, errors)
        self.weights -= self.lr * weight_loss
        self.biases -= self.lr * bias_loss
        return prev_loss
