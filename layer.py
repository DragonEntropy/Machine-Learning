import numpy as np

class Layer:
    def __init__(self, prev_dim, layer_dim, max_epochs, base_lr=0.1):
        self.base_lr = base_lr
        self.lr = base_lr
        self.max_epochs = max_epochs
        self.epoch = 0
        self.layer_dim = layer_dim
        self.prev_dim = prev_dim
        self.weights = np.random.rand(self.layer_dim, self.prev_dim) * 2 - 1
        self.biases = np.random.rand(self.layer_dim) * 2 - 1

    def __repr__(self):
        return f"Weights:\n{self.weights}\n\nBiases:\n{self.biases}\n"

    def update_epoch(self, epoch):
        self.epoch = epoch
        self.lr = self.base_lr * (1 - self.epoch / self.max_epochs)

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.matmul(self.weights, self.inputs) + self.biases
        return self.outputs

    def backward(self, errors):
        weight_loss = np.outer(errors, self.inputs)
        bias_loss = errors
        prev_loss = np.matmul(self.weights.T, errors)
        self.weights -= self.lr * weight_loss
        self.biases -= self.lr * bias_loss
        return prev_loss