from layer import *
from activation_layers import *
import random

class Network:
    def __init__(self, layer_types, epochs, lr=0.1):
        self.layers = list()
        self.epochs = epochs
        self.lr = lr

        for layer in layer_types:
            self.layers.append(layer[0](layer[1], layer[2], epochs, lr))

    def __repr__(self):
        return "\n".join(f"{i + 1}.\n{layer.__repr__()}" for i, layer in enumerate(self.layers))

    def infer(self, point):
        result = point
        for layer in self.layers:
            result = layer.forward(result)
        return result

    def train(self, data):
        for epoch in range(0, self.epochs):
            total_error = 0
            correctness = list()
            for layer in self.layers:
                layer.update_epoch(epoch)

            for i, point in enumerate(data):
                result = point[0]
                for layer in self.layers:
                    result = layer.forward(result)

                errors = result - point[1]
                total_error += sum(np.abs(errors))
                for layer in reversed(self.layers):
                    errors = layer.backward(errors)
                correctness.append(int(np.argmax(result) == np.argmax(point[1])))

            print(f"Epoch {epoch + 1} accuracy: {sum(correctness)} / {len(data)}, total loss: {total_error}, synposis: {correctness}")