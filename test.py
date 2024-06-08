from layer import *
from activation_layers import *
import random

data = [
    ((0, 0, 0, 0), (0, 1)),
    ((0, 0, 0, 1), (0, 1)),
    ((0, 0, 1, 0), (0, 1)),
    ((0, 0, 1, 1), (0, 1)),
    ((0, 1, 0, 0), (0, 1)),
    ((0, 1, 0, 1), (0, 1)),
    ((0, 1, 1, 0), (1, 0)),
    ((0, 1, 1, 1), (1, 0)),
    ((1, 0, 0, 0), (0, 1)),
    ((1, 0, 0, 1), (0, 1)),
    ((1, 0, 1, 0), (0, 1)),
    ((1, 0, 1, 1), (0, 1)),
    ((1, 1, 0, 0), (0, 1)),
    ((1, 1, 0, 1), (0, 1)),
    ((1, 1, 1, 0), (1, 0)),
    ((1, 1, 1, 1), (1, 0)),
]

def main():
    random.shuffle(data)

    epochs = 100
    layers = [Layer(4, 2, epochs)]
    for epoch in range(0, epochs):
        total_error = 0
        correctness = list()
        for layer in layers:
            layer.update_epoch(epoch)

        for i, point in enumerate(data):
            result = point[0]
            for layer in layers:
                result = layer.forward(result)

            errors = result - point[1]
            total_error += sum(np.abs(errors))
            for layer in reversed(layers):
                errors = layer.backward(errors)
            correctness.append(int(np.argmax(result) == np.argmax(point[1])))

        print(f"Epoch {epoch + 1} accuracy: {sum(correctness)} / {len(data)}, total loss: {total_error}, synposis: {correctness}")
    for layer in layers:
        print(layer)

if __name__ == "__main__":
    main()