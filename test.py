from network import *
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
    layer_types = [(Sigmoid_Layer, 4, 2)]
    network = Network(layer_types, epochs)
    network.train(data)

if __name__ == "__main__":
    main()