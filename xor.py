from network import *
import random
from matplotlib import pyplot as plt

data = [
    ((0, 0), (0, 1)),
    ((0, 1), (1, 0)),
    ((1, 0), (1, 0)),
    ((1, 1), (0, 1))
]

def main():
    random.shuffle(data)

    epochs = 100
    # layer_types = [(Layer, 2, 2)]
    layer_types = [(ReLU_Layer, 2, 4), (ReLU_Layer, 4, 4), (Sigmoid_Layer, 4, 2)]
    network = Network(layer_types, epochs)
    network.train(data)
    print(network)

    true_points = list()
    for x in range(-100, 201):
        for y in range(-100, 201):
            if np.argmax(network.infer((x / 100, y / 100))) == 0:
                true_points.append((x / 100, y / 100))
    plt.scatter(*zip(*true_points))
    plt.scatter((1, 1, 0, 0), (0, 1, 0, 1), color="red")
    plt.xlim = (-1, 2)
    plt.ylim = (-1, 2)
    plt.show()


if __name__ == "__main__":
    main()