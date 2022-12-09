import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    dataset = 2*np.random.rand(100, 3)-1
    x = dataset[:, :2]
    y = dataset[:, 2]
    layers = [2, 16, 16, 1]
    weights = [2 * np.random.randint(low=0, high=2, size=(layers[i], layers[i + 1])) -1 for i in range(len(layers) - 1)]
    # biases = [2 * np.random.randint(low=0, high=2, size=(1, layers[i + 1])) - 1 for i in range(len(layers) - 1)]
    dw = [np.zeros(shape=(layers[i], layers[i + 1])) for i in range(len(layers) - 1)]
    for i in range(len(weights)):
        new_x = np.tanh(np.dot(x, weights[i]))
        dw[i] = np.dot(x.T, new_x)
        x = new_x
    y - x.reshape(-1)



