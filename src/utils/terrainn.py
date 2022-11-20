import numpy as np
from utils import nn
import matplotlib.pyplot as plt
import render3d


class TerraiNN:
    def __init__(self, hidden_layers, activation):
        self.layers = np.hstack((2, hidden_layers, 1))
        self.activation = activation
        layers_list = []
        # create sequential model
        for i in range(len(self.layers) - 1):
            layers_list.append(nn.Linear(self.layers[i], self.layers[i + 1]))
            layers_list.append(activation())
        del layers_list[-1]
        self.model = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.model(x)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, x, grad_output):
        return self.model.backward(x, grad_output)


if __name__ == '__main__':
    terrain = TerraiNN([10, 10], activation=nn.Sin)
    scale = 5
    resolusion = 100
    # xy = scale * (2 * np.random.rand(10000, 2) - 1)
    axis = np.linspace(-scale, scale, resolusion)
    xy = np.array(np.meshgrid(axis, axis)).T.reshape(-1, 2)
    z = terrain(xy)
    z /= np.max(z)
    z = np.exp(z)
    xyz = np.hstack((xy, z))
    # ax, fig = render3d.init_3d_axis()
    # render3d.plot_3d_points(ax, xyz, alpha=0.1)
    # plt.scatter(xy[:, 0], xy[:, 1], c=z, alpha=0.1, cmap='jet')
    plt.imshow(z.reshape(resolusion, resolusion), cmap='jet')
    plt.show()



