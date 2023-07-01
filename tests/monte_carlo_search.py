import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from copy import copy
from utils import render3d


def offsprints_from(best_x, n_offsprings, noise_std=0.1):
    x = np.tile(best_x, (n_offsprings, 1))
    noise = np.random.randn(len(best_x) * n_offsprings, len(x[0]))
    noise /= np.linalg.norm(noise, axis=1)[:, np.newaxis]
    x += noise_std * noise
    return x


def random_points_on_a_sphere(n_points, dim):
    """
    :param n_points: number of points
    :param dim: dimension
    :return: n_points points on a sphere
    """
    points = np.random.randn(n_points, dim)
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
    return points


def softmax(z):
    return np.exp(z) / np.exp(z).sum()


def monte_carlo_search(x, n_offsprings, n_iterations, f, temperature=1):
    """
    :param x: initial point
    :param n_offsprings: number of offsprings
    :param n_iterations: number of iterations
    :param f: function to optimize
    :return: best point
    """
    x = np.array(x) # initial point
    x = offsprints_from(x, n_offsprings, noise_std=0.1)
    f_x = f(x) # evaluate the function on the offsprings
    idx = np.argsort(f_x)[-temperature:] # sort the offsprings by their function value and take the top 5
    next_x = x[idx] # best offsprings
    for i in range(n_iterations):
        x = offsprints_from(next_x, n_offsprings,  noise_std=0.1)
        f_x = -f(x)
        p = softmax(f_x / temperature)
        number_of_offsprings_from_each_index = np.random.multinomial(n_offsprings, p, size=1).reshape(-1)
        next_x = np.array([x[i] for i, n in enumerate(number_of_offsprings_from_each_index) for _ in range(n)])
        # idx = np.argsort(f_x)[-temperature:]
        # next_x = x[idx]
    return next_x


def euclidean_distance(s, g):
    min_dist = np.zeros(shape=(len(s),))
    for i, ss in enumerate(s):
        min_dist[i] = np.abs(np.linalg.norm(ss - g, axis=1).min() + np.random.randn() * 0.5)
    return min_dist


if __name__ == '__main__':
    n_neuron = 10
    states = [3*np.random.randn(1, 3) for i in range(n_neuron)]
    n_offsprings = 10
    n_iterations = 2
    state_list = copy(states)
    first_states = np.concatenate(copy(states))
    # creates random pallet for matplotlib
    colors = np.random.rand(n_neuron, 3)
    ax, fig = render3d.init_3d_axis()
    for t in range(2000):
        ax.clear()
        for i in range(n_neuron):
            # partial_euclidean_distance = partial(euclidean_distance, g=np.concatenate(state_list[:i] + state_list[i + 1:]))
            other_neurons = np.concatenate([first_states[:i]] + [first_states[i + 1:]])
            partial_euclidean_distance = partial(euclidean_distance, g=other_neurons)
            states[i] = monte_carlo_search(states[i], n_offsprings, n_iterations, partial_euclidean_distance, temperature=7)
            state_list[i] = np.concatenate((state_list[i], states[i]), axis=0)
            if t % 10 == 0:
                render3d.plot_3d_points(ax, state_list[i], s=1, c=colors[i].reshape(1, 3), alpha=0.5)
        if t % 10 == 0:
            render3d.plot_3d_points(ax, first_states, s=100, c='k', alpha=0.5)
            render3d.show_plot(ax, fig, edge=6.2)
    plt.show()

