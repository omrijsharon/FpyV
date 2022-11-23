import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from utils import render3d


def offsprints_from(best_x, n_offsprings):
    x = np.tile(best_x, (n_offsprings, 1))
    noise = np.random.randn(len(best_x) * n_offsprings, len(x[0]))
    noise /= np.linalg.norm(noise, axis=1)[:, np.newaxis]
    x += 0.1*noise
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


def monte_carlo_search(x, n_offsprings, n_iterations, f, top=2):
    """
    :param x: initial point
    :param n_offsprings: number of offsprings
    :param n_iterations: number of iterations
    :param f: function to optimize
    :return: best point
    """
    x = np.array(x) # initial point
    x = offsprints_from(x, n_offsprings)
    f_x = f(x) # evaluate the function on the offsprings
    idx = np.argsort(f_x)[-5:] # sort the offsprings by their function value and take the top 5
    best_x = x[idx] # best offsprings
    for i in range(n_iterations):
        x = offsprints_from(best_x, n_offsprings)
        f_x = -f(x)
        idx = np.argsort(f_x)[-top:]
        best_x = x[idx]
    return best_x


def euclidean_distance(s, g):
    min_dist = np.zeros(shape=(len(s),))
    for i, ss in enumerate(s):
        min_dist[i] = np.abs(np.linalg.norm(ss - g, axis=1).min() + 0.4 * np.random.randn())
    return min_dist


if __name__ == '__main__':
    state = 5*np.ones(shape=(1,3))
    goal = -5*np.ones(shape=(1,3))
    n_offsprings = 50
    n_iterations = 2
    state_list = np.empty(shape=(0,3))
    goal_list = np.empty(shape=(0,3))
    ax, fig = render3d.init_3d_axis()
    for i in range(1000):
        ax.clear()
        partial_euclidean_distance = partial(euclidean_distance, g=goal)
        state = monte_carlo_search(state, n_offsprings, n_iterations, partial_euclidean_distance)
        state_list = np.concatenate((state_list, state), axis=0)
        partial_euclidean_distance = partial(euclidean_distance, g=state)
        goal = monte_carlo_search(goal, n_offsprings, n_iterations, partial_euclidean_distance)
        goal_list = np.concatenate((goal_list, goal), axis=0)
        render3d.plot_3d_points(ax, state_list, s=1, c="b", alpha=0.5)
        render3d.plot_3d_points(ax, goal_list, s=1, c="g", alpha=0.5)
        render3d.show_plot(ax, fig, edge=5.2)
    plt.show()

