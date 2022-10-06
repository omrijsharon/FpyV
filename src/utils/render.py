import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.utils.kinematics import rotate_by_rates


def init_3d_axis():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    return ax, fig


def plot_3d_points(ax, points, color='b'):
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color)


def plot_3d_line(ax, points, color='b'):
    ax.plot(points[:, 0], points[:, 1], points[:, 2], c=color)


def plot_3d_plane(ax, points, color='b'):
    ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], color=color, alpha=0.5)


def plot_3d_grid(ax, points, color='b'):
    ax.plot_wireframe(points[:, 0], points[:, 1], points[:, 2], color=color, alpha=0.5)


def plot_3d_arrows(ax, points, arrows, color='b'):
    # for i in range(points.shape[0]):
    #     ax.quiver(points[i, 0], points[i, 1], points[i, 2], arrows[i, 0], arrows[i, 1], arrows[i, 2], color=color)
    points = points.reshape(-1, 3)
    arrows = arrows.reshape(-1, 3)
    ax.quiver(points[:, 0], points[:, 1], points[:, 2], arrows[:, 0], arrows[:, 1], arrows[:, 2], color=color)


def create_3d_grid(z_func, limits, resolution):
    x = np.linspace(limits[0][0] - limits[0][1]/2, limits[0][0] + limits[0][1]/2, resolution)
    y = np.linspace(limits[1][0] - limits[1][0]/2, limits[1][0] + limits[1][0]/2, resolution)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = z_func(X.reshape(-1), Y.reshape(-1)).reshape(resolution, resolution)
    return X, Y, Z


def plot_3d_grid_func(ax, z_func, limits, resolution, color='b'):
    X, Y, Z = create_3d_grid(z_func, limits, resolution)
    ax.plot_surface(X, Y, Z, color=color, alpha=0.5)


def plot_3d_rotation_matrix(ax, R, t, scale=1.0):
    for dim, color in enumerate(['r', 'g', 'b']):
        plot_3d_arrows(ax, t, scale * R[:, dim], color=color)


def show_plot(ax, fig):
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.zaxis.set_major_locator(MaxNLocator(integer=True))
    edge = 1.5
    ax.set_xlim(-edge, edge)
    ax.set_ylim(-edge, edge)
    ax.set_zlim(-edge, edge)
    fig.tight_layout()
    plt.pause(0.00001)


if __name__ == '__main__':
    ax, fig = init_3d_axis()
    n_frames = 360
    theta = np.linspace(0, 2 * np.pi, n_frames)
    x = np.cos(theta)
    y = np.sin(theta)
    z = np.zeros_like(x)
    points = np.stack([x, y, z], axis=1)
    R = np.eye(3)
    for i in range(n_frames):
        ax.clear()
        plot_3d_line(ax, points[:i+1])
        R = rotate_by_rates(R, np.array([40, 0, 0]), 0.1)
        plot_3d_rotation_matrix(ax, R, points[i], scale=0.5)
        show_plot(ax, fig)
    plt.show()
