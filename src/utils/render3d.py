import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from icosphere import icosphere
import mpl_toolkits.mplot3d

from src.utils.kinematics import rotate_body_by_rates


def init_3d_axis():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    return ax, fig


def plot_3d_icosphere(ax, t, radius, nu,  **kwargs):
    """ kwargs: facecolor, edgecolor, linewidth, alpha """
    vertices, faces = icosphere(nu=nu)
    poly = mpl_toolkits.mplot3d.art3d.Poly3DCollection(t + radius * vertices[faces], **kwargs)
    ax.add_collection3d(poly)


def plot_3d_points(ax, points,  **kwargs):
    points = points.reshape(-1, 3)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],  **kwargs)


def plot_3d_line(ax, points,  **kwargs):
    ax.plot(points[:, 0], points[:, 1], points[:, 2],  **kwargs)


def plot_3d_plane(ax, points,  **kwargs):
    ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2],  **kwargs)


def plot_3d_grid(ax, points,  **kwargs):
    ax.plot_wireframe(points[:, 0], points[:, 1], points[:, 2],  **kwargs)


def plot_3d_arrows(ax, points, arrows,  **kwargs):
    # for i in range(points.shape[0]):
    #     ax.quiver(points[i, 0], points[i, 1], points[i, 2], arrows[i, 0], arrows[i, 1], arrows[i, 2], color=color)
    points = points.reshape(-1, 3)
    arrows = arrows.reshape(-1, 3)
    ax.quiver(points[:, 0], points[:, 1], points[:, 2], arrows[:, 0], arrows[:, 1], arrows[:, 2],  **kwargs)


def create_3d_grid(z_func, limits, resolution):
    x = np.linspace(limits[0][0] - limits[0][1]/2, limits[0][0] + limits[0][1]/2, resolution)
    y = np.linspace(limits[1][0] - limits[1][0]/2, limits[1][0] + limits[1][0]/2, resolution)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = z_func(X.reshape(-1), Y.reshape(-1)).reshape(resolution, resolution)
    return X, Y, Z


def plot_3d_grid_func(ax, z_func, limits, resolution, **kwargs):
    X, Y, Z = create_3d_grid(z_func, limits, resolution)
    ax.plot_surface(X, Y, Z, **kwargs)


def plot_3d_rotation_matrix(ax, R, t, scale=1.0,  **kwargs):
    for dim, color in enumerate(['r', 'g', 'b']):
        plot_3d_arrows(ax, t, scale * R[:, dim], color=color, **kwargs)
    # plot_3d_points(ax, t, color='k')


def plot_3d_velocity(ax, t, cart_state,  **kwargs):
    plot_3d_arrows(ax, t, cart_state[3:6],  **kwargs)


def plot_3d_thrust(ax, t, thrust, **kwargs):
    plot_3d_arrows(ax, t, thrust,  **kwargs)


def plot_3d_gravity(ax, t, g,  **kwargs):
    plot_3d_arrows(ax, t, np.array([0, 0, -g]),  **kwargs)


def show_plot(ax, fig, middle=None, edge=1.0, title=None, xlabel=None, ylabel=None, zlabel=None, equal=True, grid=True, legend=True):
    if middle is None:
        middle = np.array([0.0, 0.0, 0.0])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.zaxis.set_major_locator(MaxNLocator(integer=True))
    minmax_edges = np.vstack((middle - edge, middle + edge)).T
    ax.set_xlim(*minmax_edges[0])
    ax.set_ylim(*minmax_edges[1])
    ax.set_zlim(*minmax_edges[2])
    fig.tight_layout()
    plt.pause(0.00001)


if __name__ == '__main__':
    ax, fig = init_3d_axis()
    n_frames = 360
    g = 1
    thrust = 1.1
    theta = np.linspace(0, 2 * np.pi, n_frames)
    x = np.cos(theta)
    y = np.sin(theta)
    z = np.zeros_like(x)
    points = np.stack([x, y, z], axis=1)
    R = np.eye(3)
    for i in range(n_frames):
        ax.clear()
        plot_3d_icosphere(ax, np.array([0.5, 0, 0]), 0.3, 3, facecolor='r', edgecolor='k', linewidth=0.1, alpha=0.5)
        plot_3d_icosphere(ax, np.array([0, 0, 0.5]), 0.3, 3, facecolor='g', edgecolor='k', linewidth=0.1, alpha=0.5)
        plot_3d_line(ax, points[:i+1])
        R = rotate_body_by_rates(R, np.array([40, 0, 0]), 0.1)
        plot_3d_rotation_matrix(ax, R, points[i], scale=0.5)
        plot_3d_thrust(ax, points[i], thrust*R[:, 2] + np.array([0, 0, -g]), color='orange')
        show_plot(ax, fig)
    plt.show()
