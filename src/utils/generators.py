import numpy as np

from utils.components import Gate, Target, Cylinder
from utils.helper_functions import rotation_matrix_from_euler_angles


def generate_track(count, radius, gate_size, gate_resolution):
    theta = np.linspace(0, 2 * np.pi, count + 1)[:-1]
    gates_positions = np.vstack((np.cos(theta) * gate_size, np.sin(theta) * radius, np.zeros_like(theta))).T
    gates = []
    shapes = ["rectangle", "circle", "half_circle"]
    for i, p in enumerate(gates_positions):
        rotmat = rotation_matrix_from_euler_angles(0, 0, theta[i] + np.pi / 2)
        if shapes[i % 3] == "circle":
            gates.append(Gate(p + np.array([0, 0, gate_size / 2]), rotmat, gate_size / 2, shape=shapes[i % 3], resolution=gate_resolution))
        else:
            gates.append(Gate(p, rotmat, gate_resolution, shape=shapes[i % 3], resolution=gate_resolution))
    return gates


def generate_targets(count, center, std, size, variation, nu, path):
    return [Target(np.array(center) + std * np.random.randn(3),
                   np.abs(size + variation * np.random.randn()),
                   nu, path) for _ in range(count)]


def generate_cylinders(count, center, center_std,
                       radius, radius_std,
                       height, height_std,
                       angle_resolution, height_resolution,
                       random=False):
    return [Cylinder(np.array(center) + np.array(center_std) * np.random.randn(3),
                     np.abs(radius + radius_std * np.random.randn()),
                     np.abs(height + height_std * np.random.randn()),
                     angle_resolution, height_resolution,
                     random=random) for _ in range(count)]

