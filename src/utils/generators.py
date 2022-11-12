import numpy as np

from utils.components import Gate, Target
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


def generate_targets(count, center, std, size, variation, nu):
    return [Target(np.array(center) + std * np.random.randn(3), np.abs(size + variation * np.random.randn()), nu=nu) for _ in range(count)]