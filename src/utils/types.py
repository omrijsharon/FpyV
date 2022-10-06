import numpy as np

from src.utils.helper_functions import rotation_matrix_from_euler_angles


class Vector3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    @property
    def points(self):
        return np.array([self.x, self.y, self.z])

    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)


class Rotator:
    def __init__(self, rotation_matrix):
        self.rot_mat = rotation_matrix

    def rotate(self, array):
        return self.rot_mat @ array.T

    @classmethod
    def init_from_euler_angles(cls, roll, pitch, yaw):
        return cls(rotation_matrix_from_euler_angles(roll, pitch, yaw))


class Pose:
    def __init__(self, position, rotation):
        self.location = position
        self.rotation = rotation

    def transform(self, array):
        return self.rotation.rotate(array) + self.location.points

    def transform_vector(self, vector):
        return self.rotation.rotate(vector.points) + self.location.points