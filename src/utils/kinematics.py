import numpy as np

from src.utils.helper_functions import rotation_matrix_from_euler_angles


# from numba import jit


# Rotation matrix of the body is defined as how the world sees the body.
# The body is defined as the drone.
# R.T is the rotation matrix of the world as seen by the body.
# R @ WORLD2CAM is how the camera sees the world. ???

# @jit(nopython=True)
def update_kinematic_step(cart_state, rotation_matrix, acceleration, rates, dt):
    # xyz_state = [x, y z, vx, vy, vz]
    # dt = time step
    # acceleration = [ax, ay, az] # includes gravity and drag
    # rates = [roll_rate, pitch_rate, yaw_rate] # in radians per second
    # rotation_matrix = [R11, R12, R13, R21, R22, R23, R31, R32, R33]
    cart_state[0:3] += cart_state[3:6] * dt
    cart_state[3:6] += acceleration * dt
    rotation_matrix = rotate_by_rates(rotation_matrix, rates, dt)
    return cart_state, rotation_matrix


def rotate_by_rates(rotation_matrix, rates, dt): # rates in degrees per second
    rates_dt = np.deg2rad(rates) * dt
    return (rotation_matrix_from_euler_angles(*rates_dt) @ rotation_matrix.T).T


def drag_coefficient(coefficient_vector, rotation_matrix, air_velocity):
    return coefficient_vector[0] + coefficient_vector[1] * np.dot(rotation_matrix[2, :], air_velocity) + coefficient_vector[2] * np.linalg.norm(air_velocity) #???


def calculate_drag(cart_state, drag_coefficient, air_density, wind_velocity_vector):
    velocity = cart_state[3:] + wind_velocity_vector
    return -0.5 * air_density * velocity * np.linalg.norm(velocity) * drag_coefficient


def calculate_gravity_vector(rotation_matrix, g=9.81):
    return -rotation_matrix[:, 2] * g


def calculate_thrust(thrust, rotation_matrix):
    return rotation_matrix[2, :] * thrust