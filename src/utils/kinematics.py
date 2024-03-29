import numpy as np

from utils.helper_functions import euler_angles_to_rotation_matrix


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
    rotation_matrix = rotate_body_by_rates(rotation_matrix, rates, dt)
    return cart_state, rotation_matrix


def rotate_body_by_rates(rotation_matrix, rates, dt): # rates in degrees per second
    # rotates the body in the body reference frame and returns rotation matrix in world reference frame
    rates_dt = np.deg2rad(rates) * dt
    return (euler_angles_to_rotation_matrix(*rates_dt) @ rotation_matrix.T).T


def calculate_drag(rotation_matrix, velocity, wind_velocity_vector, drag_coefficients, cross_section_areas, air_density=1.2225):
    # air_density = 1.2225 [kg/m^3] at 20 degrees C
    velocity_sum = velocity + wind_velocity_vector
    drag_force_self_ref_frame = -0.5 * drag_coefficients * air_density * cross_section_areas * (rotation_matrix.T @ velocity_sum) * np.linalg.norm(velocity_sum)
    print(drag_force_self_ref_frame)
    return rotation_matrix @ drag_force_self_ref_frame # drag force in world reference frame


def gravity_vector(mass, g=9.81):
    # gravity Force in world reference frame.
    # g = 9.81 [m/s^2]
    # gravity is in the Z axis in negative direction.
    return np.array([0, 0, -g * mass])


def thrust_vector(thrust, rotation_matrix):
    return rotation_matrix[:, 2] * thrust


def total_force(thrust, drag, gravity):
    return thrust + drag + gravity


def spring_force(distance, normal, velocity, spring_constant=1000, damping_constant=10):
    # spring_constant =  [N/m]
    # damping_constant = [Ns/m]
    return (-spring_constant * distance - damping_constant * np.dot(velocity, normal)) * normal