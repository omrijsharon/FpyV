import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import numpy as np

import kinematics
import render3d
from utils.helper_functions import rotation_matrix_from_euler_angles


class Drone:
    def __init__(self, mass=1, drag_coef=0.5, max_rates=200, dt=1e-2):
        self.dim = 3
        self.action_scale = 1000
        self.state = None # [x, y, z, vx, vy, vz] in 3d
        self.rotation_matrix = None # R [3x3] in 3d
        self.acceleration = None
        self.rates = None
        self.thrust = None
        self.gravity_force = None
        self.drag_force = None
        self.total_forces = None
        self.dt = dt
        self.mass = mass
        self.drag_coef = drag_coef
        self.thrust_multiplier = 50
        self.max_rates = max_rates #deg/s
        self.prev_rates = None
        self.prev_thrust = None
        self.rates_transition_rate = 0.5
        self.thrust_transition_rate = 0.5

    def reset(self, position, velocity, rotation_matrix):
        self.state = np.zeros(2 * self.dim)
        self.state[:self.dim] = position
        self.state[self.dim:] = velocity
        self.rotation_matrix = rotation_matrix
        self.acceleration = np.zeros(self.dim)
        self.rates = np.zeros(self.dim)
        self.thrust = np.zeros(self.dim)
        self.gravity_force = np.zeros(self.dim)
        self.drag_force = np.zeros(self.dim)
        self.total_forces = np.zeros(self.dim)
        self.prev_rates = np.zeros(self.dim)
        self.prev_thrust = 0

    @property
    def position(self):
        return self.state[:self.dim]

    @property
    def velocity(self):
        return self.state[self.dim:]

    def action2force(self, action):
        """
        :param action: [roll_rate/self.action_scale, pitch_rate/self.action_scale, yaw_rate/self.action_scale, throttle]
                       throttle is expected to be in range [0, 1]
        :return: thrust [3x1], rates [3x1]  in world reference frame
        """
        self.rates[:self.dim] = np.clip(action[:self.dim] * self.action_scale, -self.max_rates, self.max_rates)
        rates = self.rates[:self.dim] * self.rates_transition_rate + \
                self.prev_rates * (1 - self.rates_transition_rate)
        self.prev_rates = rates
        thrust_scalar = np.clip(action[self.dim], 0, 1) * self.thrust_multiplier * self.thrust_transition_rate + \
                        self.prev_thrust * (1 - self.thrust_transition_rate)
        self.prev_thrust = thrust_scalar
        trust = kinematics.thrust_vector(thrust_scalar, self.rotation_matrix)
        return trust, rates

    def update(self):
        self.state, self.rotation_matrix = kinematics.update_kinematic_step(self.state, self.rotation_matrix, self.acceleration, self.rates, self.dt)
        self.rotation_matrix = kinematics.rotate_body_by_rates(self.rotation_matrix, self.rates, self.dt)

    def step(self, action, wind_velocity_vector):
        """
        :param action: [roll_rate/self.action_scale, pitch_rate/self.action_scale, yaw_rate/self.action_scale, throttle]
        :param wind_velocity_vector: wind vector in world reference frame [3x1]
        :return: rotation_matrix [3x3] (how the world is rotated with respect to the drone), gyro_matrix, accelorometer
                !!! IRL the drone doesn't know its state: Only IMU measurements !!!
        """
        self.thrust, self.rates = self.action2force(action)
        self.drag_force = kinematics.calculate_drag(self.state, wind_velocity_vector, self.drag_coef)
        self.gravity_force = kinematics.gravity_vector(self.mass, g=9.81)
        self.total_forces = self.thrust + self.gravity_force + self.drag_force
        self.acceleration = self.total_forces / self.mass
        self.update()
        angular_velocity_matrix = kinematics.rotation_matrix_from_euler_angles(*self.rates)
        return self.rotation_matrix.T, angular_velocity_matrix, self.rotation_matrix @ self.acceleration

    def render(self, ax, rpy=True, velocity=False, thrust=False, drag=False, gravity=False, total_force=True):
        render3d.plot_3d_points(ax, self.position, color='k')
        if rpy:
            render3d.plot_3d_rotation_matrix(ax, self.rotation_matrix, self.position, scale=0.5)
        if velocity:
            render3d.plot_3d_arrows(ax, self.position, self.velocity, color='m', alpha=0.5)
        if thrust:
            render3d.plot_3d_arrows(ax, self.position, self.thrust, color='c', alpha=0.5)
        if drag:
            render3d.plot_3d_arrows(ax, self.position, self.drag_force, color='y', alpha=0.5)
        if gravity:
            render3d.plot_3d_arrows(ax, self.position, self.gravity_force, color='g', alpha=0.5)
        if total_force:
            render3d.plot_3d_arrows(ax, self.position, self.total_forces, color='k', alpha=0.5)


class Camera:
    def __init__(self, fov, resolution, focal_length, position, rotation_matrix):
        self.fov = fov
        self.resolution = resolution
        self.focal_length = focal_length
        self.position = position
        self.rotation_matrix = rotation_matrix
        self.image = None

    def reset(self, position, rotation_matrix):
        self.position = position
        self.rotation_matrix = rotation_matrix
        self.image = np.zeros(self.resolution)

    def update(self, drone_position, drone_rotation_matrix):
        self.position = drone_position
        self.rotation_matrix = drone_rotation_matrix

    def render(self):

        return self.image


class Target:
    def __init__(self, position, radius):
        self.position = position
        self.radius = radius

    def update(self, position):
        self.position = position

    def render(self):
        pass

class Gate:
    def __init__(self, position, rotation_matrix, size, shape="rectangle", resolution=17):
        self.position = position
        self.rotation_matrix = rotation_matrix
        self.size = size
        if shape == "rectangle":
            self.corners = np.array([[0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1]]) * size / 2
            self.corners += self.position
            self.corners = (self.rotation_matrix @ self.corners.T).T
        elif "circle" in shape:
            coef_factor = 1 if "half" in shape else 2
            theta = np.linspace(0, coef_factor * np.pi, resolution)
            y = np.cos(theta) * size / coef_factor
            z = np.sin(theta) * size / coef_factor
            x = np.zeros_like(y)
            self.corners = np.vstack((x, y, z)).T
            self.corners += self.position
            if "half" in shape:
                self.corners -= np.array([0, 0, size/2])
            self.corners = (self.rotation_matrix @ self.corners.T).T
        else:
            raise NotImplementedError
        self.corners = np.vstack((self.corners, self.corners[0]))

    @property
    def normal(self):
        return self.rotation_matrix[:, 0]

    def calculate_plane_equation(self):
        normal = self.normal
        d = -np.dot(normal, self.position)
        return np.append(normal, d)

    def calculate_distance(self, point):
        normal = self.normal
        d = -np.dot(normal, self.position)
        return np.dot(normal, point) + d

    def render(self, ax, mid_point=True, gate_color="blue", arrow_color="orange", text="", fill_color="blue", alpha=0.5):
        if mid_point:
            render3d.plot_3d_arrows(ax, self.position, self.rotation_matrix[:, 0], color=arrow_color)
        ax.text(self.position[0], self.position[1] - self.size/2, self.position[2] + self.size, text, color=arrow_color)
        triang = mtri.Triangulation(self.corners[:, 1], self.corners[:, 2])
        ax.plot_trisurf(self.corners[:, 0], self.corners[:, 1], self.corners[:, 2], triangles=triang.triangles, color=fill_color, alpha=alpha)
        render3d.plot_3d_line(ax, self.corners, color=gate_color)


if __name__ == '__main__':
    """#Gates test
    ax, fig = render3d.init_3d_axis()
    raduis = 4
    n_gates = 20
    theta = np.linspace(0, 2 * np.pi, n_gates + 1)[:-1]
    gates_positions = np.vstack((np.cos(theta) * raduis, np.sin(theta) * raduis, np.zeros_like(theta))).T
    gates = []
    shapes = ["rectangle", "circle", "half_circle"]
    for i, p in enumerate(gates_positions):
        gates.append(Gate(p, np.eye(3), 1, shape=shapes[i % 3]))
    for gate in gates:
        gate.show(ax, text="Gate")
    render3d.show_plot(ax, fig, edge=raduis+1)
    plt.render()
    """
    #transition test
    drone = Drone(drag_coef=0.24, dt=5e-2)
    drone.reset(position=np.array([0, 0, 0]), velocity=np.array([0, 0, 0]), rotation_matrix=rotation_matrix_from_euler_angles(*np.array([0, 0, 0])))
    N = 1000
    rates_array = np.zeros((N, 3))
    thrust_array = np.zeros((N, 1))
    position_array = np.zeros((N, 3))
    wind_velocity_vector = np.array([0, 0, 0])
    rates_array[0, :] = drone.prev_rates
    thrust_array[0, :] = drone.prev_thrust
    position_array[0, :] = drone.position
    action = np.random.uniform(-1, 1, 4)
    action[-1] = (action[-1] + 1) / 2
    action[:-1] *= drone.max_rates / drone.action_scale
    ax, fig = render3d.init_3d_axis()
    for i in range(1, N):
        ax.clear()
        if i % 50 == 0:
            action = np.random.uniform(-1, 1, 4)
            action[-1] = (action[-1] + 1) / 2
            action[:-1] *= drone.max_rates/drone.action_scale * 0.1
            # action = np.array([0, 0, 0, 0])
        drone.step(action=action, wind_velocity_vector=wind_velocity_vector)
        drone.render(ax, rpy=True, velocity=True, thrust=True, drag=True, gravity=False, total_force=True)
        position_array[i, :] = drone.position
        render3d.plot_3d_line(ax, position_array[:i, :], color="blue", alpha=0.4)
        render3d.show_plot(ax, fig, middle=drone.position, edge=20)
        # rates_array[i, :] = drone.prev_rates
        # thrust_array[i, :] = drone.prev_thrust
        # plt.subplot(2, 1, 1)
        # plt.plot(rates_array[:i, :])
        # plt.subplot(2, 1, 2)
        # plt.plot(thrust_array[:i, :])
        # plt.pause(0.01)

    plt.show()
