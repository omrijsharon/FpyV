import numpy as np
import matplotlib.pyplot as plt
from utils.components import PID
import gym
from time import sleep
from utils.helper_functions import euler_angles_to_rotation_matrix, rotation_matrix_to_euler_angles, rotation_matrix_to_axis_angle, axis_angle_to_rotation_matrix
from utils.kinematics import rotate_body_by_rates
from utils.render3d import plot_3d_rotation_matrix, init_3d_axis, show_plot

class Rotate(gym.Env):
    def __init__(self, dt=1e-2, max_rates=1000, threshold=1e-3, difficulty=1.0):
        super().__init__()
        self.goal_state: np.array = None
        self.current_state: np.array = None
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(3, 3, 2))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
        self.done = None
        self.info = {}
        self.dt = dt
        self.max_rates = max_rates
        self.threshold = threshold
        self.difficulty = difficulty
        self.ax, self.fig = init_3d_axis()

    def convert_action_to_rates(self, action):
        return action * self.max_rates

    def convert_rates_to_action(self, rates):
        return rates / self.max_rates

    def cat_goal_state(self):
        return np.concatenate([np.expand_dims(self.goal_state, axis=2), np.expand_dims(self.current_state, axis=2)], axis=2)

    def get_error(self):
        return ((self.goal_state.T @ self.current_state - np.eye(3))**2).sum()

    def reset(self):
        euler_goal = np.random.uniform(0, 2 * np.pi, size=3)
        euler_current = (euler_goal + np.random.normal(0, self.difficulty, size=3)) % (2 * np.pi)
        self.goal_state = euler_angles_to_rotation_matrix(*euler_goal)
        self.current_state = euler_angles_to_rotation_matrix(*euler_current)
        self.done = False
        return self.cat_goal_state()

    def step(self, action):
        rates = self.convert_action_to_rates(action)
        self.current_state = rotate_body_by_rates(self.current_state, rates, self.dt)
        if self.get_error() < self.threshold:
            self.done = True
        return self.cat_goal_state(), -self.get_error(), self.done, self.info


    def render(self, mode='human'):
        self.ax.clear()
        plot_3d_rotation_matrix(self.ax, self.current_state, np.zeros((3,)), scale=1.0)
        plot_3d_rotation_matrix(self.ax, self.goal_state, np.zeros((3,)), scale=1.0, alpha=0.5)
        show_plot(self.ax, self.fig, middle=None, edge=1.0, title=None, xlabel=None, ylabel=None, zlabel=None, equal=True, grid=True, legend=True)

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def configure(self, *args, **kwargs):
        pass

    def __call__(self, x):
        return self.step(x)


class PController:
    def __init__(self, Kp, max_rates):
        self.Kp = Kp
        self.max_rates = max_rates
        self.error = np.zeros(3)
        self.rates = np.zeros(3)
        self.axes_names = np.array(['x', 'y', 'z'])

    def reset(self):
        self.error = np.zeros(3)
        self.rates = np.zeros(3)

    def get_rates(self, R_current, R_goal, axis='x'):
        if axis is None:
            # Calculate the relative rotation matrix
            R_rel = np.matmul(R_goal, R_current.T) # working!
            # Convert the relative rotation matrix to axis-angle representation
            axis, angle = rotation_matrix_to_axis_angle(R_rel.T)
            self.error = angle * axis

            # Convert the relative rotation matrix to Euler angles
            # self.error = np.rad2deg(rotation_matrix_to_euler_angles(R_rel.T))
        else:
            idx = np.argwhere(self.axes_names == axis).item()
            angle = np.sign(np.dot(R_current[:, idx], R_goal[:, idx]))
            # R_axis = np.cross(R_current[:, idx], R_goal[:, idx])
            R_axis = np.cross(R_current[:, idx], R_goal[:, idx])
            # R_target = axis_angle_to_rotation_matrix(R_axis, angle)
            self.error = R_axis
        # self.error = angle * R_axis
        # print(f"error norm: {np.linalg.norm(self.error)}")
        # self.error = self.error / np.linalg.norm(self.error) * (direction-1)
        # Calculate the error and the rate of change of the error
        # self.error = np.cross(x_current, x_goal)


        # Calculate the error and the rate of change of the error
        # if self.error_prev is not None:
        #     error_dot = (error - self.error_prev) / dt
        # else:
        #     error_dot = np.zeros(3)
        # self.error_prev = self.error

        # # Calculate the rate of change of R:
        # if self.R_prev is not None:
        #     R_dot = np.dot(R_current, self.R_prev.T) / dt
        #     angular_velocity = axis * angle / dt + np.dot(R_dot, axis)
        # else:
        #     angular_velocity = np.zeros(3)
        # self.R_prev = R_current
        # error_dot = np.rad2deg(np.trace(R_dot @ R_rel.T))

        # Calculate the control signal using PD control
        # self.rates = np.clip(self.Kp * self.error, -self.max_rates, self.max_rates)
        self.rates = np.clip(self.Kp * np.rad2deg(self.error), -self.max_rates, self.max_rates)

        return self.rates


class PDController1axis:
    def __init__(self, Kp, Kd, max_torque):
        self.Kp = Kp
        self.Kd = Kd
        self.max_torque = max_torque
        self.R_target = None
        self.R_target_inv = None

    def get_torque(self, R_current, R_goal, dt):
        # Calculate the target rotation matrix
        if self.R_target is None:
            x_goal = R_goal[:, 0]
            x_current = R_current[:, 0]
            R_axis = np.cross(x_current, x_goal)
            angle = np.arccos(np.dot(x_current, x_goal))
            self.R_target = axis_angle_to_rotation_matrix(R_axis, angle)
            self.R_target_inv = self.R_target.T
        # Calculate the error and the rate of change of the error
        error = self.R_target_inv.dot(R_current) - np.eye(3)
        error_dot = -np.dot(self.R_target_inv, angular_velocity_from_rotation_matrix2(R_current, R_goal, dt))

        # Calculate the control signal using PD control
        torque = np.clip(self.Kp.dot(error.flatten()) + self.Kd.dot(error_dot.flatten()), -self.max_torque,
                         self.max_torque)

        return torque


if __name__ == '__main__':
    fps = 60
    dt = 1 / fps
    max_rates = 1000
    env = Rotate(dt=dt, max_rates=max_rates, threshold=1e-3, difficulty=0.1)
    # pids = [PID(kP=1, kI=0, kD=0, dt=env.dt, integral_clip=1, min_output=0.3, max_output=1, derivative_transition_rate=0.5) for _ in range(3)]
    controller = PController(Kp=40.0, max_rates=max_rates)
    state = env.reset()
    controller.reset()
    error_array = np.empty((0, 3))
    rates_array = np.empty((0, 3))
    reward_array = np.empty((0,))
    # [pid.reset() for pid in pids]
    is_plot = False
    plots_names = ["error roll", "error pitch", "error yaw", "rates roll", "rates pitch", "rates yaw", "reward"]
    if is_plot:
        fig, ax = plt.subplots(3, 3, sharex=True)
    axis = 'x'
    mask = np.ones(3, dtype=bool)
    while env.done is False:
        goal = state[:, :, 0]
        current = state[:, :, 1]
        # relative = goal @ current.T
        # rpy = np.rad2deg(rotation_matrix_to_euler_angles(relative))
        # print(euler_angles_to_rotation_matrix(roll, pitch, yaw), "\n", relative)
        # action = controller.get_rates(current, goal, env.dt)
        rates = controller.get_rates(current, goal, axis=axis) * mask
        if np.linalg.norm(controller.error) < 1e-2:
            axis = None
            controller.Kp *= 10.0
            mask[1:] = False
            print("mask: ", mask)
        action = env.convert_rates_to_action(rates)
        error_array = np.concatenate([error_array, np.expand_dims(controller.error, axis=0)], axis=0)
        rates_array = np.concatenate([rates_array, np.expand_dims(rates, axis=0)], axis=0)
        state, reward, done, info = env.step(action)
        reward_array = np.append(reward_array, reward)
        if is_plot:
            for i in range(3):
                ax[i, 0].cla()
                ax[i, 0].plot(error_array[:, i])
                ax[i, 0].set_ylabel(plots_names[i])
            ax[0, 2].cla()
            ax[0, 2].plot(reward_array)
            ax[0, 2].set_ylabel(plots_names[-1])
            for i in range(3):
                ax[i, 1].cla()
                ax[i, 1].plot(rates_array[:, i])
                ax[i, 1].set_ylabel(plots_names[i+3])
            plt.pause(0.000001)
        else:
            env.render()
        # sleep(0.000001)
    print("done")
    plt.show()