import numpy as np
import matplotlib.pyplot as plt
from utils.components import PID
import gym
from time import sleep
from utils import get_sticks, yaml_helper
from utils.helper_functions import euler_angles_to_rotation_matrix, rotation_matrix_to_euler_angles, rotation_matrix_to_axis_angle, axis_angle_to_rotation_matrix
from utils.kinematics import rotate_body_by_rates
from utils.render3d import plot_3d_rotation_matrix, init_3d_axis, show_plot

class Rotate(gym.Env):
    def __init__(self, dt=1e-2, max_rates=1000, threshold=1e-3, difficulty=1.0, params=None):
        super().__init__()
        # Joystick
        self.rc = get_sticks.Joystick()
        self.run = self.rc.status
        if self.run:
            print("Joystick connected")
        else:
            print("Joystick device was not found")
        self.rc.calibrate(params["drone"]["joystick_calib_path"], load_calibration_file=True)

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

    def read_sticks(self):
        throttle, roll, pitch, arm, _, yaw = self.rc.calib_read()
        action = np.array([-roll, pitch, yaw, throttle])
        return action

    def convert_action_to_angle(self, action):
        action = action[:3] * np.pi
        action[1] /= 2
        return action

    def convert_action_to_rates(self, action):
        return action * self.max_rates

    def convert_rates_to_action(self, rates):
        return rates / self.max_rates

    def cat_goal_state(self):
        return np.concatenate([np.expand_dims(self.goal_state, axis=2), np.expand_dims(self.current_state, axis=2)], axis=2)

    def get_error(self):
        return ((self.goal_state.T @ self.current_state - np.eye(3))**2).sum()

    def reset(self):
        if self.run:
            goal_action = self.read_sticks()
            euler_goal = self.convert_action_to_angle(goal_action)
        else:
            euler_goal = np.random.uniform(0, 2 * np.pi, size=3)
        euler_current = (euler_goal + np.random.normal(0, self.difficulty, size=3)) % (2 * np.pi)
        self.goal_state = euler_angles_to_rotation_matrix(*euler_goal)
        self.current_state = euler_angles_to_rotation_matrix(*euler_current)
        self.done = False
        return self.cat_goal_state()

    def step(self, action):
        if self.run:
            goal_action = self.read_sticks()
            euler_goal = self.convert_action_to_angle(goal_action)
            self.goal_state = euler_angles_to_rotation_matrix(*euler_goal)
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


class RotationRatesController:
    def __init__(self, gain, max_rates, transition_coef: dict = None):
        self.gain = gain * np.ones(shape=(3,))
        self.max_rates = max_rates
        self.error = np.zeros(3)
        self.rates = np.zeros(3)
        self.axes_names = np.array(['x', 'y', 'z'])
        self.prev_state = np.zeros(3)
        self.prev_goal = np.zeros(3)
        self.prev_error = np.zeros(3)
        assert transition_coef is not None, "Transition coefficients must be specified"
        self.state_transition_coef = transition_coef["state"]
        self.goal_transition_coef = transition_coef["goal"]
        self.error_transition_coef = transition_coef["error"]

    def reset(self):
        self.error = np.zeros(3)
        self.rates = np.zeros(3)
        self.prev_state = np.zeros(3)
        self.prev_goal = np.zeros(3)
        self.prev_error = np.zeros(3)

    def get_rates(self, R_current, R_goal):
        euler_state = self.state_transition_coef * rotation_matrix_to_euler_angles(R_current) + (1 - self.state_transition_coef) * self.prev_state
        self.prev_state = euler_state
        R_current = euler_angles_to_rotation_matrix(*euler_state)
        euler_goal = self.goal_transition_coef * rotation_matrix_to_euler_angles(R_goal) + (1 - self.goal_transition_coef) * self.prev_goal
        self.prev_goal = euler_goal
        R_goal = euler_angles_to_rotation_matrix(*euler_goal)
        # Calculate the relative rotation matrix
        R_rel = np.matmul(R_goal.T, R_current)# working!
        # print(np.diag(R_rel))

        euler_error = self.error_transition_coef * rotation_matrix_to_euler_angles(R_rel) + (1 - self.error_transition_coef) * self.prev_error
        self.prev_error = euler_error

        self.error = euler_error
        # self.error = rotation_matrix_to_euler_angles(R_rel)
        self.rates = np.clip(self.gain * np.rad2deg(self.error), -self.max_rates, self.max_rates)
        return self.rates



if __name__ == '__main__':
    fps = 60
    dt = 1 / fps
    max_rates = 480
    params = yaml_helper.yaml_reader(r"C:\Users\omri_\PycharmProjects\FpyV\config\params.yaml")
    env = Rotate(dt=dt, max_rates=max_rates, threshold=1e-3, difficulty=1.0, params=params)
    # pids = [PID(kP=1, kI=0, kD=0, dt=env.dt, integral_clip=1, min_output=0.3, max_output=1, derivative_transition_rate=0.5) for _ in range(3)]
    transition_coef = {"state": 0.75, "goal": 0.9, "error": 0.9}
    controller = RotationRatesController(gain=30.0, max_rates=max_rates, transition_coef=transition_coef)
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
    env.render()
    noise_lvl = 5.0
    done = False
    # while env.done is False:
    while not done:
        goal = state[:, :, 0]
        current = state[:, :, 1]
        noise_angles = (np.random.normal(0, noise_lvl, size=3))
        # print(noise_angles)
        current = euler_angles_to_rotation_matrix(*np.deg2rad(noise_angles % (2 * np.pi))) @ current
        # relative = goal @ current.T
        # rpy = np.rad2deg(rotation_matrix_to_euler_angles(relative))
        # print(euler_angles_to_rotation_matrix(roll, pitch, yaw), "\n", relative)
        # action = controller.get_rates(current, goal, env.dt)
        rates = controller.get_rates(current, goal)
        action = env.convert_rates_to_action(rates)
        error_array = np.concatenate([error_array, np.expand_dims(controller.error, axis=0)], axis=0)
        rates_array = np.concatenate([rates_array, np.expand_dims(rates, axis=0)], axis=0)
        state, reward, _, info = env.step(action)
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