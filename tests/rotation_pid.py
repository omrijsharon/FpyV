import numpy as np
from utils.components import PID
import gym
from time import sleep
from utils.helper_functions import rotation_matrix_from_euler_angles
from utils.kinematics import rotate_body_by_rates
from utils.render3d import plot_3d_rotation_matrix, init_3d_axis, show_plot

class Rotate(gym.Env):
    def __init__(self, dt=1e-2, max_rates=1000, threshold=1e-3):
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
        self.ax, self.fig = init_3d_axis()

    def convert_action_to_rates(self, action):
        return action * self.max_rates

    def cat_goal_state(self):
        return np.concatenate([np.expand_dims(self.goal_state, axis=2), np.expand_dims(self.current_state, axis=2)], axis=2)

    def get_error(self):
        return (self.goal_state.T @ self.current_state - np.eye(3)).sum()

    def reset(self):
        self.goal_state = rotation_matrix_from_euler_angles(*np.random.uniform(-np.pi, np.pi, size=3))
        self.current_state = rotation_matrix_from_euler_angles(*np.random.uniform(-np.pi, np.pi, size=3))
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
        plot_3d_rotation_matrix(self.ax, self.goal_state, np.zeros((3,)), scale=1.0)
        show_plot(self.ax, self.fig, middle=None, edge=1.0, title=None, xlabel=None, ylabel=None, zlabel=None, equal=True, grid=True, legend=True)

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def configure(self, *args, **kwargs):
        pass

    def __call__(self, x):
        return self.step(x)


if __name__ == '__main__':
    env = Rotate()
    env.reset()
    for i in range(100):
        env.step(env.action_space.sample())
        env.render()
        sleep(0.1)