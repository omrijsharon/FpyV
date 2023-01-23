import numpy as np
import gym
from time import sleep
import matplotlib.pyplot as plt


def argmax2d(array):
    # returns the index of the maximum value in a 2d array
    return np.unravel_index(np.argmax(array), array.shape)


def combine(goal, state):
    # return np.append(goal, state)
    return state - goal


class MaComSimpleInstructions(gym.Env):
    def __init__(self, map_size=3):
        self.map_size = map_size
        self.action_space = gym.spaces.Dict({"Instructor": gym.spaces.Box(shape=(2,), low=-1, high=1), "Apprentice": gym.spaces.Discrete(5)})
        self.observation_space = gym.spaces.Dict({"Instructor": gym.spaces.Discrete(self.map_size**2), "Apprentice": gym.spaces.Box(shape=(2,), low=-1, high=1)})
        self.state = None
        self.goal = None
        self.reward = None
        self.done = False
        self.info = {}

    def reset(self):
        # goal
        self.goal = np.zeros((self.map_size ** 2))
        self.goal[np.random.randint(0, self.map_size ** 2)] = 1
        self.goal = self.goal.reshape((self.map_size, self.map_size))
        # state
        self.state = np.zeros((self.map_size ** 2))
        self.state[np.random.randint(0, self.map_size ** 2)] = 1
        self.state = self.state.reshape((self.map_size, self.map_size))
        # observation
        obs = {"Instructor": combine(self.goal, self.state), "Apprentice": np.zeros((2,))}
        self.done = False
        return obs

    def step(self, action):
        # action = {"Instructor": np.array([0.5, 0.5]), "Apprentice": 0}
        if action["Apprentice"] == 1:
            self.state = np.roll(self.state, 1, axis=0)
        elif action["Apprentice"] == 2:
            self.state = np.roll(self.state, -1, axis=0)
        elif action["Apprentice"] == 3:
            self.state = np.roll(self.state, 1, axis=1)
        elif action["Apprentice"] == 4:
            self.state = np.roll(self.state, -1, axis=1)
        else:
            pass
        self.reward = np.sum(self.state * self.goal)
        self.done = self.reward > 0
        obs = {"Instructor": combine(self.goal, self.state), "Apprentice": action["Instructor"]}
        return obs, self.reward, self.done, {}

    def render(self, mode='human'):
        print(self.state)

    def close(self):
        pass

    def seed(self, seed=None):
        pass

if __name__ == '__main__':
    env = MaComSimpleInstructions(map_size=11)
    obs = env.reset()
    print(obs)
    steps = 0
    while not env.done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        # print(obs, reward, done, info)
        steps += 1
        # env.render()
    print("steps: ", steps)
