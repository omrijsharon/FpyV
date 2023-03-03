import numpy as np
import matplotlib.pyplot as plt
import gym

# environment of a ball that can move in 2D
class BallEnv(gym.Env):
    def __init__(self, map_size=3):
        super(BallEnv, self).__init__()
        self.map_size = map_size
        self.action_space = gym.spaces.box.Box(shape=(2,), low=-1, high=1)
        self.observation_space = gym.spaces.Box(shape=(1,), low=-np.inf, high=np.inf)
        self.state = None
        self.obs = None
        self.goal = None
        self.reward = None
        self.done = False
        self.threshold = 0.1
        self.info = {}

    def reset(self):
        self.state = np.random.uniform(-1, 1, size=(2,))
        self.goal = np.random.uniform(-1, 1, size=(2,))
        self.done = False
        self.obs = np.linalg.norm(self.state - self.goal)
        return self.obs

    def step(self, action):
        self.state += action
        self.obs = np.linalg.norm(self.state - self.goal)
        self.reward = -self.obs
        self.done = self.obs < self.threshold
        return self.obs, self.reward, self.done, {}

    def render(self, mode='human'):
        edge = 1
        # plt.scatter(self.state[0], self.state[1], 5, c='b', alpha=1-self.obs/2)
        plt.scatter(self.state[0], self.state[1], 5, c='b')
        plt.scatter(self.goal[0], self.goal[1], 5, c='g')
        plt.xlim(-edge, edge)
        plt.ylim(-edge, edge)


class ProportionalNavigation:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.prev_observation = None

    def choose_action(self, observation):
        # Implement proportional navigation using only distance information
        if self.prev_observation is not None:
            distance = observation
            velocity = distance - self.prev_observation
            direction = np.sign(velocity)
            # Calculate the course correction
            course_correction = np.abs(distance) * direction
            # Update the drone's velocity
            new_velocity = course_correction
            # Clip the velocity to the action space bounds
            new_velocity = np.clip(new_velocity, self.action_space.low, self.action_space.high)
            # Save the current observation for the next step
            self.prev_observation = distance
            return new_velocity
        else:
            # If this is the first step, choose a random action
            action = self.action_space.sample()
            # Save the current observation for the next step
            self.prev_observation = observation
            return action

if __name__ == '__main__':
    # env = BallEnv()
    # obs = env.reset()
    # speed_coef = 0.1
    # theta = np.random.uniform(0, 2*np.pi)
    # action = speed_coef * np.array([np.cos(theta), np.sin(theta)])
    # while not env.done:
    #     plt.clf()
    #     next_obs, reward, done, info = env.step(action)
    #     if obs < next_obs:
    #         theta = (theta + np.pi) % (2*np.pi)
    #         action = 2 * speed_coef * np.array([np.cos(theta), np.sin(theta)])
    #     else:
    #         ratio = np.abs(next_obs - obs)/speed_coef
    #         theta = (theta + np.pi/(1 * ratio) * np.random.randn()) % (2*np.pi)
    #         action = speed_coef * np.array([np.cos(theta), np.sin(theta)])
    #     obs = next_obs
    #     env.render()
    #     plt.pause(0.001)
    #     if done:
    #         break
    # plt.show()

    # env = BallEnv()
    # obs = env.reset()
    # speed_coef = 0.05
    # action = speed_coef * np.array([1, 0])
    # counter = 0
    # while not env.done:
    #     # plt.clf()
    #     next_obs, reward, done, info = env.step(action)
    #     if obs < next_obs and counter == 0:
    #         action = -2 * action
    #         counter += 1
    #     else:
    #         action = speed_coef * (action==0)
    #         counter = 0
    #     obs = next_obs
    #     env.render()
    #     plt.pause(0.001)
    #     if done:
    #         break
    # plt.show()

    # Create the environment
    env = BallEnv()

    # Create the agent
    agent = ProportionalNavigation(env)

    # Reset the environment
    observation = env.reset()

    # Start the episode
    done = False
    while not done:
        # plt.clf()
        # Choose an action using the agent
        action = agent.choose_action(observation)
        # Take a step in the environment
        observation, reward, done, info = env.step(action)
        env.render()
        plt.pause(0.01)
    plt.show()

