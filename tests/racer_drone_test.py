import numpy as np
from scipy.spatial.transform import Rotation as R
from copy import copy
import matplotlib.pyplot as plt
from drawnow import drawnow


dt = 1e-3


class PID:
    def __init__(self, pid_values):
        self.i_error = None
        self.last_error = None
        self.pid_values = pid_values

    def reset(self):
        self.i_error = 0
        self.last_error = 0
        self.is_first = True

    def step(self, actual_value, desired_value):
        error = desired_value - actual_value
        p_error = error
        self.i_error += error * dt
        d_error = (error - self.last_error) / dt
        if self.is_first:
            d_error = 0
            self.is_first = False
        errors = np.array([p_error, self.i_error, d_error])
        self.last_error = copy(error)
        return self.pid_values @ errors



class Ball:
    def __init__(self, mass, pid_values):
        self.position = np.zeros(shape=(2,))
        self.velocity = np.zeros(shape=(2,))
        self.acceleration = np.zeros(shape=(2,))
        self.mass = mass
        self.pid = [PID(pid_values), PID(pid_values)]
        self.desired_position = None

    def reset(self):
        self.position = np.zeros(shape=(2,))
        self.velocity = np.zeros(shape=(2,))
        self.acceleration = np.zeros(shape=(2,))
        [self.pid[i].reset() for i in range(2)]

    def step(self, desired_position):
        self.desired_position = desired_position
        force = [self.pid[i].step(self.position[i], desired_position[i]) for i in range(2)]
        self.acceleration = np.array(force)/self.mass
        self.velocity = 1 * self.velocity + self.acceleration * dt
        self.position += self.velocity * dt

    def make_fig(self):
        plt.scatter(*self.position)
        plt.scatter(*self.desired_position)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)

    def render(self):
        drawnow(self.make_fig)


class Racer:
    def __init__(self, prop_size_inch, pid_values):
        self.r = (prop_size_inch/2) * 2.54 / 100 #drone radius in meters
        self.position = np.zeros(shape=(3,))
        self.linear_velocity = np.zeros(shape=(3,))
        self.orientation = R.from_matrix(np.eye(3))
        self.angular_velocity = np.zeros(shape=(3,))
        self.force = np.zeros(shape=(3,))
        self.acceleration = np.zeros(shape=(3,))
        self.torque = np.zeros(shape=(3,))
        self.pid = {"roll": PID(pid_values=pid_values["roll"]),
                    "pitch": PID(pid_values=pid_values["pitch"]),
                    "yaw": PID(pid_values=pid_values["yaw"])
                    }
        self.mass = 0.5
        self.I = self.mass * self.r**2 * np.ones(shape=(3,))

    def reset(self):
        self.position = np.zeros(shape=(3,))
        self.linear_velocity = np.zeros(shape=(3,))
        self.orientation = R.from_matrix(np.eye(3))
        self.angular_velocity = np.zeros(shape=(3,))
        self.force = np.zeros(shape=(3,))
        self.acceleration = np.zeros(shape=(3,))
        self.torque = np.zeros(shape=(3,))
        {v.reset() for v in self.pid.values()}

    def step(self, action):
        self.torque = np.array([v.step(self.angular_velocity[i], action[i]) for i, v in enumerate(self.pid.values())])
        print(self.torque)
        self.angular_velocity = 1 * self.angular_velocity + self.torque * dt / self.I
        self.orientation = R.from_matrix(self.orientation.as_matrix() @ R.from_euler("XYZ", self.angular_velocity).as_matrix())
        self.force = action[3] * self.orientation.as_matrix()[:, 2]
        self.acceleration = self.force / self.mass
        self.linear_velocity = 0.9 * self.linear_velocity + self.acceleration * dt
        self.position += self.linear_velocity * dt





if __name__ == '__main__':
    def make_fig():
        plt.plot(np.arange(len(roll_list))*dt, np.array(roll_list))

    env = Racer(prop_size_inch=5, pid_values={"roll": [2, 0, 0], "pitch": [2, 0, 0], "yaw": [0.1, 0, 0]})
    env.reset()
    roll_list = []
    action = [80, 10, 0, 0]
    for t in range(1000):
        env.step(action=action)
        roll_list.append(env.angular_velocity)
        drawnow(make_fig)
        if t==20:
            action = [-30, -50, 0, 0]
    plt.show()

    """
    env = Ball(0.1, pid_values=[120, 2, 20])
    env.reset()
    c = 0.05
    desired_position = 2 * np.random.rand(2) - 1

    for t in range(1000):
        # desired_position = 0.5 * np.array([np.sin(c * t), np.cos(c * t)])
        env.step(desired_position=desired_position)
        env.render()

        if t%10==0:
            desired_position = 2 * np.random.rand(2) - 1

    """