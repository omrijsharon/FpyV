import numpy as np
import matplotlib.pyplot as plt

from utils.components import Drone

drone = Drone()
drone.reset(position=np.array([0, 0, 0]), velocity=np.array([0, 0, 0]), rotation_matrix=np.eye(3))
N = 100
rates_array = np.zeros((N, 3))
thrust_array = np.zeros((N, 1))
for i in range(N):
    plt.clf()
    thrust, rates = drone.action2force(action=np.array([0.2, 0.0, 0.0, 1]))
    rates_array[i, :] = rates
    thrust_array[i, :] = thrust
    plt.subplot(2, 1, 1)
    plt.plot(rates_array[:i, :])
    plt.subplot(2, 1, 2)
    plt.plot(thrust_array[:i, :])
    plt.pause(0.01)
plt.show()

