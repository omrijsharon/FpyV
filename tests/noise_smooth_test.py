import numpy as np
import matplotlib.pyplot as plt

x_list = np.array([])
x_smooth_list = np.array([])
transition = 0.1
prev_x = 0
for i in range(1000):
    plt.clf()
    x = np.random.randn(1)
    x_smooth = prev_x * (1 - transition) + x * transition
    prev_x = x_smooth
    x_list = np.append(x_list, x)
    x_smooth_list = np.append(x_smooth_list, x_smooth)
    plt.plot(x_list)
    plt.plot(x_smooth_list)
    plt.pause(0.01)

plt.show()
