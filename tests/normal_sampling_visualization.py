import numpy as np
import matplotlib.pyplot as plt

def pixelize_data(p):
    return np.array(list(set(tuple(map(tuple, p.astype(int))))))

if __name__ == '__main__':
    p1 = pixelize_data(10 * np.random.randn(20000, 2))
    p2 = pixelize_data(20 * np.random.randn(20000, 2) + np.array([1, -1]))
    plt.scatter(p1[:, 0], p1[:, 1], s=5, c='r', alpha=0.3)
    plt.scatter(p2[:, 0], p2[:, 1], s=5, alpha=0.3)
    plt.show()