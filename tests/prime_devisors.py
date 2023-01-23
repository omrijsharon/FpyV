import numpy as np


def prime_devisors(n):
    prime_devisors = []
    for i in range(2, n):
        while n % i == 0:
            prime_devisors.append(i)
            n = n / i
    return prime_devisors


def resolution_divisors(resolution):
    x = list(map(prime_devisors, resolution))
    np.unique(x)


if __name__ == '__main__':
    n = np.array([[1980, 1080], [1280, 720], [800, 600], [640, 480], [320, 240], [160, 120]])
    print(list(map(prime_devisors, n.flatten())))
    print(resolution_divisors(n[0]))
