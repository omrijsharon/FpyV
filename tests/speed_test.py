import numpy as np


def timeit(func, n=100):
    import time
    def wrapper(*args, **kwargs):
        total_time = np.zeros(n)
        for i in range(n):
            start = time.perf_counter()
            func(*args, **kwargs)
            end = time.perf_counter()
            total_time[i] = end - start
        print(f"Average time: {total_time.mean()} ± {total_time.std()}")
    return wrapper

# test weather inplace operation is faster than creating a new array
# test 1
a = np.random.rand(1000, 1000)
b = np.zeros_like(a)

@timeit
def test1(a, b):
    for i in range(20):
        b *= 0

@timeit
def test2(a, b):
    for i in range(20):
        b = 0

test1(a, b)
test2(a, b)
