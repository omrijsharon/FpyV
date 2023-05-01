import matplotlib.pyplot as plt
import numpy as np


def second_order_fit(x, y, tol=1e-8, max_iter=100):
    # initialize coefficients to zero
    a = b = c = 0
    # initialize previous residual to a large number
    prev_residual = float('inf')
    # perform iterative least-squares fitting
    for i in range(max_iter):
        # compute denominator of normal equations
        denom = sum([(xi - x.mean()) ** 2 for xi in x])
        # compute coefficients using normal equations
        a = sum([(xi - x.mean()) ** 2 * yi for xi, yi in zip(x, y)]) / denom
        b = sum([(xi - x.mean()) * yi for xi, yi in zip(x, y)]) / denom
        c = y.mean() - a * x.mean() - b * x.mean() ** 2
        # compute residual
        residual = sum([(yi - a * xi ** 2 - b * xi - c) ** 2 for xi, yi in zip(x, y)])
        # check for convergence
        if abs(residual - prev_residual) < tol:
            break
        # update previous residual
        prev_residual = residual
    # compute total sum of squares
    tss = sum([(yi - y.mean()) ** 2 for yi in y])
    # compute residual sum of squares
    rss = sum([(yi - a * xi ** 2 - b * xi - c) ** 2 for xi, yi in zip(x, y)])
    # compute R-squared
    r_squared = 1 - rss / tss
    # return coefficients and R-squared
    return a, b, c, r_squared

def is_peak_altitude(time, measurements):
    num_measurements = len(measurements)
    max_altitude = measurements[0]
    counter = 0

    a,b,c = second_order_fit(time, measurements)
    # Check for peak altitude
    for i in range(num_measurements):
        expected_altitude = a * x[i] ** 2 + b * x[i] + c
        if measurements[i] > max_altitude:
            max_altitude = measurements[i]
            counter = 0
        else:
            counter += 1
        if counter >= 3 and measurements[i] < expected_altitude:
            plt.plot(x, y, 'o')
            plt.plot(x, a * x ** 2 + b * x + c, '-')
            plt.xlabel('Measurement index')
            plt.ylabel('Altitude (m)')
            plt.title('Peak altitude detection')
            plt.show()
            return True

    return False

if __name__ == '__main__':
    x = np.linspace(0, 3, 100)
    y = -x**2 + 2*x + 2
    print(second_order_fit(x, y))
    noise = np.random.normal(0, 0.1, 100)
    y += 0*noise
    print(is_peak_altitude(x, y))