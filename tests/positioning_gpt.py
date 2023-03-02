import numpy as np

# True position of the target
true_pos = np.array([1.0, 2.0, 3.0])

# Measured distances from the sensors (without noise)
measured_dist = np.array([3.0, 4.0, 5.0, 6.0])

# Standard deviation of the measurement error (in meters)
error_std = 0.1

# Generate Gaussian noise with zero mean and std equal to error_std
noise = np.random.normal(loc=0.0, scale=error_std, size=4)

# Add noise to the measured distances
noisy_dist = measured_dist + noise

# Calculate the estimated target position using the noisy distances
# (Use the same least-squares method as in the previous example)
A = 2 * np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])
b = np.array([
    noisy_dist[0]**2 - noisy_dist[3]**2 - true_pos.dot(true_pos) + np.sum(A[0]**2),
    noisy_dist[1]**2 - noisy_dist[3]**2 - true_pos.dot(true_pos) + np.sum(A[1]**2),
    noisy_dist[2]**2 - noisy_dist[3]**2 - true_pos.dot(true_pos) + np.sum(A[2]**2)
])
x = np.linalg.lstsq(A, b, rcond=None)[0]

# The estimated target position with measurement error
est_pos_with_error = x
print(est_pos_with_error)
print(true_pos)