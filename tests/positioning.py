import numpy as np

n_sensors = 4
n_dim = 3
noise_level = 0.0
coord = 5 * np.random.randn(n_sensors, n_dim)
target = 0.1 * np.random.randn(1, n_dim)
dist = np.linalg.norm(coord - target, axis=1)
dist += np.random.randn(len(dist)) * noise_level

# Initial guess for target coordinates
x = np.zeros((1, n_dim)) + 0*target + 0.01*np.random.randn(1, n_dim)

# Learning rate for gradient descent
learning_rate = 1e-7

# Momentum for gradient descent
momentum = 0.0
previous_update = np.zeros((1, n_dim))

# Maximum number of iterations for gradient descent
max_iterations = 1000

# MSE threshold for stopping iteration
mse_threshold = 1e-15

# Iterative gradient descent algorithm with momentum
for i in range(max_iterations):
    # Calculate squared Euclidean distances between current target estimate and sensors
    dist_est = np.sum((coord - x) ** 2, axis=1)

    # Calculate error between estimated distances and measured distances
    error = np.mean(dist_est - dist ** 2)

    # Calculate gradient of MSE loss function
    # grad = -2 * np.mean((coord - x), axis=0) * error
    grad = -np.mean(coord * np.exp(-np.linalg.norm(coord - x, axis=1))[:, np.newaxis] * (np.reshape(dist_est, (n_sensors, 1)) - dist ** 2), axis=0)

    # Update target estimate using gradient descent with momentum
    update = momentum * previous_update - learning_rate * grad
    x += update

    # Update previous update
    previous_update = update

    # Calculate MSE loss
    mse = np.sqrt(error ** 2)

    # Check if MSE is below threshold
    if mse < mse_threshold:
        break

    print("Iteration:", i, "MSE:", mse)

print("Target coordinates:", x)
print("True coordinates:", target)