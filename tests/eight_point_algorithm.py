# made by GPT chatbot. Please test this code before using it in production.

import numpy as np


def eight_point(points1, points2):
    """
    This function computes the fundamental matrix from a set of corresponding points
    params:
        points1: points in image 1
        points2: points in image 2
        returns:
        F: fundamental matrix
    """
    # Normalize the points
    points1 = points1 / points1[-1]
    points2 = points2 / points2[-1]

    # Create the matrix A
    A = np.zeros((points1.shape[0], 9))
    A[:, :8] = points1[:, :8] * points2[:, None, :8]
    A[:, 8] = -1

    # Compute the SVD of A
    U, S, V = np.linalg.svd(A)

    # The fundamental matrix is the last column of V
    F = V[-1].reshape((3, 3))

    # Enforce the rank-2 constraint
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ V

    # Unnormalize the fundamental matrix
    F = F / F[-1, -1]

    return F


def extract_extrinsic(F, K):
    """
    This function extracts the extrinsic parameters from the fundamental matrix given the intrinsic parameters
    params:
        F: fundamental matrix
        K: camera intrinsic matrix
        returns:
        R: rotation matrix
        t: translation vector
    """
    # Compute the SVD of the fundamental matrix
    U, S, V = np.linalg.svd(F)

    # Create the W matrix
    W = np.array([0, -1, 0,
                  1, 0, 0,
                  0, 0, 1]).reshape((3, 3))

    # The extrinsic matrix is the product of the inverse of the intrinsic
    # matrix, U, W, and V
    E = K.T @ U @ W @ V

    return E