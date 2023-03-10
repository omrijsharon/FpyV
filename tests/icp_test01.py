import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def angle2rotation_matrix(theta):
    # Compute the rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    return R

def icp(a, b, init_pose=(0,0,0), no_iterations = 13):
    '''
    The Iterative Closest Point estimator.
    Takes two cloudpoints a[x,y], b[x,y], an initial estimation of
    their relative pose and the number of iterations
    Returns the affine transform that transforms
    the cloudpoint a to the cloudpoint b.
    Note:
        (1) This method works for cloudpoints with minor
        transformations. Thus, the result depents greatly on
        the initial pose estimation.
        (2) A large number of iterations does not necessarily
        ensure convergence. Contrarily, most of the time it
        produces worse results.
    '''

    src = np.array([a.T], copy=True).astype(np.float32)
    dst = np.array([b.T], copy=True).astype(np.float32)

    #Initialise with the initial pose estimation
    Tr = np.array([[np.cos(init_pose[2]),-np.sin(init_pose[2]),init_pose[0]],
                   [np.sin(init_pose[2]), np.cos(init_pose[2]),init_pose[1]],
                   [0,                    0,                   1          ]])

    src = cv2.transform(src, Tr[0:2])

    for i in range(no_iterations):
        #Find the nearest neighbours between the current source and the
        #destination cloudpoint
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto',
                                warn_on_equidistant=False).fit(dst[0])
        distances, indices = nbrs.kneighbors(src[0])

        #Compute the transformation between the current source
        #and destination cloudpoint
        T = cv2.estimateRigidTransform(src, dst[0, indices.T], False)
        #Transform the previous source and update the
        #current source cloudpoint
        src = cv2.transform(src, T)
        #Save the transformation from the actual source cloudpoint
        #to the destination
        Tr = np.dot(Tr, np.vstack((T,[0,0,1])))
    return Tr[0:2]



if __name__ == '__main__':
    # coefficients of the polynomial
    coeffs = np.random.randn(5)
    theta = np.deg2rad(20)
    translation = np.random.randn(2)
    # value of x
    x = np.linspace(-1, 1, 10)
    # value of the polynomial at x
    y = np.polyval(coeffs, x)
    # concatenate the x and y values
    xy = np.vstack((x, y)).T
    # rotate by angle theta
    xy_rot = np.dot(xy, angle2rotation_matrix(theta))
    xy_trans = xy_rot + translation
    xy_new = xy_trans
    R, t, error = icp(xy, xy_new)
    xy_new_est = np.dot(R, xy.T).T + t
    # plot the original points
    plt.plot(xy[:, 0], xy[:, 1], 'o')
    # plot the rotated points
    plt.plot(xy_new_est[:, 0], xy_new_est[:, 1], 'o')
    plt.show()

