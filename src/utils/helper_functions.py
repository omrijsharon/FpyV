import numpy as np

# WORLD2CAM = np.array([[0, 0, 1],
#                       [1, 0, 0],
#                       [0, -1, 0]]) # converts xyz to uvw

WORLD2CAM = np.array([[0, 1, 0],
                      [0, 0, -1],
                      [1, 0, 0]]) # converts xyz to uvw

def intrinsic_matrix(fx, fy, cx, cy):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def extrinsic_matrix(rot_mat, xyz):
    return np.vstack((np.hstack((rot_mat, xyz.reshape(-1, 1))), np.array([0, 0, 0, 1])))


def rotation_matrix(angle, axis: str):
    dR = np.eye(3)
    if axis == 'x':
        dR[1, 1] = np.cos(angle)
        dR[1, 2] = -np.sin(angle)
        dR[2, 1] = np.sin(angle)
        dR[2, 2] = np.cos(angle)
    elif axis == 'y':
        dR[0, 0] = np.cos(angle)
        dR[0, 2] = np.sin(angle)
        dR[2, 0] = -np.sin(angle)
        dR[2, 2] = np.cos(angle)
    elif axis == 'z':
        dR[0, 0] = np.cos(angle)
        dR[0, 1] = -np.sin(angle)
        dR[1, 0] = np.sin(angle)
        dR[1, 1] = np.cos(angle)
    return dR


def rotation_matrix_from_euler_angles(roll, pitch, yaw):
    R_x = rotation_matrix(roll, 'x')
    R_y = rotation_matrix(pitch, 'y')
    R_z = rotation_matrix(yaw, 'z')
    return R_z @ R_y @ R_x
    # return R_x @ R_y @ R_z


def rotation_matrix_to_quaternion(R):
    # https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    #
    # | 1-2b^2-2c^2  2bc-2ad      2bd+2ac      |
    # | 2bc+2ad      1-2a^2-2c^2  2cd-2ab      |
    # | 2bd-2ac      2cd+2ab      1-2a^2-2b^2  |
    #
    # a = qw
    # b = qx
    # c = qy
    # d = qz
    qw = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
    qx = (R[2, 1] - R[1, 2]) / (4 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw)
    return np.array([qw, qx, qy, qz])


def distance_point_to_plane(point, plane):
    #plane = [a, b, c, d]
    return np.abs(np.dot(point, plane[:3]) + plane[3]) / np.linalg.norm(plane[:3])


def focal_length_from_fov(fov, width, deg=True):
    """
    :param fov: field of view in degrees or radians
    :param width: resolution width in pixels
    :param deg: is fov in degrees or radians
    :return: focal length in pixels
    """
    if deg:
        fov = np.deg2rad(fov)
    return width / (2 * np.tan(fov / 2))


def quaternion_to_rotation_matrix(q):
    # https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    #
    # | 1-2b^2-2c^2  2bc-2ad      2bd+2ac      |
    # | 2bc+2ad      1-2a^2-2c^2  2cd-2ab      |
    # | 2bd-2ac      2cd+2ab      1-2a^2-2b^2  |
    #
    # a = qw
    # b = qx
    # c = qy
    # d = qz
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    return np.array([[1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
                     [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw],
                     [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx ** 2 - 2 * qy ** 2]])
