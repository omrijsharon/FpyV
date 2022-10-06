import numpy as np

WORLD2CAM = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]]) # converts xyz to uvw
CAM2WORLD = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]) # converts uvw to xyz


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