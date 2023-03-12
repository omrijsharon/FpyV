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


def euler_angles_to_rotation_matrix(roll, pitch, yaw):
    R_x = rotation_matrix(roll, 'x')
    R_y = rotation_matrix(pitch, 'y')
    R_z = rotation_matrix(yaw, 'z')
    return R_z @ R_y @ R_x
    # return R_x @ R_y @ R_z


def rotation_matrix_to_euler_angles(R):
    # https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToEuler/index.htm
    # Assuming the angles are in radians.
    if R[2, 0] != 1 or R[2, 0] != -1:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arcsin(-R[2, 0])
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        z = 0 # can be anything
        if R[2, 0] == -1:
            x = np.arctan2(R[0, 1], R[0, 2])
            y = np.pi / 2
        else:
            x = np.arctan2(-R[0, 1], -R[0, 2])
            y = -np.pi / 2
    return np.array([x, y, z])


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


def bbox3d(cls):
    @property
    def bbox3d_(self):
        """ this function creates a tight bounding box around a 3d object with vectices points """
        min_points = np.min(self.points, axis=0)
        max_points = np.max(self.points, axis=0)
        box = np.zeros((8, 3))
        box[:4, 0] = min_points[0] # min x
        box[4:, 0] = max_points[0] # max x
        box[::2, 1] = min_points[1] # min y
        box[1::2, 1] = max_points[1] # max y
        box[[0, 1, 4, 5], 2] = min_points[2] # min z
        box[[2, 3, 6, 7], 2] = max_points[2] # max z
        return box

    setattr(cls, 'bbox3d', bbox3d_)
    return cls


def distance(obj1, obj2):
    return np.linalg.norm(obj1.position - obj2.position)

def point_to_surface_distance(point, surface):
    """
    :param point: point in 3d space
    :param surface: surface in 3d space
    :return: distance from point to surface
    """
    return np.abs(np.dot(point, surface[:3]) + surface[3]) / np.linalg.norm(surface[:3])


def generate_circular_path(center, radius, resolution):
    theta = np.linspace(0, 2 * np.pi, resolution + 1)[:-1]
    return np.vstack((np.cos(theta) * radius, np.sin(theta) * radius, np.zeros_like(theta))).T + np.array(center)


def rotation_matrix_to_axis_angle(R):
    """
    Calculate the axis-angle representation from a rotation matrix.
    """
    angle = np.arccos((np.trace(R) - 1) / 2)
    if angle == 0:
        axis = np.array([1, 0, 0])
    elif angle == np.pi:
        s = np.sqrt((R[0,0]+1)/2)
        x = s*np.sign(R[0,1])
        y = s*np.sign(R[0,2])
        z = 0
        axis = np.array([x, y, z])
    else:
        axis = np.array([R[2,1] - R[1,2],
                         R[0,2] - R[2,0],
                         R[1,0] - R[0,1]])
        axis /= np.linalg.norm(axis)
    return axis, angle


def axis_angle_to_rotation_matrix(axis, angle):
    """
    Convert axis-angle representation to rotation matrix.

    :param axis: np.array, axis of rotation.
    :param angle: float, angle of rotation.
    :return: np.array, 3x3 rotation matrix.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x, y, z = axis
    return np.array([
        [t * x ** 2 + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y ** 2 + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z ** 2 + c]
    ])