import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import render3d, helper_functions, components
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from icosphere import icosphere
import mpl_toolkits.mplot3d

from utils.helper_functions import WORLD2CAM


def extrinsic_matrix(rotation_matrix, translation_vector):
    return np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))


# ball = components.Target(position=np.array([1, 0, 0]), radius=0.5, nu=3)
# ext_mat = extrinsic_matrix(rotation_matrix=WORLD2CAM.T, translation_vector=np.array([0, 0, 0]))
# int_mat = helper_functions.intrinsic_matrix()
# ax, fig = render3d.init_3d_axis()

translation_matrix = np.array(np.eye(4))
t = np.array([0.5, -1, 0])
translation_matrix[:3, 3] = t
coordinate = np.array([0.5, -1, 0, 1])
trans_coordinate = np.dot(translation_matrix, coordinate)
print(trans_coordinate)
