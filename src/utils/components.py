import cv2
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import mpl_toolkits.mplot3d

import numpy as np
from icosphere import icosphere

from utils import kinematics, render3d, get_sticks, yaml_helper
from utils.flight_time_calculator import read_motor_test_report, model_xy
from utils.helper_functions import rotation_matrix_from_euler_angles, intrinsic_matrix, WORLD2CAM


class Drone:
    def __init__(self, params):
        # Joystick
        self.rc = get_sticks.Joystick()
        run = self.rc.status
        if run:
            print("Joystick connected")
        else:
            print("Joystick device was not found")
        self.rc.calibrate(params["drone"]["joystick_calib_path"], load_calibration_file=True)

        # Drone parameters
        self.dim = 3
        self.max_rates = params["drone"]["max_rates"] #deg/s
        self.state = None # [x, y, z, vx, vy, vz] in 3d
        self.rotation_matrix = None # R [3x3] in 3d
        self.acceleration = None
        self.rates = None
        self.thrust = None
        self.gravity = params["simulator"]["gravity"]
        self.gravity_force = None
        self.drag_force = None
        self.total_forces = None
        self.dt = params["simulator"]["dt"]
        self.mass = params["drone"]["mass"] / 1000 #kg
        self.drag_coef = params["drone"]["drag_coefficient"]
        self.thrust_multiplier = 80
        self.prev_rates = None
        self.prev_thrust = None
        self.done = None
        self.rates_transition_rate = params["drone"]["rates_transition_rate"]
        self.thrust_transition_rate = params["drone"]["thrust_transition_rate"]
        self.camera = Camera(camera_pitch_angle=params["camera"]["camera_angle"],
                             position_relative_to_frame=np.array(params["camera"]["position_relative_to_frame"]),
                             fov=params["camera"]["fov"],
                             resolution=params["camera"]["resolution"],
                             focal_length=None
                             )

        # motors
        self.n_motors = 4
        self.motor_radius = 0.5
        self.radius = 5 * 2.54 / 100
        t = np.linspace(0, 2 * np.pi, self.n_motors + 1)[:-1]
        self.t = t + (t[1] - t[0]) / 2
        self.motors_relative_position = self.radius * np.array([np.cos(self.t), np.sin(self.t), np.zeros(self.n_motors)]).T
        self.motors_orientation = None

        self.motor_test_report = read_motor_test_report(params["drone"]["motor_test_report_path"])[params["drone"]["motor_test_report_idx"]]
        propeller = self.motor_test_report['Propeller'].values
        motor_name = self.motor_test_report['Type'].values
        extract_string = lambda x: x[list(map(isinstance, x, [str]*len(x)))][0]
        print(f"{extract_string(motor_name)}, {extract_string(propeller)}")
        # thrust is measured for a single motor in grams. We need to convert it to Newtons and multiply by the number of motors.
        thrust = self.n_motors * self.motor_test_report['Thrust'].values / 1000 * self.gravity # N
        throttle = self.motor_test_report['Throttle'].values
        self.throttle2thrust = lambda x: model_xy(throttle, thrust)(100 * (x + 1) / 2)
        self.thrust2throttle = lambda x: np.clip((model_xy(thrust, throttle)(x) / 100) * 2 - 1, -1, 1)

    def reset(self, position, velocity, ypr):
        self.state = np.zeros(2 * self.dim)
        self.state[:self.dim] = position
        self.state[self.dim:] = velocity
        self.rotation_matrix = rotation_matrix_from_euler_angles(*np.deg2rad(ypr))
        self.acceleration = np.zeros(self.dim)
        self.rates = np.zeros(self.dim)
        self.thrust = np.zeros(self.dim)
        self.gravity_force = np.zeros(self.dim)
        self.drag_force = np.zeros(self.dim)
        self.total_forces = np.zeros(self.dim)
        self.prev_rates = np.zeros(self.dim)
        self.prev_thrust = 0
        self.motors_orientation = self.motors_relative_position @ self.rotation_matrix.T
        self.camera.reset(drone_position=position, drone_rotation_matrix=self.rotation_matrix)
        self.done = False

    @property
    def position(self):
        return self.state[:self.dim]

    @property
    def velocity(self):
        return self.state[self.dim:]

    def action2force(self, action):
        """
        :param action: [roll_rate/self.action_scale, pitch_rate/self.action_scale, yaw_rate/self.action_scale, throttle]
                       throttle is expected to be in range [-1, 1]
        :return: thrust [3x1], rates [3x1]  in world reference frame
        """
        action2rates = np.clip(-action[:self.dim] * self.max_rates, -self.max_rates, self.max_rates)
        rates = action2rates * self.rates_transition_rate + \
                self.prev_rates * (1 - self.rates_transition_rate)
        self.prev_rates = rates
        # thrust_scalar = np.clip((action[self.dim] + 1) / 2, 0, 1) * self.thrust_multiplier * self.thrust_transition_rate + \
        #                 self.prev_thrust * (1 - self.thrust_transition_rate)
        thrust_scalar = self.throttle2thrust(action[self.dim]) * self.thrust_transition_rate + \
                        self.prev_thrust * (1 - self.thrust_transition_rate)
        self.prev_thrust = thrust_scalar
        trust = kinematics.thrust_vector(thrust_scalar, self.rotation_matrix)
        return trust, rates

    def update(self):
        self.state, self.rotation_matrix = kinematics.update_kinematic_step(self.state, self.rotation_matrix, self.acceleration, self.rates, self.dt)
        self.rotation_matrix = kinematics.rotate_body_by_rates(self.rotation_matrix, self.rates, self.dt)

    def step(self, action, wind_velocity_vector, rotation_matrix, thrust_force):
        """
        :param action: [roll_rate/self.action_scale, pitch_rate/self.action_scale, yaw_rate/self.action_scale, throttle]
        :param wind_velocity_vector: wind vector in world reference frame [3x1]
        :return: rotation_matrix [3x3] (how the world is rotated with respect to the drone), gyro_matrix, accelorometer
                !!! IRL the drone doesn't know its state: Only IMU measurements and orientation !!!
        """
        if action is None:
            throttle, roll, pitch, arm, _, yaw = self.rc.calib_read()
            action = np.array([-roll, pitch, yaw, throttle])
        self.thrust, self.rates = self.action2force(action)
        self.drag_force = kinematics.calculate_drag(self.state, wind_velocity_vector, self.drag_coef)
        self.gravity_force = kinematics.gravity_vector(self.mass, g=self.gravity)
        self.motors_orientation = self.motors_relative_position @ self.rotation_matrix.T
        motor_hit_ground = (self.position + self.motors_orientation)[:, 2] < self.motor_radius
        force_applied_on_motors = np.zeros(shape=(3,))
        if np.any(motor_hit_ground):
            # spring force acting on the motor that hit the ground
            force_applied_on_motors = -(((self.position + self.motors_orientation)[:, 2] - self.motor_radius) * motor_hit_ground).sum() * 5e1 * np.array([0, 0, 1])
        if np.any((self.position + self.motors_orientation)[:, 2] < 0.0):
            self.done = True
        # Just a test for go-to-pixel algorithm
        if rotation_matrix is not None:
            self.rotation_matrix = rotation_matrix
            self.thrust = kinematics.thrust_vector(thrust_force, self.rotation_matrix)
        self.total_forces = self.thrust + self.gravity_force + self.drag_force + force_applied_on_motors
        self.acceleration = self.total_forces / self.mass
        self.update()

        self.camera.update(self.position, self.rotation_matrix)
        angular_velocity_matrix = kinematics.rotation_matrix_from_euler_angles(*self.rates)
        return self.rotation_matrix.T, angular_velocity_matrix, self.rotation_matrix @ self.acceleration

    def get_gravity_force_in_drone_ref_frame(self):
        return self.rotation_matrix @ kinematics.gravity_vector(self.mass, g=9.81)

    def velocity_direction_of_target(self, dir2target):
        return

    def calculate_needed_force_orientation(self, pixel, ref_frame='world', multiplier=1, mode="level", virtual_drag_coef=1.0, virtual_lift_coef=2e1, tof_effective_dist=1.3):
        """
        calculating the force needed to move the drone to a pixel in the image
        :param pixel: [x, y] pixel coordinates
        :param ref_frame: 'world' or 'drone'
        :param multiplier: how much force to apply to the direction of the target
        :param mode: 'level' or 'frontarget' where the drone is leveled or facing the target
        :param virtual_drag_coef: virtual drag coefficient
        :return: force vector [3x1] in world reference frame
        """
        dir2target = self.camera.pixel2direction(pixel)
        # virtual drag is applied more and more when the velocity is in the opposite direction of the target
        if ref_frame == 'world':
            gravity = kinematics.gravity_vector(self.mass, g=9.81)
            velocity_direction_of_target = self.velocity/np.linalg.norm(self.velocity) @ dir2target
            virtual_drag = -(velocity_direction_of_target - 1) / 2 * -self.velocity * np.linalg.norm(self.velocity)
        elif ref_frame == 'drone':
            gravity = self.get_gravity_force_in_drone_ref_frame()
            velocity_direction_of_target = self.rotation_matrix @ self.velocity / np.linalg.norm(self.velocity) @ dir2target
            virtual_drag = -(velocity_direction_of_target - 1) / 2 * -self.rotation_matrix @ self.velocity * np.linalg.norm(self.velocity)
        else:
            raise ValueError('Unknown reference frame')
        # dir2target = multiplier * self.camera.pixel2direction(pixel, ref_frame=ref_frame)
        virtual_drag_force = virtual_drag_coef * virtual_drag
        virtual_lift_force = (self.position[2] < tof_effective_dist) * -(tof_effective_dist - self.position[2]) * virtual_lift_coef * gravity
        force_vector = multiplier * dir2target + virtual_drag_force + virtual_lift_force - gravity
        # level the drone, keep the y-axis at the horizon
        if mode == "level":
            horizon_vector = np.cross(force_vector, gravity)
            front_vector = np.cross(horizon_vector, force_vector)
        elif mode == "frontarget":
            horizon_vector = np.cross(force_vector, dir2target)
            front_vector = np.cross(horizon_vector, force_vector)
        else:
            raise ValueError('Unknown mode')
        rotation_to_apply_force = np.stack([front_vector, horizon_vector, force_vector], axis=1)
        rotation_to_apply_force = rotation_to_apply_force / np.linalg.norm(rotation_to_apply_force, axis=0)
        return rotation_to_apply_force, np.linalg.norm(force_vector)

    def render(self, ax, rpy=True, velocity=False, thrust=False, drag=False, gravity=False, total_force=True, motors=True):
        render3d.plot_3d_points(ax, self.position, color='k')
        if rpy:
            render3d.plot_3d_rotation_matrix(ax, self.rotation_matrix, self.position, scale=0.5)
        if velocity:
            render3d.plot_3d_arrows(ax, self.position, self.velocity, color='m', alpha=0.5)
        if thrust:
            render3d.plot_3d_arrows(ax, self.position, self.thrust, color='c', alpha=0.5)
        if drag:
            render3d.plot_3d_arrows(ax, self.position, self.drag_force, color='y', alpha=0.5)
        if gravity:
            render3d.plot_3d_arrows(ax, self.position, self.gravity_force, color='g', alpha=0.5)
        if total_force:
            render3d.plot_3d_arrows(ax, self.position, self.total_forces, color='k', alpha=0.5)
        if motors:
            [Target(self.position + motors_orientation, radius=0.02, nu=2).render(ax) for motors_orientation in self.motors_orientation]


class Camera:
    def __init__(self, camera_pitch_angle, position_relative_to_frame, resolution, fov=None, focal_length=None):
        self.resolution = resolution
        self.focal_length = focal_length
        self.relative_position = position_relative_to_frame
        # for fixed camera angle (like in real FPV)
        self.relative_rotation_matrix = WORLD2CAM.T @ rotation_matrix_from_euler_angles(np.deg2rad(camera_pitch_angle), 0, 0)
        self.position = None
        self.rotation_matrix = None
        self.image = None
        self.intrinsic_matrix = None
        self.focal_length = focal_length
        self.fov = fov #diagonal fov
        if focal_length is None and fov is not None:
            self.focal_length = self.convert_fov_to_focal_length(fov, self.resolution)
        elif focal_length is not None and fov is None:
            self.fov = self.convert_focal_length_to_fov(self.focal_length, self.resolution)
        if self.focal_length is not None:
            # 3x3 intrinsic matrix using focal length and resolution
            self.intrinsic_matrix = intrinsic_matrix(self.focal_length, self.focal_length, self.resolution[0]/2, self.resolution[1]/2)

    def convert_fov_to_focal_length(self, fov, resolution):
        # fov in degrees
        return resolution[0] / (2 * np.tan(np.deg2rad(fov) / 2))

    def convert_focal_length_to_fov(self, focal_length, resolution):
        return np.rad2deg(2 * np.arctan(resolution[0] / (2 * focal_length)))

    def calculate_fovs_from_resolution(self, resolution, diagonal_fov):
        """
        FOV_Horizontal = 2 * atan(W/2/f) = 2 * atan2(W/2, f)  radians
        FOV_Vertical   = 2 * atan(H/2/f) = 2 * atan2(H/2, f)  radians
        FOV_Diagonal   = 2 * atan2(sqrt(W^2 + H^2)/2, f)    radians
        :param resolution: [width, height]
        :param diagonal_fov: in degrees
        :return: fov_horizontal, fov_vertical
        """
        fov_horizontal = np.rad2deg(2 * np.arctan2(resolution[0] / 2, self.focal_length))
        fov_vertical = np.rad2deg(2 * np.arctan2(resolution[1] / 2, self.focal_length))
        return fov_horizontal, fov_vertical # in degrees

    def set_intrinsic_matrix(self, intrinsic_matrix):
        self.intrinsic_matrix = intrinsic_matrix

    @property
    def extrinsic_matrix(self):
        return np.hstack([self.rotation_matrix, self.position.reshape(-1, 1)])

    def reset(self, drone_position, drone_rotation_matrix):
        self.update(drone_position, drone_rotation_matrix)
        self.image = np.zeros(self.resolution)

    def update(self, drone_position, drone_rotation_matrix):
        self.position = drone_position + drone_rotation_matrix @ self.relative_position
        self.rotation_matrix = drone_rotation_matrix @ self.relative_rotation_matrix

    def pixel2direction(self, pixel, ref_frame='world'):
        """
        calculating a vector norm 1 pointing from camera/drone to the pixel
        :param pixel: [h, w]
        :param ref_frame: 'world', 'drone' or 'camera'
        :return: direction vector in world frame
        """
        pixel = np.array(pixel)
        pixel = np.hstack([pixel, 1])
        if ref_frame == 'world':
            direction = self.rotation_matrix @ np.linalg.inv(self.intrinsic_matrix) @ pixel
        elif ref_frame == 'drone':
            direction = self.relative_rotation_matrix @ np.linalg.inv(self.intrinsic_matrix) @ pixel
        elif ref_frame == 'camera':
            direction = np.linalg.inv(self.intrinsic_matrix) @ pixel
        else:
            raise ValueError('ref_frame must be world, drone or camera')
        return direction / np.linalg.norm(direction)

    def render(self, ax):
        render3d.plot_3d_points(ax, self.position, color='c')
        render3d.plot_3d_rotation_matrix(ax, self.rotation_matrix, self.position, scale=0.5)

    @property
    def projection_matrix(self):
        # add row to extrinsic matrix to make it 4x4
        extrinsic_matrix = np.vstack([self.extrinsic_matrix, np.array([0, 0, 0, 1])])
        return self.intrinsic_matrix @ np.linalg.inv(extrinsic_matrix)[:3, :]

    def project(self, objects_list):
        points = [obj.points.copy() for obj in objects_list]
        points = np.vstack(points)
        points = self.projection_matrix @ np.vstack([points.T, np.ones(points.shape[0])])
        points = points.T
        depth = points[:, 2]
        #keep only points in front of camera
        points = points[depth > 0]
        depth = depth[depth > 0]
        points = points[:, :2] / depth.reshape(-1, 1)
        points = points.astype(int)
        return points, depth

    def render_image(self, objects_list):
        points, depth = self.project(objects_list)
        self.image = np.zeros(self.resolution[::-1])
        for z, point in zip(depth, points):
            condition = 0 <= point[0] < self.resolution[0] and \
                        0 <= point[1] < self.resolution[1] and \
                        (self.image[point[1], point[0]] == 0 or self.image[point[1], point[0]] > z)
            if condition:
                self.image[point[1], point[0]] = 1
        return self.image

    def render_depth_image(self, objects_list, max_depth=10):
        points, depth = self.project(objects_list)
        self.image = np.zeros(self.resolution[::-1])
        for z, point in zip(depth, points):
            condition = 0 <= point[0] < self.resolution[0] and\
                        0 <= point[1] < self.resolution[1] and\
                        (self.image[point[1], point[0]] == 0 or self.image[point[1], point[0]] > z)
            if condition:
                self.image[point[1], point[0]] = z
        np.clip(self.image, 0, max_depth, out=self.image)
        self.image[self.image == 0] = max_depth
        self.image = (255 * (1 - self.image / max_depth)).astype(np.uint8)
        return self.image


class Ground:
    def __init__(self, size, resolution, random=False):
        self.size = size
        self.resolution = resolution
        self.points = self.generate_points(random)

    def generate_points(self, random):
        if random:
            random_points = self.size * (2 * np.random.rand(self.resolution**2, 3) - 1)
            random_points[:, 2] /= self.size
            random_points[:, 2] *= 0.2
            return random_points
        else:
            axis = np.linspace(-self.size/2, self.size/2, self.resolution)
            x, y = np.meshgrid(axis, axis)
            return np.vstack([x.reshape(-1), y.reshape(-1), np.zeros(x.shape).reshape(-1)]).T

    def render(self, ax, **kwargs):
        render3d.plot_3d_points(ax, self.points, color='g', **kwargs)


class Target:
    def __init__(self, position, radius, nu):
        self.position = position
        self.radius = radius
        self.vertices, self.faces = icosphere(nu=nu)
        self.vertices = self.vertices * radius
        self.current_vertices = self.vertices + self.position

    @property
    def points(self):
        return self.current_vertices

    #@TODO: add rotation matrix for a rolling ball.
    def update(self, position):
        self.position = position
        self.current_vertices = self.vertices + self.position

    def render(self, ax, **kwargs):
        """ kwargs: facecolor, edgecolor, linewidth, alpha """
        poly = mpl_toolkits.mplot3d.art3d.Poly3DCollection(self.current_vertices[self.faces], **kwargs)
        ax.add_collection3d(poly)


class Gate:
    def __init__(self, position, rotation_matrix, size, shape="rectangle", resolution=17):
        self.position = position
        self.rotation_matrix = rotation_matrix
        self.size = size
        if shape == "rectangle":
            self.corners = np.array([[0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1]]) * size / 2
        elif "circle" in shape:
            coef_factor = 1 if "half" in shape else 2
            theta = np.linspace(0, coef_factor * np.pi, resolution)
            y = np.cos(theta) * size / coef_factor
            z = np.sin(theta) * size / coef_factor
            x = np.zeros_like(y)
            self.corners = np.vstack((x, y, z)).T
            if "half" in shape:
                self.corners -= np.array([0, 0, size/2])
        else:
            raise NotImplementedError
        self.corners = (self.rotation_matrix @ self.corners.T).T
        self.corners += self.position
        self.corners = np.vstack((self.corners, self.corners[0]))

    @property
    def points(self):
        return self.corners

    @property
    def normal(self):
        return self.rotation_matrix[:, 0]

    def calculate_plane_equation(self):
        normal = self.normal
        d = -np.dot(normal, self.position)
        return np.append(normal, d)

    def calculate_distance(self, point):
        normal = self.normal
        d = -np.dot(normal, self.position)
        return np.dot(normal, point) + d

    def render(self, ax, mid_point=True, gate_color="blue", arrow_color="orange", text="", fill_color="blue", alpha=0.5):
        if mid_point:
            render3d.plot_3d_arrows(ax, self.position, self.rotation_matrix[:, 0], color=arrow_color)
        ax.text(self.position[0], self.position[1] - self.size/2, self.position[2] + self.size, text, color=arrow_color)
        triang = mtri.Triangulation(self.corners[:, 1], self.corners[:, 2])
        ax.plot_trisurf(self.corners[:, 0], self.corners[:, 1], self.corners[:, 2], triangles=triang.triangles, color=fill_color, alpha=alpha)
        render3d.plot_3d_line(ax, self.corners, color=gate_color)


def generate_track(count, radius, gate_size, gate_resolution):
    theta = np.linspace(0, 2 * np.pi, count + 1)[:-1]
    gates_positions = np.vstack((np.cos(theta) * gate_size, np.sin(theta) * radius, np.zeros_like(theta))).T
    gates = []
    shapes = ["rectangle", "circle", "half_circle"]
    for i, p in enumerate(gates_positions):
        rotmat = rotation_matrix_from_euler_angles(0, 0, theta[i] + np.pi / 2)
        if shapes[i % 3] == "circle":
            gates.append(Gate(p + np.array([0, 0, gate_size / 2]), rotmat, gate_size / 2, shape=shapes[i % 3], resolution=gate_resolution))
        else:
            gates.append(Gate(p, rotmat, gate_resolution, shape=shapes[i % 3], resolution=gate_resolution))
    return gates


def generate_targets(count, center, std, size, variation, nu):
    return [Target(np.array(center) + std * np.random.randn(3), np.abs(size + variation * np.random.randn()), nu=nu) for _ in range(count)]



if __name__ == '__main__':
    """#Gates test
    ax, fig = render3d.init_3d_axis()
    raduis = 4
    n_gates = 20
    theta = np.linspace(0, 2 * np.pi, n_gates + 1)[:-1]
    gates_positions = np.vstack((np.cos(theta) * raduis, np.sin(theta) * raduis, np.zeros_like(theta))).T
    gates = []
    shapes = ["rectangle", "circle", "half_circle"]
    for i, p in enumerate(gates_positions):
        gates.append(Gate(p, np.eye(3), 1, shape=shapes[i % 3]))
    for gate in gates:
        gate.show(ax, text="Gate")
    render3d.show_plot(ax, fig, edge=raduis+1)
    plt.render()
    """
    params = yaml_helper.yaml_reader(r"C:\Users\omrijsharon\PycharmProjects\FpyV\config\params.yaml")
    dim = params["simulator"]["render_dim"]
    drone = Drone(params)
    targets = generate_targets(**params["simulator"]["targets"])
    gates = generate_track(**params["simulator"]["track"])
    ground = Ground(size=30, resolution=100)
    drone.reset(position=np.array(params["drone"]["initial_position"]), velocity=np.array(params["drone"]["initial_velocity"]), ypr=np.array(params["drone"]["initial_orientation"]))
    timesteps = 10000
    rates_array = np.zeros((timesteps, 3))
    thrust_array = np.zeros((timesteps, 1))
    position_array = np.zeros((timesteps, 3))
    wind_velocity_vector = np.array([0, 0, 0])
    rates_array[0, :] = drone.prev_rates
    thrust_array[0, :] = drone.prev_thrust
    position_array[0, :] = drone.position

    # action = np.random.uniform(-1, 1, 4)
    # action[-1] = (action[-1] + 1) / 2
    # action[:-1] *= drone.max_rates / drone.action_scale
    ax, fig = render3d.init_3d_axis()
    sign = 1
    for i in range(0, timesteps):
        ax.clear()
        if i % 1 == 0:
            # throttle, roll, pitch, arm, _, yaw = rc.calib_read()
            # action = np.array([-roll, pitch, yaw, throttle])
            # action = np.random.uniform(-1, 1, 4)
            # action[-1] = (action[-1] + 1) / 2
            # action[:-1] *= drone.max_rates/drone.action_scale * 0.4
            action = np.array([0.0, 0.3, 0.0, -1.0])
        # drone.step(action=action, wind_velocity_vector=wind_velocity_vector)
        if drone.done:
            print("Crashed")
            break
        # print((targets[0].points - drone.camera.position).mean(axis=0))
        if dim == 3:
            # render 3d world
            # drone.render(ax, rpy=True, velocity=True, thrust=True, drag=True, gravity=True, total_force=False)
            drone.render(ax, rpy=True, velocity=False, thrust=False, drag=False, gravity=False, total_force=False, motors=True)
            # drone.camera.render(ax)
            position_array[i, :] = drone.position
            render3d.plot_3d_line(ax, position_array[:i, :], color="blue", alpha=0.4)
            [target.render(ax, alpha=0.2) for target in targets]
            # ground.render(ax, alpha=0.1)
            # direction2target = drone.camera.pixel2direction(np.array(np.where(img > 0)).mean(1)[::-1], ref_frame="world")
            # direction2target = drone.camera.pixel2direction([320, 240], ref_frame="drone")
            target_img = drone.camera.render_depth_image([targets[0]], max_depth=15)
            target_pixels = np.array(np.where(target_img > 0))
            if target_pixels.shape[1] == 0:
                drone.step(action=action, wind_velocity_vector=wind_velocity_vector, rotation_matrix=None, thrust_force=None)
            else:
                target_pixels = target_pixels.mean(1)[::-1]
                rot_mat, force_size = drone.calculate_needed_force_orientation(target_pixels, **params["calculate_needed_force_orientation"])
                drone.step(action=action, wind_velocity_vector=wind_velocity_vector, rotation_matrix=rot_mat, thrust_force=force_size)
                render3d.plot_3d_rotation_matrix(ax, rot_mat, drone.position, scale=2.5)
            # render3d.plot_3d_arrows(ax, drone.position, direction2target, color='m', alpha=1)
            render3d.show_plot(ax, fig, middle=drone.position, edge=5)
            # print(f"position{np.round(drone.position, 2)}")
        else:
            # render what the drone camera sees
            # img = 255 * drone.camera.render_image([*targets, *gates, ground]).astype(np.uint8)
            img = drone.camera.render_depth_image([*targets, *gates, ground], max_depth=25)
            target_img = drone.camera.render_depth_image([targets[0]], max_depth=15)
            target_pixels = np.array(np.where(target_img > 0))
            if target_pixels.shape[1] == 0:
                drone.step(action=action, wind_velocity_vector=wind_velocity_vector, rotation_matrix=None, thrust_force=None)
            else:
                target_pixels = target_pixels.mean(1)[::-1]
                rot_mat, force_size = drone.calculate_needed_force_orientation(target_pixels, **params["calculate_needed_force_orientation"])
                drone.step(action=action, wind_velocity_vector=wind_velocity_vector, rotation_matrix=rot_mat, thrust_force=force_size)
                # add circle where the target is on the image:
                cv2.circle(img, tuple(target_pixels.astype(int)), 10, (255, 255, 255), 2)
            img = cv2.putText(img.astype(np.uint8), f"position{np.round(drone.camera.position,2)}, velocity: {np.round(3.6 * np.linalg.norm(drone.velocity),2)} kph", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("img", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # rates_array[i, :] = drone.prev_rates
        # thrust_array[i, :] = drone.prev_thrust
        # plt.subplot(2, 1, 1)
        # plt.plot(rates_array[:i, :])
        # plt.subplot(2, 1, 2)
        # plt.plot(thrust_array[:i, :])
        # plt.pause(0.01)

    # plt.show()
