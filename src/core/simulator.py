import cv2
import numpy as np

from utils import render3d, yaml_helper
from utils.components import Drone, Ground
from utils.generators import generate_targets, generate_track, generate_cylinders
from utils.helper_functions import distance

params = yaml_helper.yaml_reader(r"C:\Users\omri_\PycharmProjects\FpyV\config\params.yaml")
half_camera_resolution = np.array(params["camera"]["resolution"]) / 2
ix,iy, prev_ix, prev_iy = *half_camera_resolution, *half_camera_resolution
flag = False


# mouse callback function
def get_target(event,x,y,flags,param):
    global ix,iy, prev_ix, prev_iy, flag
    rate = 0.1
    # if event == cv2.EVENT_MOUSEMOVE:
    #     ix,iy = x, y
    #     print(ix, iy)
    if event == cv2.EVENT_LBUTTONDOWN:
        flag = True
    elif event == cv2.EVENT_LBUTTONUP:
        flag = False
    if flag:
        ix,iy = (rate * x) + (1-rate) * prev_ix, (rate * y) + (1-rate) * prev_iy
        prev_ix, prev_iy = ix, iy
    else:
        ix, iy = ((1 - rate) * half_camera_resolution[0]) + rate * prev_ix, ((1 - rate) * half_camera_resolution[1]) + rate * prev_iy
        prev_ix, prev_iy = ix, iy


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
    cv2.namedWindow('img')
    # cv2.setMouseCallback('img', get_target)
    dim = params["simulator"]["render_dim"]
    drone = Drone(params)
    targets = generate_targets(**params["simulator"]["targets"])
    target_chase_idx = 0
    obstacles = generate_cylinders(**params["simulator"]["obstacles"])
    gates = generate_track(**params["simulator"]["track"])
    ground = Ground(**params["simulator"]["ground"])
    drone.reset(position=np.array(params["drone"]["initial_position"]), velocity=np.array(params["drone"]["initial_velocity"]), ypr=np.array(params["drone"]["initial_orientation"]))
    time_steps = 10000
    rates_array = np.zeros((time_steps, 3))
    thrust_array = np.zeros((time_steps, 1))
    wind_velocity_vector = np.array([0, 0, 0])
    rates_array[0, :] = drone.prev_rates
    thrust_array[0, :] = drone.prev_thrust
    prev_frame = np.zeros(shape=params["camera"]["resolution"][::-1])
    kernel_size = 11
    axis = np.linspace(-2, 2, kernel_size)
    X, Y = np.meshgrid(axis, axis)
    kernel = np.exp(-(X ** 2 + Y ** 2) / 0.5)
    threshold = 0.2
    kernel[kernel < threshold] = 0
    kernel[kernel >= threshold] = 1
    kernel = kernel.astype(np.uint8)
    # kernel = np.ones((7, 7), np.uint8)
    # action = np.random.uniform(-1, 1, 4)
    # action[-1] = (action[-1] + 1) / 2
    # action[:-1] *= drone.max_rates / drone.action_scale
    ax, fig = render3d.init_3d_axis()
    sign = 1
    for i in range(0, time_steps):
        # object_list = [*targets, *gates, *obstacles, drone.trail, ground] # increases rendering rate
        object_list = [*targets, *gates, *obstacles, ground] # no trail
        ax.clear()
        [target.update() for target in targets]
        if i % 1 == 0:
            action = np.array([0.2, -0.2, -0.2, -0.5])
        # drone.step(action=action, wind_velocity_vector=wind_velocity_vector)
        if drone.done:
            print("Crashed")
            break
        if dim == 3:
            # render 3d world
            # drone.render(ax, rpy=True, velocity=True, thrust=True, drag=True, gravity=True, total_force=False)
            drone.render(ax, rpy=True, velocity=False, thrust=False, drag=False, gravity=False, total_force=False, motors=True)
            # drone.camera.render(ax)
            [target.render(ax, alpha=0.2) for target in targets]
            [obstacle.render(ax, alpha=0.2) for obstacle in obstacles]
            # ground.render(ax, alpha=0.1)
            target_img = drone.camera.render_depth_image([targets[target_chase_idx]], max_depth=15)
            target_pixels = np.array(np.where(target_img > 0))
            if target_pixels.shape[1] == 0:
                drone.step(action=action, wind_velocity_vector=wind_velocity_vector, object_list=object_list)
            else:
                target_pixels = target_pixels.mean(1)[::-1]
                target_pixels[0] += 130
                rot_mat, force_size = drone.calculate_needed_force_orientation(target_pixels, targets[target_chase_idx], **params["calculate_needed_force_orientation"])
                drone.step(action=action, wind_velocity_vector=wind_velocity_vector, object_list=object_list, rotation_matrix=rot_mat, thrust_force=force_size)
                render3d.plot_3d_rotation_matrix(ax, rot_mat, drone.position, scale=2.5)
            # render3d.plot_3d_arrows(ax, drone.position, direction2target, color='m', alpha=1)
            if i % 3 == 0:
                render3d.show_plot(ax, fig, middle=drone.position, edge=5)
        elif dim == 2:
            # render what the drone camera sees
            # img = 255 * drone.camera.render_image([*targets, *gates, ground]).astype(np.uint8)
            frame = drone.camera.render_depth_image(object_list, max_depth=25)
            img = params["simulator"]["frame_transition_rate"] * frame + (1 - params["simulator"]["frame_transition_rate"]) * prev_frame
            prev_frame = img.copy()
            img = np.clip(0.8/params["simulator"]["frame_transition_rate"] * img, 0, 255).astype(np.uint8)
            img = cv2.dilate(img, kernel, iterations=1)
            target_img = drone.camera.render_depth_image([targets[target_chase_idx]], max_depth=25)
            target_pixels = np.array(np.where(target_img > 0))
            # target_pixels = np.array([ix, iy])
            if target_pixels.shape[1] == 0:
            # if target_pixels[0] == -1:
                print("lost target")
                drone.step(action=action, wind_velocity_vector=wind_velocity_vector, object_list=object_list, rotation_matrix=None, thrust_force=None)
            else:
                target_pixels = target_pixels.mean(1)[::-1]
                target_pixels[0] += 70
                # target_pixels = np.array([ix, iy])
                # rot_mat, force_size = drone.calculate_needed_force_orientation(target_pixels, targets[target_chase_idx])
                rot_mat, force_size = drone.point_and_shoot(target_pixels, throttle=0.0)
                prune_object_list = drone.camera.pruned_objects_list(object_list)
                # bbox2d_list = drone.camera.bbox2d(drone.camera.pruned_objects_list(prune_object_list))
                # for bbox2d in bbox2d_list:
                #     cv2.rectangle(img, tuple(bbox2d[0]), tuple(bbox2d[1]), 255, 1)

                # add circle where the target is on the image:
                cv2.circle(img, tuple(target_pixels.astype(int)), 10, (255, 255, 255), 1)
                drone.step(action=action, wind_velocity_vector=wind_velocity_vector, object_list=object_list, rotation_matrix=rot_mat, thrust_force=force_size*0)

                img = cv2.putText(img.astype(np.uint8), f"dist2target: {np.round(distance(drone, targets[0]), 2)} m, "
                                                        f"velocity: {np.round(3.6 * np.linalg.norm(drone.velocity) ,2)} kph, "
                                                        f"throttle: {np.round(100 * (drone.thrust2throttle(force_size) + 1) / 2, 2)} %, " 
                                                        f"position{np.round(drone.camera.position ,2)}",
                                  (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            if i % 2 == 0:
                drone.force_multiplier_pid.plot()
                cv2.imshow("img", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            raise ValueError("dim can only be 2 or 3")
        # rates_array[i, :] = drone.prev_rates
        # thrust_array[i, :] = drone.prev_thrust
        # plt.subplot(2, 1, 1)
        # plt.plot(rates_array[:i, :])
        # plt.subplot(2, 1, 2)
        # plt.plot(thrust_array[:i, :])
        # plt.pause(0.01)

    # plt.show()