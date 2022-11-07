import serial
import numpy as np
import matplotlib.pyplot as plt

from utils.helper_functions import quaternion_to_rotation_matrix
from utils.render3d import init_3d_axis, plot_3d_rotation_matrix, show_plot, plot_3d_line, plot_3d_points


def check_which_port():
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    ports_list = []
    for p in ports:
        ports_list.append(p)
    return ports_list


def get_leonardo_port():
    ports = check_which_port()
    for p in ports:
        if 'Leonardo' in p.description:
            return p.device
    return None


def count_elements_in_str_line(line):
    elements_list_in_line = [l for l in line[-1].split(":")[-1].split()]
    if len(elements_list_in_line) > 0:
        try:
            float(elements_list_in_line[-1])
            return len(elements_list_in_line)
        except ValueError:
            return len(elements_list_in_line) - 1
    else:
        return 0

def read_serial_stream():
    # Open serial port
    ser = serial.Serial(get_leonardo_port(), 115200, timeout=0.001)
    ax, fig = init_3d_axis()
    points = np.zeros(shape=(3,))
    points_list = np.zeros(shape=(3,))
    # Read data
    try:
        buffer = []
        while True:
            ax.clear()
            waiting = ser.in_waiting  # find num of bytes currently waiting in hardware
            buffer += [chr(c) for c in ser.read(waiting)]  # read them, convert to ascii
            # ser.inWaiting()
            # line = ser.readline(64).decode('ascii')
            line = "".join(buffer)
            #clean text
            line = line.split('\r\n')
            quat_line = [l for l in line if "quaternion" in l]
            rot_mat_line = [l for l in line if "Rotation matrix" in l]
            position_line = [l for l in line if "Position" in l]
            acceleration_line = [l for l in line if "Acceleration" in l]
            if len(position_line) > 1:
                print("count_elements_in_str_line(position_line)", count_elements_in_str_line(position_line))
                if count_elements_in_str_line(position_line) == 3:
                    position_line = position_line[-1]
                else:
                    position_line = position_line[-2]
                print(position_line, len(position_line))
                position_line = position_line.replace('Position: ', '')
                points = np.array([float(x) for x in position_line.split()]) / 16384
                points_list = np.vstack((points_list, points))
                plot_3d_line(ax, points_list, color='b')
            if len(acceleration_line) > 1:
                print("count_elements_in_str_line(acceleration_line)", count_elements_in_str_line(acceleration_line))
                if count_elements_in_str_line(acceleration_line) == 3:
                    acceleration_line = acceleration_line[-1]
                else:
                    acceleration_line = acceleration_line[-2]
                print(acceleration_line, len(acceleration_line))
                acceleration_line = acceleration_line.replace('Acceleration: ', '')
                acceleration = np.array([float(x) for x in acceleration_line.split()]) / 16384
                # points_list = np.vstack((points_list, acceleration))[-50:]
                # plot_3d_points(ax, points_list, color='r')
            if len(quat_line) > 1:
                if count_elements_in_str_line(quat_line) == 4:
                    quat_line = quat_line[-1]
                else:
                    quat_line = quat_line[-2]
                # Parse data
                quat_line = quat_line.replace('quaternion: ', '')
                q = np.array([float(x.split(": ")[-1]) for x in quat_line.split(',')])/16384
                if len(q) == 4:
                    R = quaternion_to_rotation_matrix(q)
                    plot_3d_rotation_matrix(ax, R, points, scale=0.5)
                    show_plot(ax, fig)
            elif len(rot_mat_line) > 1:
                if count_elements_in_str_line(rot_mat_line) == 9:
                    rot_mat_line = rot_mat_line[-1]
                else:
                    rot_mat_line = rot_mat_line[-2]
                print(rot_mat_line)
                # Parse data
                rot_mat_line = rot_mat_line.replace('Rotation matrix: ', '')
                R = np.array([float(x) / 16384.0 for x in rot_mat_line.split()]).reshape(3, 3)
                plot_3d_rotation_matrix(ax, R, points, scale=0.2)
            show_plot(ax, fig, middle=points, edge=0.5)
    except KeyboardInterrupt:
        # Close serial port
        ser.close()
        print("Serial port closed")

if __name__ == '__main__':
    read_serial_stream()
