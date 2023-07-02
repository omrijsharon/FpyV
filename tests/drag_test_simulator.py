import numpy as np
import matplotlib.pyplot as plt

from utils.helper_functions import euler_angles_to_rotation_matrix

GRAVITY = 9.81
AIR_DENSITY = 1.225
DT = 0.01
DIM = 2

class Particle:
    def __init__(self, mass, cross_section_areas, drag_coefficients):
        self.position = None
        self.velocity = None
        self.mass = mass
        self.gravity = np.zeros(DIM)
        self.gravity[-1] = -GRAVITY * self.mass
        self.drag = np.zeros(DIM)
        self.cross_section_areas = np.array(cross_section_areas)
        self.drag_coefficients = np.array(drag_coefficients)
        self.rotation_matrix = None
        # init plot
        self.fig, self.ax = plt.subplots()
        self.edge = 20
        self.ax.set_aspect('equal')

    def reset(self, position=None, velocity=None, angle=None):
        if position is None:
            position = np.zeros(DIM)
        if velocity is None:
            velocity = np.zeros(DIM)
        if angle is None:
            rotation_matrix = np.eye(DIM)
        else:
            if DIM==2:
                rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            elif DIM==3:
                rotation_matrix = euler_angles_to_rotation_matrix(*angle)
            else:
                raise ValueError('DIM must be 2 or 3')
        self.position = position
        self.velocity = velocity
        self.rotation_matrix = rotation_matrix

    def calculate_drag_force(self):
        drag_force_self_ref_frame = -0.5 * self.drag_coefficients * AIR_DENSITY * self.cross_section_areas * (self.rotation_matrix.T @ self.velocity) * np.linalg.norm(self.velocity)
        return self.rotation_matrix @ drag_force_self_ref_frame # drag force in world reference frame

    def update(self):
        self.drag = self.calculate_drag_force()
        total_force = self.drag + self.gravity
        acceleration = total_force / self.mass
        self.velocity += acceleration * DT
        self.position += self.velocity * DT

    def plot(self):
        #clear plot
        self.ax.clear()
        self.ax.plot(self.position[0], self.position[1], 'o')
        # add arrows for the forces
        self.ax.arrow(self.position[0], self.position[1], self.drag[0], self.drag[1], head_width=0.5, head_length=0.5, fc='r', ec='r')
        self.ax.arrow(self.position[0], self.position[1], self.gravity[0], self.gravity[1], head_width=0.5, head_length=0.5, fc='g', ec='g')
        # add arrows for the velocity
        self.ax.arrow(self.position[0], self.position[1], self.velocity[0], self.velocity[1], head_width=0.5, head_length=0.5, fc='b', ec='b')
        # add legent to the arrows
        self.ax.text(self.position[0]+self.drag[0], self.position[1]+self.drag[1], 'drag')
        self.ax.text(self.position[0]+self.gravity[0], self.position[1]+self.gravity[1], 'gravity')
        self.ax.text(self.position[0]+self.velocity[0], self.position[1]+self.velocity[1], 'velocity')
        self.ax.set_xlim(-self.edge, self.edge)
        self.ax.set_ylim(-self.edge, self.edge)
        plt.pause(0.0001)


if __name__ == '__main__':
    # test
    particle = Particle(1, [0.01, 0.1], [1, 1])
    particle.reset(angle=np.pi/16)
    for i in range(1000):
        particle.update()
        particle.plot()
    plt.show()
