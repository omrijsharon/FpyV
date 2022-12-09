import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Load the STL file into a mesh object
your_mesh = mesh.Mesh.from_file(r'C:\Users\omri_\Downloads\Frame_1_6.stl')

# Create a new figure
fig = plt.figure()

# Add an 3D axes to the figure
ax = plt.axes(projection='3d')

# Add the mesh to the axes
ax.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

# Set the limits of the axes
# ax.set_xlim(0, your_mesh.points[:,0].max())
# ax.set_ylim(0, your_mesh.points[:,1].max())
# ax.set_zlim(0, your_mesh.points[:,2].max())

# Show the figure
plt.show()