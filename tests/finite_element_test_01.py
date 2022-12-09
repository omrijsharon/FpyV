import numpy as np
import matplotlib.pyplot as plt

def compute_stiffness_matrix(vertices, faces, E, nu):
    # Compute the number of nodes and elements in the 3D object
    num_nodes = vertices.shape[0]
    num_elements = faces.shape[0]

    # Create the stiffness matrix as a zero matrix
    K = np.zeros((num_nodes, num_nodes))

    # Loop over all elements in the 3D object
    for i in range(num_elements):
        # Get the indices of the nodes of the current element
        node_indices = faces[i]

        # Get the coordinates of the nodes of the current element
        x1, y1, z1 = vertices[node_indices[0]]
        x2, y2, z2 = vertices[node_indices[1]]
        x3, y3, z3 = vertices[node_indices[2]]

        # Compute the area of the current element using Heron's formula
        a = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        b = np.sqrt((x1-x3)**2 + (y1-y3)**2 + (z1-z3)**2)
        c = np.sqrt((x2-x3)**2 + (y2-y3)**2 + (z2-z3)**2)
        s = (a + b + c) / 2
        A = np.sqrt(s * (s-a) * (s-b) * (s-c))

        # Compute the stiffness matrix for the current element
        k = (A/12) * np.array([[1, 1, 1, -1, -1, -1],
                               [1, 1, 1, -1, -1, -1],
                               [1, 1, 1, -1, -1, -1],
                               [-1, -1, -1, 1, 1, 1],
                               [-1, -1, -1, 1, 1, 1],
                               [-1, -1, -1, 1, 1, 1]])

        # Add the stiffness matrix of the current element to the global stiffness matrix
        K[node_indices, :][:, node_indices] += k

    # Compute the material matrix of the 3D object
    D = (E / (1 - nu**2)) * np.array([[1, nu, nu, 0, 0, 0],
                                      [nu, 1, nu, 0, 0, 0],
                                      [nu, nu, 1, 0, 0, 0],
                                      [0, 0, 0, (1-nu)/2, 0, 0],
                                      [0, 0, 0, 0, (1-nu)/2, 0],
                                      [0, 0, 0, 0, 0, (1-nu)/2]])

    # Multiply the global stiffness matrix by the material matrix
    K = np.dot(D, K)
    return K

def compute_nodal_forces(vertices, faces, pressure, K):
    # Compute the number of nodes in the 3D object
    num_nodes = vertices.shape[0]

    # Create the nodal forces vector as a zero vector
    F = np.zeros((num_nodes,))

    # Loop over all faces of the 3D object
    for i in range(faces.shape[0]):
        # Get the indices of the nodes of the current face
        node_indices = faces[i]

        # Get the coordinates of the nodes of the current face
        x1, y1, z1 = vertices[node_indices[0]]
        x2, y2, z2 = vertices[node_indices[1]]
        x3, y3, z3 = vertices[node_indices[2]]

        # Compute the area of the current face using Heron's formula
        a = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        b = np.sqrt((x1-x3)**2 + (y1-y3)**2 + (z1-z3)**2)
        c = np.sqrt((x2-x3)**2 + (y2-y3)**2 + (z2-z3)**2)
        s = (a + b + c) / 2
        A = np.sqrt(s * (s-a) * (s-b) * (s-c))

        # Compute the normal vector of the current face
        normal = np.cross(np.array([x1-x2, y1-y2, z1-z2]),
                         np.array([x1-x3, y1-y3, z1-z3]))

        # Compute the nodal forces for the current face
        f = (pressure * A / 3) * normal

        # Add the nodal forces of the current face to the global nodal forces vector
        F[node_indices] += f

    return F

def compute_displacements(K, F):
# Compute the displacements of the 3D object
    U = np.linalg.solve(K, F)
    return U


def compute_vibration_modes(K, U):
    # Compute the eigenvalues and eigenvectors of the stiffness matrix
    eigenvalues, eigenvectors = np.linalg.eig(K)

    # Sort the eigenvalues and eigenvectors in ascending order of eigenvalue
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Compute the frequencies and mode shapes of the 3D object
    frequencies = np.sqrt(eigenvalues) / (2 * np.pi)
    mode_shapes = eigenvectors.T

    return frequencies, mode_shapes

if __name__ == '__main__':

    # Define the vertices of the 3D object
    vertices = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0],
                         [0,0,1], [1,0,1], [1,1,1], [0,1,1]])

    # Define the faces of the 3D object as a list of vertices
    # Each face is defined by three vertices, forming a triangle
    faces = np.array([[0,1,2], [2,3,0],
                      [1,5,6], [6,2,1],
                      [5,4,7], [7,6,5],
                      [4,0,3], [3,7,4],
                      [3,2,6], [6,7,3],
                      [4,5,1], [1,0,4]])
    #plot vertices and faces
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], c='r', marker='o')
    for i in range(faces.shape[0]):
        # Get the indices of the nodes of the current face
        node_indices = faces[i]
        x1, y1, z1 = vertices[node_indices[0]]
        x2, y2, z2 = vertices[node_indices[1]]
        x3, y3, z3 = vertices[node_indices[2]]
        ax.plot([x1,x2,x3,x1], [y1,y2,y3,y1], [z1,z2,z3,z1], c='k')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    # Define the material properties of the 3D object
    # In this case, we assume the object is made of steel
    E = 200e9  # Young's modulus of steel
    nu = 0.3   # Poisson's ratio of steel

    # Define the loading conditions on the 3D object
    # In this case, we apply a uniform pressure on all faces
    pressure = 1e6  # Pressure in Pa

    # Compute the stiffness matrix of the 3D object using the finite element method
    K = compute_stiffness_matrix(vertices, faces, E, nu)

    # Compute the nodal forces on the 3D object using the loading conditions and stiffness matrix
    F = compute_nodal_forces(vertices, faces, pressure, K)

    # Solve for the nodal displacements of the 3D object using the stiffness matrix and nodal forces
    U = np.linalg.solve(K, F)

    # Compute the frequencies and mode shapes of the 3D object using the stiffness matrix and nodal displacements
    frequencies, mode_shapes = compute_vibration_modes(K, U)

    # Print the frequencies and mode shapes of the 3D object
    print("Frequencies:", frequencies)
    print("Mode shapes:", mode_shapes)