import numpy as np
import icosphere
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def mesh_plot(vertices, faces):
    gm = go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                   i=faces[:, 0], j=faces[:, 1], k=faces[:, 2])
    return gm


def wireframe_plot(vertices, faces):
    Xe = np.concatenate((vertices[faces, 0], np.full((faces.shape[0], 1), None)), axis=1).ravel()
    Ye = np.concatenate((vertices[faces, 1], np.full((faces.shape[0], 1), None)), axis=1).ravel()
    Ze = np.concatenate((vertices[faces, 2], np.full((faces.shape[0], 1), None)), axis=1).ravel()
    gm = go.Scatter3d(x=Xe, y=Ye, z=Ze, mode='lines', name='', line=dict(color='rgb(40,40,40)', width=1))
    return gm


nu = 15
vertices, faces = icosphere.icosphere(nu)

fig = go.Figure()

fig.add_trace(mesh_plot(vertices, faces))
fig.add_trace(wireframe_plot(vertices, faces))

fig.update_layout(title_text='Icosphere', height=600, width=600)
fig.show()

fig = make_subplots(rows=2, cols=3,
                    specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}],
                           [{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]])

for i in range(2):
    for j in range(3):
        nu = 1 + j + 3 * i
        vertices, faces = icosphere.icosphere(nu)
        fig.add_trace(mesh_plot(vertices, faces), row=i + 1, col=j + 1)
        fig.add_trace(wireframe_plot(vertices, faces), row=i + 1, col=j + 1)

fig.update_layout(title_text='Different values of nu', height=600, width=800, showlegend=False)
fig.show()