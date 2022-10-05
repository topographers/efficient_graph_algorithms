from simple_3dviz.renderables import Mesh
from simple_3dviz.behaviours.io import SaveGif
from simple_3dviz.behaviours.movements import RotateModel
from simple_3dviz import render

import numpy as np

import plotly.graph_objects as go
from plotly.offline import plot
import plotly.figure_factory as ff

def get_rgb(input_arr: float):
    '''
    input: 1d array, length is equal to the number of vertices. each entry represents color for a vertex
    output: 2d array, shape (n_vertices, 3), representing RGB colors. normalized to [0,1]
    '''
    maximum = max(input_arr)
    minimum = min(input_arr)
    ratio = (input_arr-minimum) / (maximum - minimum) 
    r_arr = ratio
    g_arr = ratio
    b_arr = np.ones(len(input_arr))
    rgb_colors = np.zeros((len(input_arr), 3))
    rgb_colors[:,0], rgb_colors[:,1], rgb_colors[:,2] = r_arr, g_arr, b_arr
    return rgb_colors

def simple3d_save_gif(distribution: float, mesh: dict, output_file_path: str):
    '''
    input: distribution is a 1d arry, length is equal to the number of vertices
    
    save the plot as a gif file
    '''
    rgb_colors = get_rgb(distribution)
    input_colors = np.ones(((len(mesh.vertices)), 4))
    input_colors[:,0:3] = rgb_colors
    scaled_vertices = mesh.vertices / (np.max(mesh.vertices) - np.min(mesh.vertices))
    m = Mesh.from_faces(scaled_vertices, mesh.faces, colors=input_colors)
    render(
        m,
        [RotateModel(), SaveGif(output_file_path)],
        180,
        camera_position=(1, 1, 1),
        background=(1,)*4
    )


def plot_mesh(world_pos: np.ndarray, vertices_interpolate_pos: np.ndarray, true_fields: np.ndarray, interpolated_fields: np.ndarray, 
              snapshot_index: int, scale: int = 15):
    """
    this function uses plotly for meshgraphnet data visualization.
    predicted and true fields(velocities) of vertices to be interpolated are shown as arrows on the 3d plot 
    """
    n_vertices_interpolate = len(vertices_interpolate_pos)
    # calculate arrows representing velocity directions
    true_velocity_directions = vertices_interpolate_pos + scale * true_fields
    interpolated_velocity_directions = vertices_interpolate_pos + scale * interpolated_fields
    true_arrow_matrix = np.zeros((3, 3*n_vertices_interpolate))
    interpolated_arrow_matrix = np.zeros((3, 3*n_vertices_interpolate))
    true_arrow_matrix[:,2::3] = None
    interpolated_arrow_matrix[:,2::3] = None
    true_arrow_matrix[:,::3] = vertices_interpolate_pos.T
    interpolated_arrow_matrix[:,::3] = vertices_interpolate_pos.T
    true_arrow_matrix[:,1::3] = true_velocity_directions.T
    interpolated_arrow_matrix[:,1::3] = interpolated_velocity_directions.T
    
    # plot
    x, y, z = world_pos.T
    mesh = go.Mesh3d(x=x, y=y, z=z, alphahull=5, opacity=0.4, color='grey')

    px, py, pz = vertices_interpolate_pos.T
    points = go.Scatter3d(x = px, y = py, z = pz, mode = 'markers', name = 'interpolated_points',
                           marker = dict( size = 2, color = "black"))

    vx_true, vy_true, vz_true = true_arrow_matrix
    true_velocities = go.Scatter3d(x=vx_true, y=vy_true, z=vz_true, mode='lines', name='true_velocities', 
                                   line = dict(color = 'red', width=4))

    vx_interpolated, vy_interpolated, vz_interpolated = interpolated_arrow_matrix
    interpolated_velocities = go.Scatter3d(x=vx_interpolated, y=vy_interpolated, z=vz_interpolated,
        mode='lines', name='interpolated_velocities', line = dict(color = 'blue', width=4))
    
    fig = go.Figure(data=[points, mesh, true_velocities, interpolated_velocities, ],)

    fig.show()
    #fig.write_image(
    #    os.path.join(default_meshgraphnet_dataset_path,'flag_simple','flag_{}.png'.format(snapshot_index)))
    plot(fig,filename="vector.html",auto_open=True,image='png',image_height=1000,image_width=1100)

