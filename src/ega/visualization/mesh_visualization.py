from simple_3dviz.renderables import Mesh
from simple_3dviz.behaviours.io import SaveGif
from simple_3dviz.behaviours.movements import RotateModel
from simple_3dviz import render
import numpy as np

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

