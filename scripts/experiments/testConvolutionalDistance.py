import argparse
import numpy as np
from getMeshData import get_mesh_data
from blurOnMesh import blur_on_mesh
import trimesh
from simple_3dviz.renderables import Mesh
from simple_3dviz.behaviours.io import SaveGif
from simple_3dviz.behaviours.movements import RotateModel
from simple_3dviz import render
from convolutionalDistance import convolutional_distance
import os

def get_args_parser():
    parser = argparse.ArgumentParser('convbarycenter', add_help=False)
    parser.add_argument('--niter', dest='niter', type=int, default=100,
                        help="""number of iterations""")
    parser.add_argument('--tol', dest='tol', type=float, default=1e-6,
                        help="""stopping tolerance""")
    parser.add_argument('--verb', dest='verb', type=int, default=1,
                        help="""if set to 1, print information at each iteration""")
    parser.add_argument('--object_folder', dest='object_folder', type=str, default='./Solomon_2015/meshes',
                        help="""path for sample data.""")
    parser.add_argument('--output_folder', dest='output_folder', type=str, default='./gif_files',
                        help="""path for saving plots.""")
    return parser

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
    rgb_colors = np.zeros((len(input_arr),3))
    rgb_colors[:,0], rgb_colors[:,1], rgb_colors[:,2] = r_arr, g_arr, b_arr
    return rgb_colors

def simple3d_save_gif(distribution: float, mesh: dict, fig_name: str, output_folder=''):
    '''

    input: distribution is a 1d arry, length is equal to the number of vertices
    
    save the plot as a gif file

    '''
    rgb_colors = get_rgb(distribution)
    input_colors = np.ones(((len(mesh.vertices)),4))
    input_colors[:,0:3] = rgb_colors
    scaled_vertices = mesh.vertices/(np.max(mesh.vertices)-np.min(mesh.vertices))
    m = Mesh.from_faces(scaled_vertices,mesh.faces, colors = input_colors)
    render(
        m,
        [RotateModel(), SaveGif(output_folder+fig_name + ".gif")],
        180,
        camera_position=(1, 1, 1),
        background=(1,)*4
    )
    
def main():
    parser = argparse.ArgumentParser('TopoGrapher', parents=[get_args_parser()])
    args = parser.parse_args()

    object_mesh_path = os.path.join(args.object_folder, 'moomoo_s0.obj')
    mesh = trimesh.load(object_mesh_path)
    mesh_dictionary = get_mesh_data(mesh.vertices, mesh.faces, 10)
    
    blur_time = .00001 # if this gets too small, distances get noisy
    blur_steps = 3
    graph_field_integrator_solomon_2015 = lambda x: blur_on_mesh(x,mesh_dictionary,blur_time,blur_steps)

    options={}
    options['niter'] = args.niter 
    options['tol'] = args.tol 
    options['verb'] = args.verb 

    # compute distances from delta function at a single source to all targets
    sourceVtx = 0 
    source = np.zeros((mesh_dictionary['num_vertices'],mesh_dictionary['num_vertices']))
    source[sourceVtx,:] = 1 / mesh_dictionary['area_weights'][sourceVtx]
    target = np.diag(1 / mesh_dictionary['area_weights'])

    wass_dist = convolutional_distance(source,target,mesh_dictionary['area_weights'],graph_field_integrator_solomon_2015,options);

    print(wass_dist)

    simple3d_save_gif(wass_dist, mesh, "wasserstein_distance", args.output_folder)    
    
if __name__ == '__main__':
    main()
