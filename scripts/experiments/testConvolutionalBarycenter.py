import argparse
import numpy as np
from getMeshData import get_mesh_data
from blurOnMesh import blur_on_mesh
from convolutionalBarycenter import convolutional_barycenter
import trimesh
from ega.visualization.mesh_visualization import simple3d_save_gif
import os
import pdb

def get_args_parser():
    parser = argparse.ArgumentParser('convbarycenter', add_help=False)
    parser.add_argument(
        '--niter',
        dest='niter',
        type=int,
        default=1500,
        help="""number of iterations"""
        )
    parser.add_argument(
        '--tol',
        dest='tol',
        type=float,
        default=1e-7,
        help="""stopping tolerance"""
        )
    parser.add_argument(
        '--verb',
        dest='verb',
        type=int, default=1,
        help="""if set to 1, print information at each iteration"""
        )
    parser.add_argument(
        '--object_folder',
        dest='object_folder',
        type=str,
        default=os.getcwd() + '/meshes',
        help="""path for sample data."""
        )
    parser.add_argument(
        '--output_folder',
        dest='output_folder',
        type=str,
        default=os.getcwd() + '/gif_files',
        help="""path for saving plots."""
        )
    return parser

def main():
    parser = argparse.ArgumentParser('TopoGrapher', parents=[get_args_parser()])
    args = parser.parse_args()

    object_mesh_path = os.path.join(args.object_folder, 'moomoo_s0.obj')
    mesh = trimesh.load(object_mesh_path)
    mesh_dictionary = get_mesh_data(mesh.vertices, mesh.faces, 10)
    
    blur_time = .001 # if this gets too small, distances get noisy
    blur_steps = 3
    graph_field_integrator_solomon_2015 = lambda x: blur_on_mesh(x, mesh_dictionary, blur_time, blur_steps)

    # Design a few functions to average
    center_verts = [300, 100, 600] # want to adjust this numbers if the input data has less than 600 vertices
    n_functions = len(center_verts)
    distributions = np.zeros((mesh_dictionary['num_vertices'], n_functions))
    for i in range(n_functions):
        distributions[center_verts[i] - 1,i] = 1 / mesh_dictionary['area_weights'][center_verts[i] - 1]
        distributions[:,i] =  blur_on_mesh(
            distributions[:,i], mesh_dictionary, blur_time, blur_steps)

    options={}
    options['niter'] = args.niter 
    options['tol'] = args.tol 
    options['verb'] = args.verb 
    options['disp'] = []
    options['initial_v'] = np.ones(distributions.shape)
    options['initial_barycenter'] = np.ones((distributions.shape[0], 1))
    options['unit_area_projection'] = 0
    options['Display'] = None
    options['tolfun'] = 1e-4
    options['tolx'] = 1e-4


    alpha = np.ones(3)
    barycenter = convolutional_barycenter(distributions, alpha, mesh_dictionary['area_weights'],
                                          graph_field_integrator_solomon_2015, options)
    print(barycenter)

    # plot wasserstein barycentern and different input distributions
    output_file_path = args.output_folder + "/wasserstein_barycenter" + ".gif"
    simple3d_save_gif(barycenter, mesh, output_file_path)    
    for i in range(n_functions):
        output_file_path = args.output_folder + "/input_distribution_" + str(i + 1) + ".gif"
        simple3d_save_gif(distributions[:,i], mesh, output_file_path)

if __name__ == '__main__':
    main()
