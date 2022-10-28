import argparse
import numpy as np
import trimesh
import os
from ega.util.get_mesh_data import get_mesh_data
from ega.util.blur_on_mesh import blur_on_mesh
from ega.algorithms.convolutional_barycenter import ConvolutionalBarycenter
from ega.visualization.mesh_visualization import simple3d_save_gif
from ega.algorithms.separation_gf_integrator import PlaceboSeparationGFIntegrator
from ega.algorithms.brute_force import BFGFIntegrator
from ega.util.gaussian_kernel import GaussianKernel
import ega.util.mesh_utils as mu
from ega.evaluation.evaluator import Evaluator
from memory_profiler import profile
from line_profiler import LineProfiler 
import networkx
import time

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
    
    blur_time = .03 # if this gets too small, distances get noisy
    blur_steps = 1
    graph_field_integrator_solomon_2015 = lambda x: blur_on_mesh(x, mesh_dictionary, blur_time, blur_steps)

    # Design a few functions to average
    center_verts = [300, 100, 600] # want to adjust this numbers if the input data has less than 600 vertices
    n_functions = len(center_verts)
    distributions = np.zeros((mesh_dictionary['num_vertices'], n_functions))
    for i in range(n_functions):
        distributions[center_verts[i] - 1,i] = 1 / mesh_dictionary['area_weights'][center_verts[i] - 1]
        distributions[:,i] =  blur_on_mesh(
            distributions[:,i], mesh_dictionary, blur_time, blur_steps)
    alpha = np.ones(3)

    # create adjacency lists and weights lists
    adjacency_lists = mu.trimesh_to_adjacency_matrices(mesh, seed=0)  
    sigma = 0.25
    kernel_function = GaussianKernel(sigma)
    vertices = np.arange(mesh_dictionary['num_vertices'])
    weights_lists = []
    for i in range(len(adjacency_lists)):
        weights_lists.append([1 for i in adjacency_lists[i]])

    # brute force integrator initialization
    start_time = time.time()
    brute_force = BFGFIntegrator(adjacency_lists, weights_lists, vertices, kernel_function)
    end_time = time.time()
    print('brute force preprocessing time: ', end_time - start_time)

    # planer separator integrator initialization
    start_time = time.time()
    psgf_integrator = PlaceboSeparationGFIntegrator(adjacency_lists, weights_lists, vertices, kernel_function)
    end_time = time.time()
    print('planer separator preprocessing time: ', end_time - start_time)

    # compare brute force integrator with planer separation integrator on wass barycenter task
    conv_barycenter = ConvolutionalBarycenter()
    start_time = time.time()
    barycenter_brute_force = conv_barycenter.get_convolutional_barycenter(distributions, alpha,
                     brute_force.integrate_graph_field, mesh_dictionary['area_weights'])
    end_time = time.time()
    print('brute force compute time: ', end_time - start_time)

    start_time = time.time()
    barycenter_psgf = conv_barycenter.get_convolutional_barycenter(distributions, alpha,
                     psgf_integrator.integrate_graph_field, mesh_dictionary['area_weights'])
    end_time = time.time()
    print('planer separator compute time: ', end_time - start_time)

    print('Wasserstein Barycenter, brute force: ')
    print(barycenter_brute_force)
    print('Wasserstein Barycenter, planer separator: ')
    print(barycenter_psgf)
    
    # plot wasserstein barycentern and different input distributions
    output_file_path = args.output_folder + "/bfgf_wass_barycenter" + ".gif"
    simple3d_save_gif(barycenter_brute_force, mesh, output_file_path)    
    output_file_path = args.output_folder + "/psgf_wass_barycenter" + ".gif"
    simple3d_save_gif(barycenter_psgf, mesh, output_file_path)    
    for i in range(n_functions):
        output_file_path = args.output_folder + "/input_distribution_" + str(i + 1) + ".gif"
        simple3d_save_gif(distributions[:,i], mesh, output_file_path)


if __name__ == '__main__':
    main()