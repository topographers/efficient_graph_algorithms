import argparse
import numpy as np
import trimesh
import os
from ega.util.get_mesh_data import get_mesh_data
from ega.util.blur_on_mesh import blur_on_mesh
from ega.algorithms.convolutional_distance import ConvolutionalDistance
from ega.visualization.mesh_visualization import simple3d_save_gif
from ega.algorithms.brute_force import BFGFIntegrator
from ega.algorithms.separation_gf_integrator import PlaceboSeparationGFIntegrator
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
        default=100,
        help="""number of iterations"""
        )
    parser.add_argument(
        '--tol',
        dest='tol',
        type=float,
        default=1e-6,
        help="""stopping tolerance"""
        )
    parser.add_argument(
        '--verb',
        dest='verb',
        type=int,
        default=1,
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

    # create adjacency lsits and weights lists
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
    
    # separation integrator initialization
    start_time = time.time()
    psgf_integrator = PlaceboSeparationGFIntegrator(adjacency_lists, weights_lists, vertices, kernel_function)
    end_time = time.time()
    print('planer separator preprocessing time: ', end_time - start_time)

    # compute distances from delta function at a single source to all targets
    sourceVtx = 0 
    source = np.zeros((mesh_dictionary['num_vertices'], mesh_dictionary['num_vertices']))
    source[sourceVtx,:] = 1 / mesh_dictionary['area_weights'][sourceVtx]
    target = np.diag(1 / mesh_dictionary['area_weights'])
    conv_distance = ConvolutionalDistance()

    # compare brute force integrator with planer separation integrator on wass barycenter task
    start_time = time.time()
    wass_dist_brute_force = conv_distance.get_convolutional_distance(source, target,
                    mesh_dictionary['area_weights'], brute_force.integrate_graph_field)
    end_time = time.time()
    print('brute force compute time: ', end_time - start_time)

    start_time = time.time()
    wass_dist_psgf = conv_distance.get_convolutional_distance(source, target,                                                              
                    mesh_dictionary['area_weights'], psgf_integrator.integrate_graph_field)
    end_time = time.time()
    print('planer separator compute time: ', end_time - start_time)

    print('Wasserstein Distance, brute force: ')
    print(wass_dist_brute_force)
    print('Wasserstein Distance, planer separator: ')
    print(wass_dist_psgf)
    
    output_file_path = args.output_folder + "/bfgf_wass_distance" + ".gif"
    simple3d_save_gif(wass_dist_brute_force, mesh, output_file_path)    
    
if __name__ == '__main__':
    main()
