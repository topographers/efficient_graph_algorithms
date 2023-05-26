import argparse
import numpy as np
import trimesh
import os
from ega.util.get_mesh_data import get_mesh_data
from ega.util.blur_on_mesh import blur_on_mesh
from ega.algorithms.convolutional_barycenter import ConvolutionalBarycenter
from ega.visualization.mesh_visualization import simple3d_save_gif
from ega.algorithms.separation_gf_integrator import SeparationGFIntegrator
from ega.util.separator_utils import base_compute_clustering
from ega.algorithms.lt_separation_finder import LTSeparationFinder
from ega.util.gaussian_kernel import GaussianKernel
import ega.util.mesh_utils as mu
from ega.evaluation.evaluator import Evaluator
from ega.algorithms.graph_diffusion_gf_integrator import DFGFIntegrator
from ega.util.mesh_utils import random_projection_creator, density_function, fourier_transform
from memory_profiler import profile
from line_profiler import LineProfiler
from sklearn.metrics import pairwise_distances
from sklearn.metrics import mean_squared_error
from scipy import linalg
import networkx
import time
import pdb

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
        default=0.03,
        help="""stopping tolerance"""
        )
    parser.add_argument(
        '--verb',
        dest='verb',
        type=int, default=True,
        help="""if set to True, print information at each iteration"""
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
    parser.add_argument(
        '--sparse',
        dest='sparse',
        type=bool,
        default=True,
        help="""if we use sparse implementation for (Solomon, 2015)"""
        )
    return parser

def main():
    parser = argparse.ArgumentParser('TopoGrapher', parents=[get_args_parser()])
    args = parser.parse_args()

    obj_name = 'duck'
    object_mesh_path = os.path.join(args.object_folder, obj_name + '.obj') 
    mesh = trimesh.load(object_mesh_path)

    # (Solomon, 2015) method initialization. The runtime is mostly Cotangent Laplacian computations.
    blur_time = .2 # if this gets too small, distances get noisy
    blur_steps = 2
    start_time = time.time()
    mesh_dictionary = get_mesh_data(mesh.vertices, mesh.faces, blur_time, blur_steps, args.sparse)   
    graph_field_integrator_solomon_2015 = lambda x: blur_on_mesh(x, mesh_dictionary)
    end_time = time.time()
    solomon_pre_time = end_time - start_time
    print('(Solomon, 2015) preprocessing time: ', end_time - start_time)

    # Design a few functions to average
    if obj_name == 'alien_pilot_standing':
        center_verts = [1000, 3000, 5000]
    elif obj_name == 'duck':
        center_verts = [1000, 5000, 6000]
    elif obj_name == 'land':
        center_verts = [1000, 5000, 10000]
    elif obj_name == 'octocat-v1':
        center_verts = [1000, 8000, 100]
    n_functions = len(center_verts)
    distributions = np.zeros((mesh_dictionary['num_vertices'], n_functions))
    for i in range(n_functions):
        distributions[center_verts[i] - 1,i] = 1 / mesh_dictionary['area_weights'][center_verts[i] - 1]
        distributions[:,i] =  blur_on_mesh(distributions[:,i], mesh_dictionary)
    alpha = np.ones(3) # alpha is normalized when computing conv wass barycenter. 

    # create adjacency lists and weights lists
    lambda_par = 0.1
    epsilon = 0.01 
    number_points = mesh_dictionary['num_vertices']
    dim = 3
    num_rand_features = 30
    positions = mesh_dictionary['vertices']
    positions = (positions-positions.mean()) / (positions.max()*10)

    # graph diffusion integrator initialization
    start_time = time.time()
    dfgf_integrator = DFGFIntegrator(positions,  epsilon, lambda_par, 
                                     num_rand_features, dim, 
                                     random_projection_creator, density_function, 
                                     fourier_transform)
    end_time = time.time()
    dfgf_pre_time = end_time - start_time
    print('graph diffusion preprocessing time: ', end_time - start_time)
    
    # brute force integrator initialization
    class BFGFIntegrator_diffusion:
        def __init__(self, positions, epsilon, lambda_par):
            self.dist_matrix = pairwise_distances(positions, metric='l1') # use L1 distance here because the Fourier Transform in Diffusion integrator uses L1 norm
            self.dist_matrix[self.dist_matrix > epsilon]=0
            self.M  = linalg.expm(lambda_par * self.dist_matrix) 
        def integrate_graph_field(self, field):
            return self.M@field
    start_time = time.time()
    bf_integrator = BFGFIntegrator_diffusion(positions, epsilon, lambda_par)
    end_time = time.time()
    bfgf_pre_time = end_time - start_time
    print('brute force preprocessing time: ', end_time - start_time)

    # define the barycenter problem
    conv_barycenter = ConvolutionalBarycenter(niter=args.niter, tol=args.tol, verb=args.verb)

    # (Solomon, 2015) method integration. Note this method does not consider an epsilon-graph. so discrepancy is expected 
    start_time = time.time()
    barycenter_solomon = conv_barycenter.get_convolutional_barycenter(distributions, alpha,
                     graph_field_integrator_solomon_2015, mesh_dictionary['area_weights'])
    end_time = time.time()
    solomon_compute_time = end_time - start_time
    print('(Solomon, 2015) compute time: ', end_time - start_time)
  
    # brute force integrator integration
    start_time = time.time()
    barycenter_brute_force = conv_barycenter.get_convolutional_barycenter(distributions, alpha,
                     bf_integrator.integrate_graph_field, mesh_dictionary['area_weights'])
    end_time = time.time()
    bfgf_compute_time = end_time - start_time
    print('brute force compute time: ', end_time - start_time)

    # graph diffusion kernel integrator integration
    start_time = time.time()
    barycenter_dfgf = conv_barycenter.get_convolutional_barycenter(distributions, alpha,
                     dfgf_integrator.integrate_graph_field, mesh_dictionary['area_weights'])
    end_time = time.time()
    dfgf_compute_time = end_time - start_time
    print("graph diffusion compute time: ", end_time - start_time)

    print('Wasserstein Barycenter, (Solomon, 2015): ')
    print(barycenter_solomon)
    print('Wasserstein Barycenter, brute force: ')
    print(barycenter_brute_force)
    print('Wasserstein Barycenter, graph kernel: ')
    print(barycenter_dfgf)

    # mean squared error
    print('mean squared error of diffusion integrator')
    dfgf_mse = mean_squared_error(barycenter_dfgf, barycenter_brute_force)
    print(dfgf_mse)
    print('mean squared error of (Solomon, 2015)')
    solomon_mse = mean_squared_error(barycenter_dfgf, barycenter_solomon)
    print(solomon_mse)
    
    # plot wasserstein barycenter and different input distributions
    output_file_path = args.output_folder + "/bfgf_wass_barycenter" + ".gif"
    simple3d_save_gif(barycenter_brute_force, mesh, output_file_path)     
    output_file_path = args.output_folder + "/dfgf_wass_barycenter" + ".gif"
    simple3d_save_gif(barycenter_dfgf, mesh, output_file_path)
    output_file_path = args.output_folder + "/solomon_wass_barycenter" + ".gif"
    simple3d_save_gif(barycenter_solomon, mesh, output_file_path)
    for i in range(n_functions):
        output_file_path = args.output_folder + "/input_distribution_" + str(i + 1) + ".gif"
        simple3d_save_gif(distributions[:,i], mesh, output_file_path)

    # sanity check
    avg_nonzero_entries = np.average(np.sum(np.abs(bf_integrator.M) > 0.001, axis=1))
    print('num of nonzero entries per row: ', avg_nonzero_entries)

    # save log files
    outfile = "log_file.txt"
    pre_time_arr = [dfgf_pre_time, solomon_pre_time, bfgf_pre_time]
    compute_time_arr = [dfgf_compute_time, solomon_compute_time, bfgf_compute_time]
    mse_arr = [dfgf_mse, solomon_mse]
    with open(outfile, "w") as f:
        f.write("pre-processing time (dfgf, solomon, bfgf):\n")
        [f.write('{:.4f} '.format(item)) for item in pre_time_arr]
        f.write("computation time (dfgf, solomon, bfgf):\n")
        [f.write('{:.4f} '.format(item)) for item in compute_time_arr]
        f.write("mse (dfgf, solomon):\n")
        [f.write('{:.4f} '.format(item)) for item in mse_arr]
        f.write("avg nonzero entries per row:\n")
        f.write('{:.4f} '.format(avg_nonzero_entries))       



if __name__ == '__main__':
    main()
