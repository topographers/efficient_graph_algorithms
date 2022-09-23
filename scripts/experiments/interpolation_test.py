import numpy as np
import random
import os
import pickle
import copy 

from ega import default_meshgraphnet_dataset_path
from ega.algorithms.brute_force import BFGFIntegrator
from ega.util.gaussian_kernel import GaussianKernel
from ega.util.interpolator import Interpolator


def main():    
    
    # load sample mesh data 
    trajactory_index = 0
    snapshot_index = 10 
    meshgraph_path = os.path.join(default_meshgraphnet_dataset_path, 'flag_simple', 'processed_data')
    meshgraph_file = os.path.join(meshgraph_path, 'trajectory_{}.pkl'.format(trajactory_index))
    mesh_data = pickle.load(open(meshgraph_file,'rb'))[snapshot_index]
    print(mesh_data.keys())
    
    vertices = mesh_data['vertices']
    adjacency_list = mesh_data['adjacency_list']
    weight_list = mesh_data['weight_list']
    field = mesh_data['node_features']
    n_vertices = len(vertices)
    
    f_fun = GaussianKernel(0.1)
    
    mask_ratio = 0.1
    vertices_interpolate = random.sample(vertices, int(mask_ratio * n_vertices))
    vertices_known = list(set(vertices) - set(vertices_interpolate))
    
    brute_force = BFGFIntegrator(adjacency_list, weight_list, vertices, f_fun)
    
    interpolator = Interpolator(brute_force, vertices_known, vertices_interpolate)
    interpolated_fields = interpolator.interpolate(field)

    true_fields = field[vertices_interpolate]
    MSE = np.linalg.norm(interpolated_fields - true_fields)
    print("MSE for interpolation", MSE)


if __name__ == '__main__':
    main()
   


