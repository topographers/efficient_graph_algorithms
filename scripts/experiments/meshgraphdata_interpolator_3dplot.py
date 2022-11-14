import numpy as np
import random
import os
import pickle
import copy
import trimesh
import copy
import time

import plotly.graph_objects as go
from plotly.offline import plot
import plotly.figure_factory as ff

from ega import default_meshgraphnet_dataset_path
from ega.algorithms.brute_force import BFGFIntegrator
from ega.algorithms.separation_gf_integrator import SeparationGFIntegrator
from ega.util.gaussian_kernel import GaussianKernel
from ega.util.interpolator import Interpolator
from ega.util.mesh_utils import calculate_interpolation_metrics
from ega.visualization.mesh_visualization import plot_mesh


def main():
    trajactory_index = 0  # specifies the pkl file id to be used for generating snapshots
    snapshot_index = 5  # the number of snapshot of current trajectory to be used
    mask_ratio = 0.02  # divide the known and unknown vertices according to this mask ratio
    scale = 10  # scales the velocity vector for better visualization
    sigma = 10  # parameter in gaussian kernel

    # read data
    meshgraph_path = os.path.join(default_meshgraphnet_dataset_path, 'flag_simple', 'processed_data')
    meshgraph_file = os.path.join(meshgraph_path, 'trajectory_{}.pkl'.format(trajactory_index))
    mesh_data = pickle.load(open(meshgraph_file, 'rb'))[snapshot_index]
    print(mesh_data.keys())

    vertices = mesh_data['vertices']
    adjacency_list = mesh_data['adjacency_list']
    weight_list = mesh_data['weight_list']
    all_fields = mesh_data['node_features'][:, :3]  # the first three columns represent velocities
    world_pos = mesh_data['world_pos']
    n_vertices = len(vertices)

    # divide vertices into known vertices and vertices to be interpolated
    random.seed(0)
    vertices_interpolate = random.sample(vertices, int(mask_ratio * n_vertices))
    vertices_known = list(set(vertices) - set(vertices_interpolate))
    true_fields = all_fields[vertices_interpolate]
    vertices_interpolate_pos = world_pos[vertices_interpolate]
    n_vertices_interpolate = len(vertices_interpolate)
    f_fun = GaussianKernel(sigma)

    # field_known represents the existed fields on the manifold
    # our goal is to predict the fields that is unknown (the points to be interpolated)
    field_known = copy.deepcopy(all_fields)
    field_known[vertices_interpolate] = 0
    ones_known = np.ones(shape=(len(field_known), 1))
    ones_known[vertices_interpolate] = 0

    '''  brute force integrator  '''
    brute_force = BFGFIntegrator(adjacency_list, weight_list, vertices, f_fun)
    interpolator_bf = Interpolator(brute_force, vertices_known, vertices_interpolate)
    # mask out vertices to be interpolated 
    interpolator_bf.integrator._m_matrix[vertices_interpolate, :][:, vertices_interpolate] = 0
    interpolator_bf.integrator._m_matrix /= \
        interpolator_bf.integrator._m_matrix.sum(axis=0, keepdims=True)

    start = time.time()
    interpolated_fields_bf = interpolator_bf.interpolate(copy.deepcopy(field_known))
    print("Brute-force GF Interpolator takes: {} seconds".format(time.time() - start))

    # the following two lines of code can print the statistics and plot the fields between true fields and predicted
    # fields from seperator GF Integrator
    """
    plot_mesh(world_pos, vertices_interpolate_pos, true_fields, interpolated_fields_bf, snapshot_index)
    calculate_interpolation_metrics(true_fields, interpolated_fields_bf)
    """

    '''  seperation graph-field integrator  '''
    laplace = True
    threshold_nb_vertices = 3000  # controls the threshold to apply brute force integrator
    unit_size = 0.001  # controls the bucket size for approximation

    sgf_integrator = SeparationGFIntegrator(adjacency_list, weight_list, vertices, f_fun,
                                            unit_size=unit_size, threshold_nb_vertices=threshold_nb_vertices)
    interpolator_sgf = Interpolator(sgf_integrator, vertices_known, vertices_interpolate)

    start = time.time()
    interpolated_fields_sgf = interpolator_sgf.interpolate(copy.deepcopy(field_known)) / \
                              interpolator_sgf.interpolate(copy.deepcopy(ones_known))
    print("Seperator GF Interpolator takes: {} seconds".format(time.time() - start))

    # the following two lines of code can print the statistics and plot the fields between true fields and predicted
    # fields from seperator GF Integrator
    """
    plot_mesh(world_pos, vertices_interpolate_pos, true_fields, interpolated_fields_sgf, snapshot_index)
    calculate_interpolation_metrics(true_fields, interpolated_fields_sgf)
    """

    plot_mesh(world_pos, vertices_interpolate_pos, interpolated_fields_bf, interpolated_fields_sgf, snapshot_index,
              names=['Brute-force GF Integrator', 'Seperator GF Integrator'])
    calculate_interpolation_metrics(interpolated_fields_bf, interpolated_fields_sgf)


if __name__ == '__main__':
    main()
