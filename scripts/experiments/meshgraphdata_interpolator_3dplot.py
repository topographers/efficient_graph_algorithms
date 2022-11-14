import copy
import os
import pickle
import random

from ega import default_meshgraphnet_dataset_path
from ega.algorithms.brute_force import BFGFIntegrator
from ega.util.gaussian_kernel import GaussianKernel
from ega.util.interpolator import Interpolator
from ega.util.mesh_utils import calculate_interpolation_metrics
from ega.visualization.mesh_visualization import plot_mesh


def main():
    trajactory_index = 0  # specifies the pkl file id to be used for generating snapshots
    snapshot_index = 7  # the number of snapshot of current trajectory to be used
    mask_ratio = 0.03  # divide the known and unknown vertices according to this mask ratio
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
    field = mesh_data['node_features'][:, :3]  # the first three columns represent velocities
    world_pos = mesh_data['world_pos']
    # faces = mesh_data['faces']
    n_vertices = len(vertices)

    # divide vertices into known vertices and vertices to be interpolated
    random.seed(0)
    vertices_interpolate = random.sample(vertices, int(mask_ratio * n_vertices))
    vertices_known = list(set(vertices) - set(vertices_interpolate))
    true_fields = field[vertices_interpolate]
    vertices_interpolate_pos = world_pos[vertices_interpolate]
    n_vertices_interpolate = len(vertices_interpolate)

    # create integrator and interpolator
    f_fun = GaussianKernel(sigma)
    brute_force = BFGFIntegrator(adjacency_list, weight_list, vertices, f_fun)
    interpolator = Interpolator(brute_force, vertices_known, vertices_interpolate)
    # mask out vertices to be interpolated 
    interpolator.integrator._m_matrix[vertices_interpolate, :][:, vertices_interpolate] = 0
    interpolator.integrator._m_matrix /= interpolator.integrator._m_matrix.sum(axis=0, keepdims=True)
    interpolated_fields = interpolator.interpolate(copy.deepcopy(field))

    plot_mesh(world_pos, vertices_interpolate_pos, true_fields, interpolated_fields, snapshot_index)
    calculate_interpolation_metrics(true_fields, interpolated_fields)


if __name__ == '__main__':
    main()
