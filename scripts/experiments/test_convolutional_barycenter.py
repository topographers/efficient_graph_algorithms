import argparse
import numpy as np
from ega.data_structures.mesh_data import MeshData
from src.ega.util.blur_on_mesh import blur_on_mesh
from src.ega.algorithms.convolutional_barycenter import ConvolutionalBarycenter
import trimesh
from ega.visualization.mesh_visualization import simple3d_save_gif
import os


def parse_args():
    parser = argparse.ArgumentParser('convolutional_barycenter', add_help=False)

    parser.add_argument(
        '-n', '--number_of_iterations',
        dest='number_of_iterations',
        type=int,
        default=1500,
        help="""number of iterations"""
    )
    parser.add_argument(
        '-t', '--tolerance',
        dest='tolerance',
        type=float,
        default=1e-7,
        help="""stopping tolerance"""
    )
    parser.add_argument(
        '-v', '--verbose',
        dest='verbose',
        type=int, default=1,
        help="""if set to 1, print information at each iteration"""
    )
    parser.add_argument(
        '-m', '--object_folder',
        dest='object_folder',
        type=str,
        default=os.getcwd() + '/meshes',
        help="""path for sample data."""
    )
    parser.add_argument(
        '-o', '--output_folder',
        dest='output_folder',
        type=str,
        default=os.getcwd() + '/gif_files',
        help="""path for saving plots."""
    )
    return parser.parse_args()


def main():
    args = parse_args()

    object_mesh_path = os.path.join(args.object_folder, 'moomoo_s0.obj')
    mesh_data = MeshData(object_mesh_path)

    blur_time = .001  # if this gets too small, distances get noisy
    blur_steps = 3

    def graph_field_integrator_solomon_2015(x):
        return blur_on_mesh(x, mesh_data, blur_time, blur_steps)

    # Design a few functions to average
    center_verts = [300, 100, 600]  # want to adjust this numbers if the input data has less than 600 vertices
    n_functions = len(center_verts)
    distributions = np.zeros((mesh_data.num_vertices, n_functions))
    for i in range(n_functions):
        distributions[center_verts[i] - 1, i] = 1 / mesh_data.area_weights[center_verts[i] - 1]
        distributions[:, i] = blur_on_mesh(distributions[:, i], mesh_data, blur_time, blur_steps)

    alpha = np.ones(3)
    conv_barycenter = ConvolutionalBarycenter()
    barycenter = conv_barycenter.get_convolutional_barycenter(distributions, alpha,
                                                              graph_field_integrator_solomon_2015,
                                                              mesh_data.area_weights)
    print(f'Wasserstein Barycenter: {barycenter}')

    # plot wasserstein barycenter and different input distributions
    output_file_path = os.path.join(args.output_folder, "wasserstein_barycenter.gif")
    simple3d_save_gif(barycenter, mesh_data, output_file_path)
    for i in range(n_functions):
        output_file_path = os.path.join(args.output_folder, f"input_distribution_{i+1}.gif")
        simple3d_save_gif(distributions[:, i], mesh_data, output_file_path)


if __name__ == '__main__':
    main()
