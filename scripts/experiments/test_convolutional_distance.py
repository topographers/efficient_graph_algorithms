import argparse
import numpy as np
from ega.data_structures.mesh_data import MeshData
from src.ega.util.blur_on_mesh import blur_on_mesh
import trimesh
from src.ega.algorithms.convolutional_distance import ConvolutionalDistance
from ega.visualization.mesh_visualization import simple3d_save_gif
import os


def parse_args():
    parser = argparse.ArgumentParser('convolutional_distance', add_help=False)

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

    # TODO: Make these parameters
    blur_time = .00001  # if this gets too small, distances get noisy
    blur_steps = 3

    def graph_field_integrator_solomon_2015(x):
        return blur_on_mesh(x, mesh_data, blur_time, blur_steps)

    # compute distances from delta function at a single source to all targets
    sourceVtx = 0
    source = np.zeros((mesh_data.num_vertices, mesh_data.num_vertices))
    source[sourceVtx, :] = 1 / mesh_data.area_weights[sourceVtx]
    target = np.diag(1 / mesh_data.area_weights)

    conv_distance = ConvolutionalDistance()
    wasserstein_distance = conv_distance.get_convolutional_distance(source, target,
                                                                    mesh_data.area_weights,
                                                                    graph_field_integrator_solomon_2015)
    print(f'Wasserstein Distance: {wasserstein_distance}')

    output_file_path = os.path.join(args.output_folder, "wasserstein_distance.gif")
    simple3d_save_gif(wasserstein_distance, mesh_data, output_file_path)


if __name__ == '__main__':
    main()
