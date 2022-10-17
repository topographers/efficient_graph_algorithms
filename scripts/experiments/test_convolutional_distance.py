import argparse
import numpy as np
import trimesh
import os
from ega.util.get_mesh_data import get_mesh_data
from ega.util.blur_on_mesh import blur_on_mesh
from ega.algorithms.convolutional_distance import ConvolutionalDistance
from ega.visualization.mesh_visualization import simple3d_save_gif

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
    
    blur_time = .00001 # if this gets too small, distances get noisy
    blur_steps = 3
    graph_field_integrator_solomon_2015 = lambda x: blur_on_mesh(x, mesh_dictionary, blur_time, blur_steps)

    # compute distances from delta function at a single source to all targets
    sourceVtx = 0 
    source = np.zeros((mesh_dictionary['num_vertices'], mesh_dictionary['num_vertices']))
    source[sourceVtx,:] = 1 / mesh_dictionary['area_weights'][sourceVtx]
    target = np.diag(1 / mesh_dictionary['area_weights'])

    conv_distance = ConvolutionalDistance()
    wass_dist = conv_distance.get_convolutional_distance(source, target,
                    mesh_dictionary['area_weights'], graph_field_integrator_solomon_2015)
    print('Wasserstein Distance: ')
    print(wass_dist)

    output_file_path = args.output_folder + "/wasserstein_distance" + ".gif"
    simple3d_save_gif(wass_dist, mesh, output_file_path)    
    
if __name__ == '__main__':
    main()
