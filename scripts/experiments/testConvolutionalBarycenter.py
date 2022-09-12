import argparse
import numpy as np
from getMeshData import get_mesh_data
from blurOnMesh import blur_on_mesh
from convolutionalBarycenter import convolutional_barycenter
import trimesh
import pdb
import os


def get_args_parser():
    parser = argparse.ArgumentParser('convbarycenter', add_help=False)
    parser.add_argument('--niter', dest='niter', type=int, default=1500,
                        help="""number of iterations""")
    parser.add_argument('--tol', dest='tol', type=float, default=1e-7,
                        help="""stopping tolerance""")
    parser.add_argument('--verb', dest='verb', type=int, default=1,
                        help="""if set to 1, print information at each iteration""")
    parser.add_argument('--object_folder', dest='object_folder', type=str, default='./Solomon_2015/meshes',
                        help="""path for sample data.""")
    return parser
 
def main():
    parser = argparse.ArgumentParser('TopoGrapher', parents=[get_args_parser()])
    args = parser.parse_args()

    object_mesh_path = os.path.join(args.object_folder, 'moomoo_s0.obj')
    mesh = trimesh.load(object_mesh_path)
    mesh_dictionary = get_mesh_data(mesh.vertices, mesh.faces, 10)
    
    blur_time = .001 # if this gets too small, distances get noisy
    blur_steps = 3

    blur = lambda x: blur_on_mesh(x,mesh_dictionary,blur_time,blur_steps)
    blur_transpose = lambda x: blur_on_mesh(x,mesh_dictionary,blur_time,blur_steps,1)

    # Design a few functions to average

    center_verts = [300, 100, 600] # want to adjust this numbers if the input data has less than 600 vertices
    n_functions = len(center_verts)
    distributions = np.zeros((mesh_dictionary['num_vertices'],n_functions))
    area_weight_np = np.array(mesh_dictionary['area_weights'].todense()).T[0]

    for i in range(n_functions):
        distributions[center_verts[i]-1,i] = 1 / area_weight_np[center_verts[i]-1]
        # index - 1 because matlab index starts at 1
        distributions[:,i] =  blur_on_mesh(
            distributions[:,i],mesh_dictionary,blur_time,blur_steps)

    options={}
    options['niter'] = args.niter # 1500
    options['tol'] = args.tol # 1e-7
    options['verb'] = args.verb # 1
    options['disp'] = []
    options['initial_v'] = np.ones(distributions.shape)
    options['initial_barycenter'] = np.ones((distributions.shape[0], 1))
    options['unit_area_projection'] = 0
    options['Display'] = None
    options['tolfun'] = 1e-4
    options['tolx'] = 1e-4


    alpha = np.ones(3)
    barycenter, _ = convolutional_barycenter(distributions,alpha,mesh_dictionary['area_weights'],blur,blur_transpose,options)
    print(barycenter)

if __name__ == '__main__':
    main()
