
import argparse
import trimesh 
import os 
import numpy as np 
import torch 
from distutils.util import strtobool
import random 
import open3d as o3d
from torch import nn
from algorithms import BruteForce
from kernel_functions import GaussianKernel
from evaluation import Evaluator



def get_args_parser():
    parser = argparse.ArgumentParser('TopoGrapher', add_help=False)

    # Model parameters
    parser.add_argument('--object_folder', default='../curvox_dataset/meshes/ycb/014_lemon', type=str, help="""path for sample data.""")
    
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1, help='seed of the experiment')
    parser.add_argument('--torch_deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--visualize_mesh', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True, help='whether or not to visualize meshes in open3d')
    
    parser.add_argument('--method', default='BruteForce', type=str, help="""select method to do graph multiplication """)
    parser.add_argument('--kernel', default='GaussianKernel', type=str, help="""select a kernel or function for f""")
    parser.add_argument('--sigma', default=0.1, type=str, help="""a scalar used in gaussian kernel. exp(- sigma * dist(i,j))""")
    parser.add_argument('--use_fp16', type=lambda x: bool(strtobool(x)), default=False, help="""if this is set as True, then we will use torch.float32, otherwise, torch.float64""")
    return parser


  


if __name__ == '__main__':
     
    print("Initial version 0")
    
    parser = argparse.ArgumentParser('TopoGrapher', parents=[get_args_parser()])
    
    args = parser.parse_args()
    
    """
    args = parser.parse_args(['--object_folder','../curvox_dataset/meshes/ycb/014_lemon', 
                              '--method', 'BruteForce',
                              '--kernel','GaussianKernel',
                              '--cuda', 'False',
                              '--visualize_mesh',
                              ])
    """
    
    
    # set device and seed 
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    if device == torch.device(type='cuda'):
        torch.cuda.set_device(args.gpu)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    # load ycb data
    dtype = torch.float32 if args.use_fp16 else torch.float64 
    object_mesh_path = os.path.join(args.object_folder, 'textured.obj')
    mesh = torch.FloatTensor(trimesh.load(object_mesh_path).vertices).to(device) if args.use_fp16 else torch.DoubleTensor(trimesh.load(object_mesh_path).vertices).to(device)
    
    # visualization of meshes 
    if args.visualize_mesh:
        pcd = o3d.io.read_triangle_mesh(object_mesh_path)
        o3d.visualization.draw_geometries([pcd], mesh_show_wireframe = True)
            
    # initialize evaluator 
    evaluator = Evaluator(device = device, f = args.kernel, method = args.method)
    
    # start graph multiplication 
    
    num_pts = mesh.shape[0]
    x = torch.randn(size = (num_pts, 10), dtype = dtype).to(device) # generate a random feature matrix 
    
    if args.kernel == 'GaussianKernel':
        f = GaussianKernel(0.1)
    else:
        print("TODO")
    
    if args.method == 'BruteForce':
        M = torch.cdist(mesh.unsqueeze(0), mesh.unsqueeze(0)).squeeze(0).to(device)
        
        calculator = BruteForce(device, evaluator)
        Mx = calculator(f, M, x)
    else:
        print("TODO")
        
    evaluator.print_statistics()
        
    
    
    