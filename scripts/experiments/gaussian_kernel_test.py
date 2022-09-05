import trimesh
import numpy as np
import random

import os
import sys  
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../", "src")))  

from ega.algorithms.brute_force import BruteForce
from ega.util.gaussian_kernel import GaussianKernel
import ega.util.mesh_utils as mu
from ega.evaluation.evaluator import Evaluator

import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from memory_profiler import profile
from line_profiler import LineProfiler 
import networkx 


@profile
def evaluate_brute_force_memory(adjacency_lists, kernel_function, graph_field):
    brute_force = BruteForce(adjacency_lists, kernel_function)
    Mx = brute_force.model_top_field(graph_field)

def evaluate_brute_force_time(adjacency_lists, kernel_function, graph_field):
    brute_force = BruteForce(adjacency_lists, kernel_function)
    Mx = brute_force.model_top_field(graph_field)


if __name__ == '__main__':

    # load sample mesh data 
    default_curvox_dataset_path = os.path.abspath(os.path.join("../../", "data", "curvox_dataset"))
    mesh_file = os.path.join(default_curvox_dataset_path, "meshes/ycb/014_lemon", 'nontextured.stl')
    mesh = trimesh.load(mesh_file)
    graph = trimesh.graph.vertex_adjacency_graph(mesh)
    print(graph)

    # proprocess 
    adjacency_lists = mu.trimesh_to_adjacency_matrices(mesh, seed=0)  
    n_vertices = mesh.vertices.shape[0]
    kernel_function = GaussianKernel(0.1)
    graph_field = np.random.randn(n_vertices, 1)

    # evaluate memory 
    evaluate_brute_force_memory(adjacency_lists, kernel_function, graph_field)

    # evaluate time
    lp = LineProfiler()
    lp_wrapper = lp(evaluate_brute_force_time)
    lp_wrapper(adjacency_lists, kernel_function, graph_field)
    lp.print_stats()
   



