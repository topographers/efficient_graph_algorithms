import trimesh
import numpy as np
import random
import os

from ega.algorithms.brute_force import BFGFIntegrator
from ega.util.gaussian_kernel import GaussianKernel
import ega.util.mesh_utils as mu
from ega.evaluation.evaluator import Evaluator

import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from memory_profiler import profile
from line_profiler import LineProfiler 
import networkx 

from ega import default_curvox_dataset_path


@profile
def evaluate_brute_force_memory(adjacency_lists, weights_lists, vertices, f_fun, field):
    brute_force = BFGFIntegrator(adjacency_lists, weights_lists, vertices, f_fun)
    Mx = brute_force.integrate_graph_field(field)

def evaluate_brute_force_time(adjacency_lists, weights_lists, vertices, f_fun, field):
    brute_force = BFGFIntegrator(adjacency_lists, weights_lists, vertices, f_fun)
    Mx = brute_force.integrate_graph_field(field)

def main():    
    # load sample mesh data 
    mesh_file = os.path.join(default_curvox_dataset_path, "meshes/ycb/014_lemon", 'nontextured.stl')
    mesh = trimesh.load(mesh_file)
    graph = trimesh.graph.vertex_adjacency_graph(mesh)
    print(graph)

    # proprocess 
    adjacency_lists = mu.trimesh_to_adjacency_matrices(mesh, seed=0)
    n_vertices = mesh.vertices.shape[0]
    f_fun = GaussianKernel(0.1)
    field = np.random.randn(n_vertices, 1)
    weights_lists = mu.generate_weights_from_adjacency_list(adjacency_lists) 
    vertices = list(range(n_vertices))

    # evaluate time
    lp = LineProfiler()
    lp_wrapper = lp(evaluate_brute_force_time)
    lp_wrapper(adjacency_lists, weights_lists, vertices, f_fun, field)
    lp.print_stats()

    # evaluate memory 
    evaluate_brute_force_memory(adjacency_lists, weights_lists, vertices, f_fun, field)


if __name__ == '__main__':
    main()
   


