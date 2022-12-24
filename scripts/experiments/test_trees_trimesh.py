import os
import time

import numpy as np
import trimesh

from ega import default_trimesh_dataset_path
from ega.algorithms.brute_force import BFGFIntegrator
from ega.algorithms.separation_gf_integrator import SeparationGFIntegrator
from ega.algorithms.bartal_trees import BartalTreeGFIntegrator
from ega.algorithms.spanning_trees import SpanningTreeGFIntegrator
from ega.util.mesh_utils import trimesh_to_adjacency_matrices

def get_rel_diff(a, b=None, den=None):
    if not type(a).__module__ == np.__name__:
        a = np.array([a])
        b = np.array([b])  
    if a.ndim==0 or a.shape[0] == a.size or a.ndim>1 and a.shape[1] == a.size: ord = None
    else: ord = 'fro'
    if den is None:
        return np.linalg.norm(a-b, ord=ord) / min(np.linalg.norm(a, ord=ord), np.linalg.norm(b, ord=ord))
    else:
        return np.linalg.norm(a-den, ord=ord) / np.linalg.norm(den, ord=ord)


def main():
    """
    the following example is from GraphFieldIntegrator.ipynb
    """

    # we load busted.STL datafile from default_trimesh_dataset_path. 
    trimesh_datapath = os.path.join(default_trimesh_dataset_path, 'models/busted.STL')
    if os.path.exists(trimesh_datapath):
        mesh = trimesh.load(trimesh_datapath)
    else:
        # in case you did not download trimesh data under this path, we also upload the datafile under
        # data/trimesh/models directory.
        mesh = trimesh.load(os.path.join("data/trimesh/models", "busted.STL"))

    adjacency_lists = trimesh_to_adjacency_matrices(mesh)
    num_edges = 0
    weights_lists = [[] for _ in range(len(adjacency_lists))]
    for i in range(len(adjacency_lists)):
        for j in range(len(adjacency_lists[i])):
            weights_lists[i].append(0.01)
        num_edges += len(adjacency_lists[i])
    print(f"N = {len(adjacency_lists)}, |E| = {num_edges/2}")

    # GENERAL PARAMETERS
    lambda_par = 1.0
    feature_dim = 10
    field = np.ones((len(adjacency_lists), feature_dim))
    f_fun = lambda x: np.exp(-lambda_par * x)
    vertices = np.arange(len(adjacency_lists))

    # SOME OF THE PARAMETERS OF THE BALANCED SEPARATOR APPROACH
    laplace = True
    # parameters to be tuned
    threshold_nb_vertices = 800  # controls the threshold to apply brute force integrator
    unit_size = 1.0  # controls the bucket size for approximation

    # PREPROCESSING
    start = time.time()
    num_trees = 30
    bartal_trees = BartalTreeGFIntegrator(adjacency_lists, weights_lists, vertices, \
                                                    f_fun, num_trees)
    end = time.time()
    print("Constructor for Bartal trees (BT) takes time: ", end - start)
    start = time.time()
    num_trees = 30
    spanning_trees = BartalTreeGFIntegrator(adjacency_lists, weights_lists, vertices, \
                                                    f_fun, num_trees=1)
    end = time.time()
    print("Constructor for Spanning trees (ST) takes time: ", end - start)
    start = time.time()
    sgf_integrator = SeparationGFIntegrator(adjacency_lists, weights_lists, vertices, f_fun,
                                            unit_size=unit_size, threshold_nb_vertices=threshold_nb_vertices)
    end = time.time()
    print("Constructor for Balanced separator (BS) takes time: ", end - start)
    start = time.time()
    bfgf_integrator = BFGFIntegrator(adjacency_lists, weights_lists, vertices, f_fun)
    end = time.time()
    print("Constructor for Brute-force (BF) takes time: ", end - start)

    # Brute force GFI
    start = time.time()
    result_bf = bfgf_integrator.integrate_graph_field(field)
    end = time.time()
    print("Graph field integration for Brute-force (BF) takes time: ", end - start)

    # Bartal trees GFI 
    start = time.time()
    result = bartal_trees.integrate_graph_field(field)
    end = time.time()
    print("Graph field integration for Bartal trees (BT) takes time: ", end - start)
    print(f"{get_rel_diff(result, den=result_bf)  = }")

    # Spanning trees GFI 
    start = time.time()
    result = spanning_trees.integrate_graph_field(field)
    end = time.time()
    print("Graph field integration for Spanning trees (ST) takes time: ", end - start)
    print(f"{get_rel_diff(result, den=result_bf)  = }")

    # Seperator GFI
    start = time.time()
    result = sgf_integrator.integrate_graph_field(field)
    end = time.time()
    print("Graph field integration for Balanced Separator (BS) takes time: ", end - start)
    print(f"{get_rel_diff(result, den=result_bf)  = }")


if __name__ == '__main__':
    main()
