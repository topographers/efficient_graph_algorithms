import os
import time

import numpy as np
import networkx as nx


from ega.algorithms.brute_force import BFGFIntegrator
from ega.algorithms.bartal_trees import BartalTreeGFIntegrator
from ega.util.gaussian_kernel import GaussianKernel


def get_adjacency_lists_from_A(A):
    n = A.shape[0]
    adjacency_lists = [[] for _ in range(n)]
    weights_lists = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if A[i,j] != 0:
                adjacency_lists[i] += [j]
                adjacency_lists[j] += [i]
                weights_lists[i] += [A[i,j]]
                weights_lists[j] += [A[i,j]]
    return adjacency_lists, weights_lists


def get_adjacency_nx(G):
    A_srs = nx.adjacency_matrix(G)
    A = np.array(A_srs.todense())
    print(f"{A[A > 0].min() = }, {A[A > 0].max() = }")
    return A


def get_rel_diff(a, b=None):
    if not type(a).__module__ == np.__name__:
        a = np.array([a])
        b = np.array([b])  
    if a.ndim==0 or a.shape[0] == a.size or a.ndim>1 and a.shape[1] == a.size: ord = None
    else: ord = 'fro'
    return np.linalg.norm(a-b, ord=ord) / min(np.linalg.norm(a, ord=ord), np.linalg.norm(b, ord=ord))


def main():
    # GENERAL PARAMETERS
    n = 1000
    k = 10
    f_fun = lambda x: np.exp(-x)
    vertices = np.arange(n)
    num_samples = 20

    # TESTING matvec_dp function, which is exact for a tree
    
    # RANDOM TREE
    G = nx.random_tree(n=n)
    for (u, v) in G.edges():
        G.edges[u,v]['weight'] = np.random.rand()*10

    A = get_adjacency_nx(G)
    tadj_lists, tw_lists = get_adjacency_lists_from_A(A)

    # BRUTE FORCE GFI
    brute_force = BFGFIntegrator(tadj_lists, tw_lists, n, f_fun)
    # BARTAL TREE GFI
    bartal_trees = BartalTreeGFIntegrator(tadj_lists, tw_lists, vertices, f_fun)

    # distance matrix match test
    dist_T = bartal_trees._distance_matrix(tadj_lists, tw_lists)
    assert np.allclose(f_fun(dist_T), brute_force.get_kernel_graph()), \
        print("distance matrices not computed correctly")

    root = np.random.randint(n)
    levels, parents = bartal_trees._tree_root2leaf_levels_parents(root, tadj_lists)
    # construct a tree data structure
    tree = {'root':root, 'parents':parents, \
            'adj':tadj_lists,  'w':tw_lists, \
            'levels':levels}
    print(f"{n = }, {len(levels) = }")

    for _ in range(num_samples):
        field = np.random.randn(n, k)
        true_x = brute_force.integrate_graph_field(field)
        x = bartal_trees._matvec_dynamic_programming(tree, field)
        assert np.allclose(x, true_x) and get_rel_diff(x, true_x) < 1e-6, \
            print("_matvec_dynamic_programming is not implemented correctly")
    
    print("PASSED matvec_dp implementation tests")

if __name__ == '__main__':
    main()