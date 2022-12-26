import numpy as np
import networkx as nx


from ega.algorithms.brute_force import BFGFIntegrator
from ega.algorithms.spanning_trees import SpanningTreeGFIntegrator
from graphs_networkx_utils import *


def main():
    # GENERAL PARAMETERS
    n = 1000
    k = 10
    f_fun = lambda x: np.exp(-x)
    vertices = np.arange(n)
    num_samples = 10

    ## CREATE GRAPHS
    graphs = {}
    # Random tree
    print("Random tree graph")
    G = nx.random_tree(n=n)
    for (u, v) in G.edges():
        G.edges[u,v]['weight'] = np.random.rand()*10
    A = get_adjacency_nx(G)
    adjacency_lists, weights_lists = get_adjacency_lists_from_A(A)
    graphs['tree'] = (adjacency_lists, weights_lists)
    # Regular graph
    print("Regular random graph")
    G = nx.random_regular_graph(5, n)
    A = get_adjacency_nx(G)
    adjacency_lists, weights_lists = get_adjacency_lists_from_A(A)
    print(f"{1.*(A>0).sum(axis=0).min() = }, {1.*(A>0).sum(axis=0).mean() = }, {1.*(A>0).sum(axis=0).max() = }")
    graphs['regular'] = (adjacency_lists, weights_lists)

    # --------------- TESTING matvec_dp function on a random tree (which is exact for a tree) ---------------
    
    adjacency_lists, weights_lists = graphs['tree']
    # GFI methods
    brute_force = BFGFIntegrator(adjacency_lists, weights_lists, n, f_fun)
    spanning_trees = SpanningTreeGFIntegrator(adjacency_lists, weights_lists, vertices, f_fun, num_trees=0)

    # distance matrix match test
    dist_T = spanning_trees._distance_matrix(adjacency_lists, weights_lists)
    assert np.allclose(f_fun(dist_T), brute_force.get_kernel_graph()), \
        print("distance matrices not computed correctly")

    root = np.random.randint(n)
    levels, parents = spanning_trees._tree_root2leaf_levels_parents(root, adjacency_lists)
    # construct a tree data structure
    tree = {'root':root, 'parents':parents, \
            'adj':adjacency_lists,  'w':weights_lists, \
            'levels':levels}
    print(f"Random tree on {n = } with {len(levels) = }")

    # Integration via spanning tree gfi and brute force must match
    for _ in range(num_samples):
        field = np.random.randn(n, k)
        true_x = brute_force.integrate_graph_field(field)
        x = spanning_trees._matvec_dynamic_programming(tree, field)
        assert np.allclose(x, true_x) and get_rel_diff(x, true_x) < 1e-6, \
            print("_matvec_dynamic_programming is not implemented correctly")  
    print("PASSED matvec_dp implementation tests")

    # --------------- TESTING spanning trees ---------------

    for graph_type in graphs.keys():   
        adjacency_lists, weights_lists = graphs[graph_type]
        num_trees = 1
        spanning_trees = SpanningTreeGFIntegrator(adjacency_lists, weights_lists, vertices, f_fun, num_trees)

        for tree in spanning_trees._trees:
            for node in range(spanning_trees.n):
                assert len(tree['adj'][node]) == len(tree['w'][node])
                assert len(tree['adj'][node]) > 0
                for ni_idx, ni in enumerate(tree['adj'][node]):
                    assert node in tree['adj'][ni], print("adjacency lists does not is not complete")
                    node_idx = (tree['adj'][ni]).index(node)
                    assert tree['w'][node][ni_idx] == tree['w'][ni][node_idx]
            
    print("PASSED adjacency and weights lists are consistent")

    # --------------- spanning trees distortion: averaging vs minimum ---------------
    for num_trees in [1, 3]:
        for graph_type in graphs.keys():
            print(f"{graph_type = }, {num_trees = }")    
            adjacency_lists, weights_lists = graphs[graph_type]
            spanning_trees = SpanningTreeGFIntegrator(adjacency_lists, weights_lists, vertices, f_fun, num_trees)
            dist_G = spanning_trees._dist_G

            print("Averaging trees: histogram for suboptimality ratio")
            dist_T = np.zeros((n,n))
            for i in range(spanning_trees._num_trees):
                tadj_lists, tw_lists = spanning_trees._trees[i]['adj'], spanning_trees._trees[i]['w']
                dist_Ti = spanning_trees._distance_matrix(tadj_lists, tw_lists)
                dist_T  += dist_Ti
            dist_T /= spanning_trees._num_trees
            print_subopt_ratios(dist_T, dist_G)

            print("Minimum over trees: histogram for suboptimality ratio")
            min_dist_T = np.inf * np.ones((n,n))
            for i in range(spanning_trees._num_trees):
                tadj_lists, tw_lists = spanning_trees._trees[i]['adj'], spanning_trees._trees[i]['w']
                dist_Ti = spanning_trees._distance_matrix(tadj_lists, tw_lists)
                min_dist_T = np.minimum(min_dist_T, dist_Ti)
            print_subopt_ratios(min_dist_T, dist_G)


if __name__ == '__main__':
    main()