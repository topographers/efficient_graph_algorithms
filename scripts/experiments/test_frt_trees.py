import numpy as np
import networkx as nx

from ega.algorithms.frt_trees import FRTTreeGFIntegrator
from ega.util.graphs_networkx_utils import get_adjacency_nx, get_adjacency_lists_from_A, \
                                            print_subopt_ratios, get_rel_diff


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
    # Geometric random graph
    print("Geometric random graph")
    positions = np.random.rand(n, 3)
    r = 0.35
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1, n):
            w  = np.linalg.norm(positions[i] - positions[j])
            if w <= r:
                A[i,j] = w; A[j,i] = w
    adjacency_lists, weights_lists = get_adjacency_lists_from_A(A)
    print(f"{1.*(A>0).sum(axis=0).min() = }, {1.*(A>0).sum(axis=0).mean() = }, {1.*(A>0).sum(axis=0).max() = }")
    assert (A>0).sum(axis=0).min() > 0, print('disconnected graph')
    graphs['geom'] = (adjacency_lists, weights_lists)
    # Regular graph
    print("Regular random graph")
    G = nx.random_regular_graph(5, n)
    A = get_adjacency_nx(G)
    adjacency_lists, weights_lists = get_adjacency_lists_from_A(A)
    print(f"{1.*(A>0).sum(axis=0).min() = }, {1.*(A>0).sum(axis=0).mean() = }, {1.*(A>0).sum(axis=0).max() = }")
    graphs['regular'] = (adjacency_lists, weights_lists)

    # --------------- TESTING matvec_dp function on FRT tree ---------------
    nodes = np.arange(n)
    for graph_type in graphs.keys():  
        print(f"{graph_type = }") 
        adjacency_lists, weights_lists = graphs[graph_type]
        num_trees = 1
        frt_trees = FRTTreeGFIntegrator(adjacency_lists, weights_lists, vertices, f_fun, num_trees)
        
        for i in range(frt_trees._num_trees):
            tree = frt_trees._trees[i]
            tadj_lists, tw_lists = tree['adj'], tree['w']
            levels = tree['levels']
            print(f"FRT tree on {n = } (all {len(tree['adj']) = }) with {len(levels) = }")
            # distance matrix on graph nodes
            dist_T = frt_trees.distance_matrix(tadj_lists, tw_lists)[nodes, :][:,nodes]
            M = f_fun(dist_T)

            nodes_levels = [node for lev in tree['levels'] for node in lev]
            all_n = len(tree['adj'])
            assert len(nodes_levels) == all_n and set(nodes_levels) == set(range(all_n)) 

            # Integration via frt tree gfi and brute force must match
            for _ in range(num_samples):
                field = np.random.randn(n, k)
                true_x = M @ field
                x = frt_trees.integrate_graph_field(field)
                assert np.allclose(x, true_x) and get_rel_diff(x, true_x) < 1e-6, \
                    print("_frt_tree is not implemented correctly")  

            # --------------- TESTING FRT trees ---------------

            for node in range(frt_trees.n):
                assert len(tree['adj'][node]) == len(tree['w'][node])
                assert len(tree['adj'][node]) > 0
                for ni_idx, ni in enumerate(tree['adj'][node]):
                    assert node in tree['adj'][ni], print("adjacency lists does not is not complete")
                    node_idx = (tree['adj'][ni]).index(node)
                    assert tree['w'][node][ni_idx] == tree['w'][ni][node_idx]

            # each node appears once in the levels
            nodes_levels = [node for lev in tree['levels'] for node in lev]
            all_n = frt_trees.n + frt_trees.extra_n[i]
            assert len(nodes_levels) == all_n and set(nodes_levels) == set(range(all_n))

    print("PASSED _frt_tree implementation tests")     
    print("PASSED adjacency and weights lists are consistent; levels are correct")

    # --------------- FRT trees distortion: averaging vs minimum ---------------

    for graph_type in graphs.keys():
        print(f"{graph_type = }")    
        adjacency_lists, weights_lists = graphs[graph_type]
        num_trees = 10
        frt_trees = FRTTreeGFIntegrator(adjacency_lists, weights_lists, vertices, f_fun, num_trees)
        dist_G = frt_trees._dist_G

        print("Averaging trees: histogram for suboptimality ratio")
        dist_T = np.zeros((n,n))
        for i in range(frt_trees._num_trees):
            tadj_lists, tw_lists = frt_trees._trees[i]['adj'], frt_trees._trees[i]['w']
            dist_Ti = frt_trees.distance_matrix(tadj_lists, tw_lists)[nodes, :][:,nodes]
            dist_T  += dist_Ti
        dist_T /= frt_trees._num_trees
        print_subopt_ratios(dist_T, dist_G)

        print("Minimum over trees: histogram for suboptimality ratio")
        min_dist_T = np.inf * np.ones((n,n))
        for i in range(frt_trees._num_trees):
            tadj_lists, tw_lists = frt_trees._trees[i]['adj'], frt_trees._trees[i]['w']
            dist_Ti = frt_trees.distance_matrix(tadj_lists, tw_lists)[nodes, :][:,nodes]
            min_dist_T = np.minimum(min_dist_T, dist_Ti)
        print_subopt_ratios(min_dist_T, dist_G)


if __name__ == '__main__':
    main()