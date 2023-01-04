import numpy as np 
import networkx as  nx
import random
from typing import Callable

from ega.algorithms.trees import TreeGFIntegrator, TreeDict
import random


class SpanningTreeGFIntegrator(TreeGFIntegrator):
    """
    The class for fast exact multiplication of the distance-based matrix
    M_ab \in R ^ {N * N} defined as (M_ab)_ij = \sum_Tk exp(a*dist_Tk(i,j)+b), 
    where dist_Tk(i,j) is approximation of the shortest path distance  between
    nodes i and j over the edges of the undirected graph G, 
    with a given tensor field \in R^{N * d_1 * ... * d_l}.

    Specifically, T1, ... , Tk are random spanning trees that approximates the shortest
    path distances in the graph (embed the arbitrary weighted graph metric into 
    the distribution of trees. 

    The fast multiplication procedure is based on the dynamic programming for
    computing bottom-up and top-down sums.
    Implementation of matrix vector multiplication using trees is exact when 
    f_fun(x)=exp(ax), ie, is exponential function.
    """
    def __init__(self, adjacency_lists:list[list[int]], \
                       weights_lists:list[list[float]], \
                       vertices:list, f_fun:Callable, num_trees:int):
        super().__init__(adjacency_lists, weights_lists, vertices, f_fun, num_trees)
        
        A = self._numpy_adjacency_matrix()
        self._G = nx.from_numpy_matrix(A)

        for i in range(self._num_trees):
            self._trees[i] = self._sample_tree()
    
    def _sample_tree(self) -> TreeDict:
        """
        Sample a random spanning tree
        """
        if self._num_trees == 1:
            T = nx.minimum_spanning_tree(self._G)
        else:
            # get random spanning tree
            T = nx.random_spanning_tree(self._G)
        A_T = self._adjacency_lists_nx(T)
        tadj_lists, tw_lists = self._get_adjacency_lists_from_A(A_T)
        # convert it into our data structure format
        root = random.choice(np.arange(self.n))
        levels, parents = self._tree_root2leaf_levels_parents(root, tadj_lists)
        tree = {'root':root, 'parents':parents, \
                'adj':tadj_lists,  'w':tw_lists, \
                'levels':levels}
        return tree

    def _numpy_adjacency_matrix(self) -> np.ndarray:
        A = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j_idx, j in enumerate(self._adjacency_lists[i]):
                w = self._weights_lists[i][j_idx]
                A[i,j] = w
                A[j,i] = w
        return A

    def _adjacency_lists_nx(self, G:nx.Graph) -> np.ndarray:
        A_srs = nx.to_scipy_sparse_array(G)
        A = np.array(A_srs.todense())
        return A

    def _get_adjacency_lists_from_A(self, A:np.ndarray) -> tuple[list[list[int]],list[list[float]]]:
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
        return (adjacency_lists, weights_lists)
        