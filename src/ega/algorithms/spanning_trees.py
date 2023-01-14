import numpy as np 
import networkx as  nx
import random
from typing import Callable, List,Tuple

from ega.algorithms.trees import TreeGFIntegrator, TreeDict
from ega.util.graphs_networkx_utils import numpy_adjacency_matrix
import random


class DisjointSetADT:
    """
    Class supporting fast union/find operations for MST implementation
    https://courses.cs.washington.edu/courses/cse332/16sp/lectures/Lecture25/25_ho.pdf
    """
    def __init__(self, n):
        self.size = np.ones(n)
        self.up = np.arange(n)
        
    def union(self, i:int, j:int):
        """
        Merge two sets to create their union
        by attaching smaller cluster to bigger one
        Assuming that find(i) and find(j) has been called previously
        """
        root_i = self.up[i]
        root_j = self.up[j]
        if root_i == root_j: return
        # assert root_i == self.up[root_i] and root_j == self.up[root_j]
        if self.size[root_i] < self.size[root_j]:
            # attach cluster of x to cluster of y
            self.up[root_i] = root_j
            self.size[root_j] += self.size[root_i] 
        else:
            # attach cluster of y to cluster of x
            self.up[root_j] = root_i
            self.size[root_i] += self.size[root_j]  
        
    def find(self, i:int) -> int:
        """
        Follow i up the tree to return the root of its cluster
        root_i
        """
        if self.up[i] == i: return i
        # path compression: update the tree during search
        self.up[i] = self.find(self.up[i])
        return self.up[i]


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
    def __init__(self, adjacency_lists:List[List[int]], \
                       weights_lists:List[List[float]], \
                       vertices:List[int], f_fun:Callable, num_trees:int):
        super().__init__(adjacency_lists, weights_lists, vertices, f_fun, num_trees)
        
        if num_trees > 1:
            A = numpy_adjacency_matrix(adjacency_lists, weights_lists)
            self._G = nx.from_numpy_matrix(A)

        for i in range(self._num_trees):
            self._trees[i] = self._sample_tree()
    
    def _sample_tree(self) -> TreeDict:
        """
        Sample a random spanning tree
        """
        if self._num_trees == 1:
            tadj_lists, tw_lists = self.minimum_spanning_tree()
        else:
            # get random spanning tree
            T = nx.random_spanning_tree(self._G)
            A_T = self._adjacency_lists_nx(T)
            tadj_lists, tw_lists = self._get_adjacency_lists_from_A(A_T)
        # convert it into our data structure format
        root = random.choice(np.arange(self.n))
        levels, parents = self._tree_root2leaf_levels_parents(root, tadj_lists)
        tree = {'root':root, 'parents':parents, \
                'adj':tadj_lists, 'w':tw_lists, \
                'levels':levels}
        return tree

    def minimum_spanning_tree(self) -> Tuple[List[List[int]],List[List[float]]]:
        """
        Kruskal's algorithm for constructing MST 
        Using Disjoint Set that supports fast union/find methods
        The runtime is O(E log E  +E log V) = O(E log V) 
        """
        wij_list = []
        for i in range(self.n):
            for j_idx, j in enumerate(self._adjacency_lists[i]):
                if j<i: continue
                wij_list += [(self._weights_lists[i][j_idx], i, j)]
        # sort edges in increasing order of weights
        wij_list.sort(key=lambda x: x[0])
        dset = DisjointSetADT(self.n)
        # construct MST using Disjoint Set ADT
        tadj_lists, tw_lists = [[] for _ in range(self.n)], [[] for _ in range(self.n)]
        for w, i, j in wij_list:
            root_i = dset.find(i)
            root_j = dset.find(j)
            if root_i != root_j:
                # clusters are disjoint, add edge to MST
                dset.union(i,j)
                tadj_lists[i] += [j]
                tadj_lists[j] += [i]
                tw_lists[i] += [w]
                tw_lists[j] += [w]
        return (tadj_lists,tw_lists)
    
    def _adjacency_lists_nx(self, G:nx.Graph) -> np.ndarray:
        A_srs = nx.to_scipy_sparse_array(G)
        A = np.array(A_srs.todense())
        return A

    def _get_adjacency_lists_from_A(self, A:np.ndarray) -> Tuple[List[List[int]],List[List[float]]]:
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
        