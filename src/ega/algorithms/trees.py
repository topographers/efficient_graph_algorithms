import numpy as np 
import abc
from typing import Callable, Optional, TypedDict, Union, List, Set, Tuple
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from ega.algorithms.gf_integrator import GFIntegrator


class TreeDict(TypedDict):
    root:int 
    adj: List[List[int]] 
    w: List[List[float]] 
    parents: List[int]
    levels: List[List[int]]


class TreeGFIntegrator(GFIntegrator):
    """
    The class for fast exact multiplication of the distance-based matrix
    M_ab \in R ^ {N * N} defined as (M_ab)_ij = \sum_Tk exp(a*dist_Tk(i,j)+b), 
    where dist_Tk(i,j) is approximation of the shortest path distance  between
    nodes i and j over the edges of the undirected graph G, 
    with a given tensor field \in R^{N * d_1 * ... * d_l}.

    Specifically, T1, ... , Tk are some random trees that approximates the shortest
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
        super().__init__(adjacency_lists, weights_lists, vertices, f_fun)
        self.n = len(vertices)
        self._num_trees = num_trees
        # construct Bartal trees
        self._trees  = [0]*self._num_trees
        # minimum edge weight
        self._min_w = min([w for w_list in weights_lists for w in w_list])
        self._dist_G = self.distance_matrix(adjacency_lists, weights_lists)
        self._diam = self._dist_G.max() + 1e-6
        
    @abc.abstractmethod
    def _sample_tree(self):
        raise NotImplementedError("Abstract method")

    def integrate_graph_field(self, field:np.ndarray) -> np.ndarray:
        res = 0
        for tree in self._trees:
            res += self._matvec_dynamic_programming(tree, field)
        return res / self._num_trees

    def distance_matrix(self, adjacency_lists:List[List[int]], \
                              weights_lists:List[List[float]]) -> np.ndarray:
        n = len(adjacency_lists)
        edges = np.zeros((n, n))
        for i in range(n):
            for j_idx, j in enumerate(adjacency_lists[i]):
                w = weights_lists[i][j_idx]
                edges[i,j] = w
                edges[j,i] = w
        csr_adjacency = csr_matrix(edges)
        dist_G = shortest_path(csgraph=csr_adjacency, directed=False)
        return dist_G

    def _tree_root2leaf_levels_parents(self, root:int, \
                                             tadj_lists:List[List[int]]) -> Tuple[List[List[int]],List[int]]:
        """
        Given tree adjacency lists and a root, build levels of the tree
        and list of parents for each node
        """
        n = len(tadj_lists)
        parents = [0]*n
        parents[root] = None
        levels = [[root]]
        # top-down traversal of a tree
        previous_level = levels[0]
        while len(previous_level) != 0:
            levels += [[]]
            for node in previous_level:
                for child_node in tadj_lists[node]:
                        if child_node == parents[node]: continue
                        parents[child_node] = node
                        levels[-1] += [child_node]
            previous_level = levels[-1]
        return (levels[:-1], parents)

    def _matvec_dynamic_programming(self, tree:TreeDict, \
                                          field:np.ndarray) -> np.ndarray:
        """
        Compute matrix field product using dynamic programming,
        assuming that
                    f_fun(x) = exp(ax), exponential function
        tree = dict()
                'root':root_node: int, 
                'adj':t_adj_lists: List[List[int]], 
                'w':t_w_lists: List[List[float]], 
                'parents': list with parent nodes: List[int],
                'levels':list with nodes on each level of the tree: List[List[int]]
        """
        # bottom-up traversal
        partial_sums = np.zeros(field.shape)
        for level in reversed(tree['levels']):
            for node in level:
                partial_sums[node] += field[node]
                for idx_child_node, child_node in enumerate(tree['adj'][node]):
                    if child_node == tree['parents'][node]: continue
                    w = tree['w'][node][idx_child_node]
                    partial_sums[node] += self._f_fun(w)*partial_sums[child_node]
        # top-down traversal
        sums = np.zeros(field.shape)
        sums[tree['root']] = partial_sums[tree['root']]
        for level in tree['levels']:
            for node in level:
                for idx_child_node, child_node in enumerate(tree['adj'][node]):
                    if child_node == tree['parents'][node]: continue
                    w = tree['w'][node][idx_child_node]
                    sums[child_node] = self._f_fun(w) * sums[node] + \
                                            (1 - self._f_fun(2*w)) * partial_sums[child_node]
        return sums

    def _shortest_path_ball(self, center:int, R:float, \
                                  unsampled_nodes:Optional[Set[int]]=None, \
                                  list_type:Optional[bool]=True) -> Union[Set[int],List[int]]:
        """
        Return a cluster (connected component) from unsampled_nodes 
        centered at node center with radius < R.
        Specifically, grow a shortest path ball from center node over the 
        unsampled nodes with radius < R.
        """
        cluster = [center]
        visited = np.zeros(self.n)
        candidates = [center] # candidates to be added to a cluster
        if unsampled_nodes is None: 
            func_nj_sampled = lambda nj: False
        else:
            func_nj_sampled = lambda nj: nj not in unsampled_nodes
        while len(candidates) >= 1:
            next_candidates = []
            for ni in candidates:
                if visited[ni] == 1: continue
                visited[ni] = 1
                for nj in self._adjacency_lists[ni]:
                    # check if nj is inside the ball(center, R) & unsampled
                    if func_nj_sampled(nj) or (self._dist_G[center, nj] >= R) or (visited[nj] == 1): 
                        visited[nj] = 1
                        continue
                    cluster += [nj]
                    next_candidates += [nj]
            candidates = next_candidates
        return list(set(cluster)) if list_type else set(cluster)
        