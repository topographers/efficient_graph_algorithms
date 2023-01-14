import numpy as np
from typing import Callable, TypedDict, List, Set, Dict 

from ega.algorithms.trees import TreeGFIntegrator, TreeDict
import random


class BartalTreeDict(TypedDict):
    root:int 
    adj: List[List[int]] 
    w: List[List[float]] 
    node2idx: Dict[int, int]


class BartalTreeGFIntegrator(TreeGFIntegrator):
    """
    The class for fast exact multiplication of the distance-based matrix
    M_ab \in R ^ {N * N} defined as (M_ab)_ij = \sum_Tk exp(a*dist_Tk(i,j)+b), 
    where dist_Tk(i,j) is approximation of the shortest path distance  between
    nodes i and j over the edges of the undirected graph G, 
    with a given tensor field \in R^{N * d_1 * ... * d_l}.

    Specifically, T1, ... , Tk are sampled Bartal trees that approximates the shortest
    path distances in the graph (embed the arbitrary weighted graph metric into 
    the distribution of trees. It is done by using the low diameter randomized
    decomposition, and by building the random hierarchically well-separated
    trees with expected distortion factor O(log^2 N).

    The fast multiplication procedure is based on the dynamic programming for
    computing bottom-up and top-down sums.
    Implementation of matrix vector multiplication using Bartal trees is exact when 
    f_fun(x)=exp(ax), ie, is exponential function.
    """
    def __init__(self, adjacency_lists:List[List[int]], \
                       weights_lists:List[List[float]], \
                       vertices:List[int], f_fun:Callable, num_trees:int):
        super().__init__(adjacency_lists, weights_lists, vertices, f_fun, num_trees)
        
        for i in range(self._num_trees):
            self._trees[i] = self._sample_tree()

    def _sample_tree(self) -> TreeDict:
        # node2idx, idx2node
        cluster = list(range(self.n))
        diam = self._diam
        tree = self._bartal_tree(cluster, diam)
        # permute indices of adjacency list according to node2idx mapping
        root = tree['root']
        tadj_lists = [[] for _ in range(self.n)]
        tw_lists = [[] for _ in range(self.n)]
        for node, node_idx in tree['node2idx'].items():
            tadj_lists[node] = tree['adj'][node_idx]
            tw_lists[node] = tree['w'][node_idx]
        levels, parents = self._tree_root2leaf_levels_parents(root, tadj_lists)
        tree = {'root':root, 'parents':parents, \
                'adj':tadj_lists,  'w':tw_lists, \
                'levels':levels}
        return tree

    def _bartal_tree(self, cluster:List[int], diam:float) -> BartalTreeDict:
        """
        Bartal tree in the dictionary format
        tree = dict()
                'root':root_node, 
                'adj':t_adj_lists
                    mapping of node_index to the list of adjacent  nodes
                'w':t_w_lists
                    mapping of node_index to the list of weights of adjacent nodes
                'node2idx': dict()
                    mapping of nodes to indices in the 'adj' list
        """
        if len(cluster) == 1:
            root = cluster[0]
            tree = {'root':root, 'adj':[[]],  'w':[[]], 'node2idx':{cluster[0]:0}}
            return tree
        new_clusters = self._low_diameter_decomposition(cluster, diam / 2)
        trees = [0]*len(new_clusters)
        for cl_idx, cluster_i in enumerate(new_clusters):
            trees[cl_idx] = self._bartal_tree(cluster_i, diam / 2)
        return self._merge_trees(trees, diam)

    def _merge_trees(self, trees:BartalTreeDict, w:float) -> BartalTreeDict:
        """
        Merge trees by their roots using weight w
        Use node2idx lists to merge adjacency_lists and weights_lists properly
        """
        tree = trees[0]
        root = tree['root']
        root_idx = tree['node2idx'][root]
        count = len(tree['node2idx'])
        for tree_i in trees[1:]:
            tree['adj'] += tree_i['adj']
            tree['w'] += tree_i['w']
            for node, node_idx in tree_i['node2idx'].items():
                tree['node2idx'][node] = node_idx + count
            # connect root with root_i using weight w
            root_i = tree_i['root']
            root_i_idx = tree['node2idx'][root_i]
            tree['adj'][root_idx] += [root_i]
            tree['w'][root_idx] += [w]
            tree['adj'][root_i_idx] += [root]
            tree['w'][root_i_idx] += [w]
            count += len(tree_i['node2idx'])
        return tree 

    def _low_diameter_decomposition(self, cluster:List[int], diam:float) -> List[int]:
        """
        Decompose current cluster into lower diameter clusters
        """
        if len(cluster) == 1: return [cluster]
        p = min(1, (4*np.log2(self.n))  / (diam / self._min_w))
        clusters = []
        unsampled_nodes = set(cluster)
        while len(unsampled_nodes) > 0:
            center = random.sample(unsampled_nodes, 1)[0]
            R = np.random.geometric(p=p) * self._min_w
            # obtain shortest path ball fom center node with radius < R
            # over unsampled nodes only
            new_cluster = self._shortest_path_ball(center, R, unsampled_nodes)
            clusters += [new_cluster]
            unsampled_nodes = unsampled_nodes.difference(new_cluster)
        return clusters
