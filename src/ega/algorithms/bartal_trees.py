import numpy as np 
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from ega.algorithms.gf_integrator import GFIntegrator
from tqdm import tqdm 
import random


class BartalTreeGFIntegrator(GFIntegrator):
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
    def __init__(self, adjacency_lists, weights_lists, vertices, f_fun, num_trees=None):
        super().__init__(adjacency_lists, weights_lists, vertices, f_fun)
        self.n = len(vertices)
        if  num_trees is None: self._num_trees = int(np.log2(self.n)) + 1
        else: self._num_trees = num_trees
        # construct Bartal trees
        self._trees  = []
        # minimum edge weight
        self._min_w = min([w for w_list in weights_lists for w in w_list])
        self._dist_G = self._distance_matrix(adjacency_lists, weights_lists)
        self._diam = self._dist_G.max()
        
        for i in range(num_trees):
            self._trees[i] = self._sample_Bartal_tree()
        

    def integrate_graph_field(self, field):
        res = 0
        for tree in self._trees:
            res += self._matvec_dynamic_programming(tree, field)
        return res / self._num_trees


    def _sample_Bartal_tree(self):
        # node2idx, idx2node
        pass


    def _low_diameter_decomposition(self, cluster, diam):
        if len(cluster) == 1: return [cluster]
        p = min(1, (4*np.log2(self.n))  / (diam / self._min_w))
        clusters = []
        unsampled_nodes = set(cluster)
        while len(unsampled_nodes) > 0:
            center = random.sample(unsampled_nodes)
            R = np.random.geometric(p=p) * self._min_w
            # obtain shortest path ball fom center node with radius <= R
            # over unsampled nodes only
            new_cluster = self._shortest_path_ball(center, R, unsampled_nodes)
            clusters += [new_cluster]
            unsampled_nodes = unsampled_nodes.difference(new_cluster)
        return clusters


    def _distance_matrix(self, adjacency_lists, weights_lists):
        n = len(adjacency_lists)
        edges = np.zeros((n, n))
        for i in range(n):
            for j_idx, j in enumerate(adjacency_lists[i]):
                w = weights_lists[i][j_idx]
                edges[i,j] = w
                edges[j,i] = w
        csr_adjacency = csr_matrix(edges)
        dist_G = shortest_path(csgraph=csr_adjacency, directed=False)
        return  dist_G


    def _tree_root2leaf_levels_parents(self, root, tadj_lists):
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
        while previous_level != []:
            levels += [[]]
            for node in previous_level:
                for child_node in tadj_lists[node]:
                        if child_node == parents[node]: continue
                        parents[child_node] = node
                        levels[-1] += [child_node]
            previous_level = levels[-1]
        # if debug:
        #     assert set(range(n)) == set([node for level in levels for node in level])
        return levels[:-1], parents


    def _matvec_dynamic_programming(self, tree, field):
        """
        Compute matrix field product using dynamic programming,
        assuming that
                    f_fun(x) = exp(ax), exponential function
        tree = dict()
                'root':root_node, 
                'adj':t_adj_lists, 
                'w':t_w_lists, 
                'parents': list with parent nodes,
                'levels':list with nodes on each level of the tree
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