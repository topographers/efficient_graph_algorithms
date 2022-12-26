import numpy as np 

from ega.algorithms.trees import TreeGFIntegrator
import copy



class FRTTreeGFIntegrator(TreeGFIntegrator):
    """
    The class for fast exact multiplication of the distance-based matrix
    M_ab \in R ^ {N * N} defined as (M_ab)_ij = \sum_Tk exp(a*dist_Tk(i,j)+b), 
    where dist_Tk(i,j) is approximation of the shortest path distance  between
    nodes i and j over the edges of the undirected graph G, 
    with a given tensor field \in R^{N * d_1 * ... * d_l}.

    Specifically, T1, ... , Tk are sampled FRT trees that approximates the shortest
    path distances in the graph (embed the arbitrary weighted graph metric into 
    the distribution of trees. It is done by using the low diameter randomized
    decomposition, and by converting a laminar family into a tre with expected 
    distortion factor O(log N).

    The fast multiplication procedure is based on the dynamic programming for
    computing bottom-up and top-down sums.
    Implementation of matrix vector multiplication using Bartal trees is exact when 
    f_fun(x)=exp(ax), ie, is exponential function.
    """
    def __init__(self, adjacency_lists, weights_lists, vertices, f_fun, num_trees=None):
        super().__init__(adjacency_lists, weights_lists, vertices, f_fun, num_trees)
        
        self.extra_n = [0]*self._num_trees
        for i in range(self._num_trees):
            self._trees[i] = self._sample_tree()
            self.extra_n[i] = len(self._trees[i]['adj']) - self.n


    def _sample_tree(self):
        # node2idx, idx2node
        cluster = set(range(self.n))
        return self._frt_tree(cluster)

    
    def integrate_graph_field(self, field):
        res = 0
        for idx, tree in enumerate(self._trees):
            field_i = np.concatenate([field, np.zeros([self.extra_n[idx]]+list(field.shape[1:]))], axis = 0)
            res += self._matvec_dynamic_programming(tree, field_i)[:self.n]
        return res / self._num_trees

    
    def _frt_tree(self, cluster0):
        """
        FRT tree in the dictionary format
        tree = dict()
                'root':root_node, 
                'adj':t_adj_lists, 
                'w':t_w_lists, 
                'parents': list with parent nodes,
                'levels':list with nodes on each level of the tree
        """
        delta = np.ceil(np.log2(self._diam / self._min_w))
        max_cluster_size = self.n

        root = self.n
        tadj_lists, tw_lists = [[] for _ in range(self.n+1)], [[] for _ in range(self.n+1)]
        levels = [[self.n]]
        parents = [0] * (self.n+1)
        parents[self.n] = None
        node_size = self.n + 1
        leaf_clusters = [cluster0] # clusters on the current leaf level
        leaf_node_clusters = [self.n] # node numbers of leaf clusters
        pi = np.random.permutation(self.n)
        beta = np.random.uniform(0.5, 1)
        i = delta # diam(V)/2 = 2**i * w_min
        while i >= 0 and max_cluster_size >= 2:
            R = beta * 2**i * self._min_w
            next_leaf_clusters, next_leaf_node_clusters = [], []
            levels += [[]]
            for node in range(self.n):
                # partition a cluster onto new clusters according to a 
                # permutation pi indicating priority of nodes 
                for cl_idx, cluster in enumerate(leaf_clusters):
                    if len(cluster) == 0: continue
                    ball = self._shortest_path_ball(pi[node], R, list_type=False)
                    new_cluster = ball.intersection(cluster)
                    if len(new_cluster) == 0: continue
                    # add new_cluster node to a tree with node number (node_size-1)
                    cluster_node = leaf_node_clusters[cl_idx] # parent node number
                    if len(new_cluster) == 1:
                        new_cluster_node = list(new_cluster)[0]
                    else:
                        # create a new node for a cluster
                        new_cluster_node = node_size
                        node_size += 1
                        parents += [0]
                        tadj_lists += [[]]
                        tw_lists += [[]]
                    w = self._min_w * 2**i
                    self._add_node2tree(w, new_cluster_node, cluster_node, parents, levels, \
                                                                tadj_lists, tw_lists)
                    # remove assigned vertices from cluster
                    cluster -= new_cluster
                    if len(new_cluster) >= 2:
                        next_leaf_clusters += [new_cluster]
                        next_leaf_node_clusters += [new_cluster_node]
            i -= 1
            if next_leaf_clusters == []: break
            leaf_clusters = next_leaf_clusters
            leaf_node_clusters = next_leaf_node_clusters
            max_cluster_size = max([len(cl) for cl in next_leaf_clusters])
            
        # add nodes from the graph to the leaves
        for cl_idx, cluster in enumerate(leaf_clusters):
            if len(cluster) == 0: continue
            new_cluster_node = list(new_cluster)[0]
            cluster_node = leaf_node_clusters[cl_idx]
            w = self._min_w * 2**max(0,i)
            self._add_node2tree(w, new_cluster_node, cluster_node, parents, levels, tadj_lists, tw_lists)
        return {'root':root, 'parents':parents, 'adj':tadj_lists,  'w':tw_lists, 'levels':levels}


    def _add_node2tree(self, w, new_cluster_node, cluster_node, parents, levels, tadj_lists, tw_lists):
        """
        Add an edge between parent node cluster_node to a child node new_cluster_node, 
        modifying input lists.
        """
        parents[new_cluster_node] = cluster_node
        levels[-1] += [new_cluster_node]
        tadj_lists[new_cluster_node] += [cluster_node]
        tadj_lists[cluster_node] += [new_cluster_node]
        tw_lists[new_cluster_node] += [w]
        tw_lists[cluster_node] += [w]