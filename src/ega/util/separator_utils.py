import numpy as np 
import math 
from typing import List, Tuple
import networkx as nx 
from ega.algorithms.brute_force import BFGFIntegrator
from ega.data_structures.graph_factorization import GraphFactorization, NodeContent, Slicer 


def find_indices(list_to_check: List[int], item_to_find: int) -> List[int]:
    """
    this function returns a list of all indices where the values in list_to_check 
    equal to the value of item_to_find.
    """
    indices = []
    for idx, value in enumerate(list_to_check):
        if value == item_to_find:
            indices.append(idx)
    return indices


def apply_dict_map(x: List[int], dictionary: dict):
    """ 
    this function returns the values corresponds to the keys in the list of x
    """    
    return [dictionary[e] for e in x]


def base_compute_clustering(adjacency_lists: List[List[int]], 
                            weights_lists: List[List[int]], 
                            aset_indices: List[int], 
                            bset_indices: List[int], 
                            sources: List[int], 
                            unit_size: float) \
                            ->Tuple[List[List[List[int]]], List[List[List[int]]], List[List[int]]]:
    """ 
    a base function for clustering 
    """
    G = nx.Graph()
    for i in range(len(adjacency_lists)):
        G.add_node(i)
    for i in range(len(weights_lists)):
        for j in range(len(weights_lists[i])):
            G.add_edge(i, adjacency_lists[i][j], weight=weights_lists[i][j])
    dist_dict, _ = nx.multi_source_dijkstra(G, sources)
    dist_indices = [(math.floor(e[1] / unit_size), e[0]) for e in list(dist_dict.items())]
    aset = set(aset_indices)
    bset = set(bset_indices)
    a_max_value = max([e[0] for e in dist_indices if e[1] in aset])
    b_max_value = max([e[0] for e in dist_indices if e[1] in bset])
    a_clustering = [[[] for _ in range(a_max_value + 1)]]
    b_clustering = [[[] for _ in range(b_max_value + 1)]]
    for e in dist_indices:
        if e[1] in aset:
            a_clustering[0][e[0]].append(e[1])
        else:  
            b_clustering[0][e[0]].append(e[1])
    return a_clustering, b_clustering, [[0]]


def construct_placebo_graph_factorization(nb_vertices: int, 
                                          threshold_nb_vertices: int, 
                                          nb_clusters: int) -> GraphFactorization:
    """
    This function calls the recursive function to construct graph factorization
    
    nb_vertices: 
        number of the vertices of the graph,
    threshold_nb_vertices: 
        if the number of vertices is smaller than this value, 
        then a brute-force node of the tree encoding graph factorization is constructed,
    nb_clusters: 
        the upper bound on the number of clusters in each node of the tree 
        encoding graph factorization.
    """
    return recursively_contruct_placebo_graph_factorization(nb_vertices, 
                                                            threshold_nb_vertices, 
                                                            nb_clusters)


def recursively_contruct_placebo_graph_factorization(nb_vertices: int, 
                                                     threshold_nb_vertices: int, 
                                                     nb_clusters: int) -> GraphFactorization:
    """ 
    this is the recursive function called by construct_placebo_graph_factorization
    
    definitions for input variables are the same as in construct_placebo_graph_factorization
    """
    if nb_vertices == 0:
        return None
    gf = GraphFactorization()
    nc = NodeContent()
    if nb_vertices < threshold_nb_vertices:
        adjacency_lists = [np.arange(nb_vertices).tolist() for _ in range(nb_vertices)]
        weights_lists = [np.ones(nb_vertices).tolist() for _ in range(nb_vertices)]
        vertices = np.arange(nb_vertices).tolist()
        f_fun = lambda x : x
        nc._bfgf_integrator = BFGFIntegrator(adjacency_lists, weights_lists, vertices, f_fun)
        gf._data = nc
        gf._left_child = None
        gf._right_child = None
    else:
        slicer_size = int(math.ceil(nb_vertices / 2))
        cluster_size = int(math.ceil(slicer_size / nb_clusters))
        nc._cl_relator = np.ones((nb_clusters, nb_clusters))
        left_slicer = Slicer()
        right_slicer = Slicer()
        clust_instant = np.array([[x] for x in range(cluster_size)])
        left_slicer._clustering = [clust_instant for _ in range(nb_clusters)]
        right_slicer._clustering = [clust_instant for _ in range(nb_clusters)]
        nc._left_slicer = left_slicer
        nc._right_slicer = right_slicer 
        gf._data = nc
        gf._left_child = recursively_contruct_placebo_graph_factorization(
            int(nb_vertices / 2), threshold_nb_vertices, nb_clusters)
        gf._right_child = recursively_contruct_placebo_graph_factorization(
            int(nb_vertices / 2), threshold_nb_vertices, nb_clusters)
    return gf


def construct_graph_factorization(adjacency_lists: List[List[int]], 
                                  weights_lists: List[List[int]], 
                                  separation_finder, 
                                  compute_clustering, 
                                  unit_size: float, 
                                  threshold_nb_vertices: int, 
                                  vertices: List[int], 
                                  f_fun):
    """
    This function calls the recursive function to construct graph factorization
    
    compute_clustering: 
        a function to specify the way to compute clusterings
    separation_finder: 
        it is an object of the class SeparationFinder responsible for finding balanced separators
    threshold_nb_vertices: 
        if the number of vertices is smaller than this value, 
        then a brute-force node of the tree encoding graph factorization is constructed
    unit_size: 
        a scalar used to produce the quantized versions of shortest-path-distances 
        according to the formula: 
            quantized_path_length = floor(path_length / unit_size) 
            (i.e. express the shortest-path distances as natural numbers in pre-defined units).
    """
    return recursively_construct_graph_factorization(adjacency_lists, 
                                                     weights_lists, 
                                                     separation_finder, 
                                                     compute_clustering,
                                                     unit_size, 
                                                     threshold_nb_vertices, 
                                                     vertices, 
                                                     f_fun)


def recursively_construct_graph_factorization(adjacency_lists: List[List[int]], 
                                              weights_lists: List[List[int]], 
                                              separation_finder, 
                                              compute_clustering, 
                                              unit_size: float, 
                                              threshold_nb_vertices: int, 
                                              vertices: List[int], 
                                              f_fun):
    """
    a recursive function for graph factorization called by construct_graph_factorization
    
    definitions for input variables are the same as in construct_graph_factorization
    """
    nb_vertices = len(adjacency_lists)
    
    if nb_vertices == 0:
        return None
    
    gf = GraphFactorization()
    nc = NodeContent()
    
    if nb_vertices < threshold_nb_vertices:
        nc._bfgf_integrator = BFGFIntegrator(adjacency_lists, weights_lists, vertices, f_fun)
        gf._data = nc
        gf._left_child = None
        gf._right_child = None
        
    else:
        separation = separation_finder.compute_separation(adjacency_lists)
        sepset_indices = find_indices(separation, 2)
        sepsize = len(sepset_indices)
        aset_indices = find_indices(separation, 0) + sepset_indices[0: int(sepsize / 2)]
        bset_indices = find_indices(separation, 1) + sepset_indices[int(sepsize / 2):]
        a_clustering, b_clustering, cl_relator = compute_clustering(adjacency_lists, 
                                                                    weights_lists, 
                                                                    aset_indices, 
                                                                    bset_indices, 
                                                                    sepset_indices, 
                                                                    unit_size)
                                  
        adict = dict(zip(aset_indices, np.arange(len(aset_indices))))
        bdict = dict(zip(bset_indices, np.arange(len(bset_indices))))
        a_adjacency_lists = [adjacency_lists[i] for i in aset_indices]
        b_adjacency_lists = [adjacency_lists[i] for i in bset_indices]
        a_weights_lists = [weights_lists[i] for i in aset_indices]
        b_weights_lists = [weights_lists[i] for i in bset_indices]
        a_weights_lists = [[e for index1, e in enumerate(x) 
                            if a_adjacency_lists[index2][index1] in aset_indices] 
                           for index2, x in enumerate(a_weights_lists)] 
        b_weights_lists = [[e for index1, e in enumerate(x) 
                            if b_adjacency_lists[index2][index1] in bset_indices] 
                           for index2, x in enumerate(b_weights_lists)] 
        a_adjacency_lists = [[e for e in x if e in aset_indices] for x in a_adjacency_lists]
        b_adjacency_lists = [[e for e in x if e in bset_indices] for x in b_adjacency_lists] 
        a_adjacency_lists = [apply_dict_map(x, adict) for x in a_adjacency_lists]
        b_adjacency_lists = [apply_dict_map(x, bdict) for x in b_adjacency_lists] 

        nc._cl_relator = cl_relator
        a_slicer = Slicer()
        b_slicer = Slicer()
        a_slicer._clustering = a_clustering
        b_slicer._clustering = b_clustering
        nc._left_slicer = a_slicer
        nc._right_slicer = b_slicer 
        gf._data = nc
        
        gf._left_child = recursively_construct_graph_factorization(a_adjacency_lists, 
                                                                   a_weights_lists, 
                                                                   separation_finder, 
                                                                   compute_clustering, 
                                                                   unit_size, 
                                                                   threshold_nb_vertices, 
                                                                   vertices[aset_indices], 
                                                                   f_fun)
        gf._right_child = recursively_construct_graph_factorization(b_adjacency_lists, 
                                                                    b_weights_lists, 
                                                                    separation_finder, 
                                                                    compute_clustering, 
                                                                    unit_size, 
                                                                    threshold_nb_vertices, 
                                                                    vertices[bset_indices], 
                                                                    f_fun)
    return gf


def fast_multiply(hankel_row_column: np.ndarray, 
                  input_tensor: np.ndarray, 
                  multiplicative_shift=False) -> np.ndarray:
    """ 
    Computes H * input_tensor, 
    where: 
        input_tensor is the tensor of shape: 
            N x k1 x k2 x ..., 
        H is a Hankel matrix of shape M x N  and furthermore, 
            the first row of H given as: hankel_row_column[0:N] 
            and the last column of H is given as: hankel_row_column[N-1:].
    
    The output tensor is of shape M x k1 x k2 x ...
    
    If the multiplicave_shift is True, then furthermore the (i+1)th row of H
    is obtained from the ith row of H by multiplying with a fixed multiplier
    for i = 0,...,M-2.  
    """
    assert len(hankel_row_column) == 1 or multiplicative_shift, \
            "Either len(hankle_row_column) should be 1, or multiplicative_shift should be True"
        
    if len(hankel_row_column) == 1:
        return hankel_row_column[0] * input_tensor
    if multiplicative_shift:
        def func1d(x):
            first_entry = np.dot(hankel_row_column[:len(x)], x)
            multipliers = np.power(hankel_row_column[1] / hankel_row_column[0], 
                                   np.arange(len(hankel_row_column) - len(x) + 1))
            return first_entry * multipliers
        return np.apply_along_axis(func1d, 0, input_tensor)


def integrate_factorized_graph_field(field: np.ndarray, 
                                     graph_factorization: GraphFactorization, 
                                     f_fun, 
                                     unit_size: float, 
                                     laplace: bool) -> np.ndarray:
    """ 
    this function conducts integrate_graph_field with the data from graph_factorization. 
    """
    result = np.zeros(shape=field.shape)
    recursively_integrate_factorized_graph_field(field, 
                                                 graph_factorization, 
                                                 f_fun, 
                                                 unit_size, 
                                                 laplace, 
                                                 result)
    return result 


def recursively_integrate_factorized_graph_field(field: np.ndarray, 
                                                 gf: GraphFactorization, 
                                                 f_fun, 
                                                 unit_size: float, 
                                                 laplace:bool, 
                                                 result: np.ndarray):
    """ 
    this recursive function is called by integrate_factorized_graph_field to do graph field integration.
    """
    def sum_field_slices(e):
        return np.sum(field[e], axis=0)
    if gf is None: # empty node, nothing needs to be done
        return
    # Leaf node with small set of vertices attached - brute force calculation.   
    if gf._data._bfgf_integrator is not None: 
        v_indices = gf._data._bfgf_integrator.vertices()
        local_field = field[v_indices]  
        local_int_result = gf._data._bfgf_integrator.integrate_graph_field(local_field)
        result[v_indices] += local_int_result
        return
    # Calculating cross-terms.
    for cl_ind_l in range(len(gf._data._left_slicer._clustering)):
        for cl_ind_r in range(len(gf._data._right_slicer._clustering)):
            accum_tensors_array_l = np.array([sum_field_slices(e) 
                                              for e in gf._data._left_slicer._clustering[cl_ind_l]])
            accum_tensors_array_r = np.array([sum_field_slices(e) 
                                              for e in gf._data._right_slicer._clustering[cl_ind_r]])
            
            nb_slices_l = len(gf._data._left_slicer._clustering[cl_ind_l])
            nb_slices_r = len(gf._data._right_slicer._clustering[cl_ind_r])
            shift_term = int(gf._data._cl_relator[cl_ind_l][cl_ind_r])
            end_term = shift_term + nb_slices_l + nb_slices_r - 1
            hankel_row_column = f_fun(np.arange(shift_term, end_term, step=1) * unit_size)
            
            cross_contrib_l = fast_multiply(hankel_row_column, accum_tensors_array_l, laplace)
            cross_contrib_r = fast_multiply(hankel_row_column, accum_tensors_array_r, laplace)
         
            for index in range(len(gf._data._right_slicer._clustering[cl_ind_r])):
                e = gf._data._right_slicer._clustering[cl_ind_r][index]
                result[e] += np.repeat(np.expand_dims(cross_contrib_l[index], axis=0), len(e), axis=0)
            
            for index in range(len(gf._data._left_slicer._clustering[cl_ind_l])):
                e = gf._data._left_slicer._clustering[cl_ind_l][index]
                result[e] += np.repeat(np.expand_dims(cross_contrib_r[index], axis=0), len(e), axis=0)

    recursively_integrate_factorized_graph_field(field, gf._left_child, f_fun, unit_size, laplace, result)
    recursively_integrate_factorized_graph_field(field, gf._right_child, f_fun, unit_size, laplace, result)
  
