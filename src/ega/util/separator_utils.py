import numpy as np 
import math 
from typing import List

from ega.algorithms.brute_force import BFGFIntegrator
from ega.data_structures.graph_factorization import GraphFactorization, NodeContent, Slicer 



def find_indices(list_to_check: List[int], item_to_find: int) -> List[int]:
    """
    this function returns a list of all indices where the values in list_to_check equal to the value of item_to_find.
    """
    indices = []
    for idx, value in enumerate(list_to_check):
        if value == item_to_find:
            indices.append(idx)
    return indices



def construct_placebo_graph_factorization(nb_vertices: int, threshold_nb_vertices: int, nb_clusters: int) -> GraphFactorization:
    """
    This function calls the recursive function to construct graph factorization
    
    nb_vertices: 
        number of the vertices of the graph,
    threshold_nb_vertices: 
        if the number of vertices is smaller than this value, 
        then a brute-force node of the tree encoding graph factorization is constructed,
    nb_clusters: 
        the upper bound on the number of clusters in each node of the tree encoding graph factorization.
    """
    return recursively_contruct_placebo_graph_factorization(nb_vertices, threshold_nb_vertices, nb_clusters)



def recursively_contruct_placebo_graph_factorization(nb_vertices: int, threshold_nb_vertices: int, nb_clusters: int) -> GraphFactorization:
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




def fast_multiply(hankel_row_column: np.ndarray, input_tensor: np.ndarray, multiplicative_shift=False) -> np.ndarray:
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
    if len(hankel_row_column) == 1:
        return hankel_row_column[0] * input_tensor
    if multiplicative_shift:
        def func1d(x):
            first_entry = np.dot(hankel_row_column[:len(x)], x)
            multipliers = np.power(hankel_row_column[1] / hankel_row_column[0], 
                                   np.arange(len(hankel_row_column) - len(x) + 1))
            return first_entry * multipliers
        return np.apply_along_axis(func1d, 0, input_tensor)




def integrate_factorized_graph_field(field: np.ndarray, graph_factorization: GraphFactorization, f_fun, 
                                     unit_size: float, laplace: bool) -> np.ndarray:
    """ 
    this function conducts integrate_graph_field with the data from graph_factorization. 
    """
    result = np.zeros(shape=field.shape)
    recursively_integrate_factorized_graph_field(field, graph_factorization, f_fun, unit_size, laplace, result)
    return result 


def recursively_integrate_factorized_graph_field(field: np.ndarray, gf: GraphFactorization, f_fun, 
                                                 unit_size: float, laplace:bool, result: np.ndarray):
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
            
            accum_tensors_array_l = np.apply_along_axis(sum_field_slices, 1, gf._data._left_slicer._clustering[cl_ind_l])
            accum_tensors_array_r = np.apply_along_axis(sum_field_slices, 1, gf._data._right_slicer._clustering[cl_ind_r])
            
            nb_slices_l = len(gf._data._left_slicer._clustering[cl_ind_l])
            nb_slices_r = len(gf._data._right_slicer._clustering[cl_ind_r])
            shift_term = int(gf._data._cl_relator[cl_ind_l][cl_ind_r])
            end_term = shift_term + nb_slices_l + nb_slices_r - 1
            hankel_row_column = f_fun(np.arange(shift_term, end_term, step=1))
            
            cross_contrib_l = fast_multiply(hankel_row_column, accum_tensors_array_l, laplace)
            cross_contrib_r = fast_multiply(hankel_row_column, accum_tensors_array_r, laplace)
         
            for index in range(len(gf._data._right_slicer._clustering[cl_ind_r])):
                e = gf._data._right_slicer._clustering[cl_ind_r][index]
                # result[e] += np.array([cross_contrib_l[index] for _ in range(len(e))])
                result[e] += np.repeat(np.expand_dims(cross_contrib_l[index], axis=0), len(e), axis=0)
            
            for index in range(len(gf._data._left_slicer._clustering[cl_ind_l])):
              e = gf._data._left_slicer._clustering[cl_ind_l][index]
              # result[e] += np.array([cross_contrib_r[index] for _ in range(len(e))])
              result[e] += np.repeat(np.expand_dims(cross_contrib_r[index], axis=0), len(e), axis=0)

    recursively_integrate_factorized_graph_field(field, gf._left_child, f_fun, unit_size, laplace, result)
    recursively_integrate_factorized_graph_field(field, gf._right_child, f_fun, unit_size, laplace, result)
  
  
