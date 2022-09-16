#from torch import nn

import ega.util.mesh_utils as mu
from ega.algorithms.top_field_modeler import TopFieldModeler

import os 
import time
import numpy as np 
from scipy.sparse.csgraph import shortest_path


class BruteForce(TopFieldModeler):
    
    """
    inputs:
        adjacency_lists: list of lists with the i^th list containing the indices of vertex i's neighbors.
        kernel_function: f: R -> R is a fixed function applied elementwise on top of the similarity matrix 
                         calculated from teh adjacency_lists.
       
    class description:
        Given the adjacency_lists, the constructor computes the shortest path distance matrix (M) explicitly.
        The shortest path is calculated with scipy.sparse.csgraph.shortest_path function: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.shortest_path.html
        We can also specify explicitly the method to calculate shortest path algorithm in the constructor
        
        model_top_field: takes the kernel_function f and graph_field x, and outputs Mx, where M[i,j] = f(dist(i,j))
    """
    
    def __init__(self, adjacency_lists, kernel_function, shortest_path_algorithm='auto'):
  
        self.shortest_path_algorithm = shortest_path_algorithm 
        self.kernel_function = kernel_function

        self.distance_matrix = self.generate_shortest_path_matrix(adjacency_lists)
        
    def generate_shortest_path_matrix(self, adjacency_lists):
        
        sparse_adjacency_matrix = mu.adjacency_list_to_sparse_matrix(adjacency_lists)
        dist_matrix = shortest_path(csgraph=sparse_adjacency_matrix, directed=False, method=self.shortest_path_algorithm)
        
        return dist_matrix

    def model_top_field(self, graph_field):
        
        assert self.distance_matrix.shape[1] == graph_field.shape[0], "Dimention Mismatch Error: the first dimension of graph_field should be the same as the number of vertices in distance_matrix"
        
        # if graph_field has dimension 1 and is of shape (n,), we expand it into 2 dim numpy array with shape (n, 1)
        if graph_field.ndim ==1 and graph_field.shape[0] == self.distance_matrix.shape[1]:
            graph_field = np.expand_dims(graph_field, axis = 1)
        
        Mx = np.matmul(self.kernel_function(self.distance_matrix), graph_field)
        return Mx

        