import time
import numpy as np 
from typing import List

from ega.util.separator_utils import construct_placebo_graph_factorization, construct_graph_factorization, \
                                     integrate_factorized_graph_field, base_compute_clustering
from ega.algorithms.lt_separation_finder import LTSeparationFinder
from ega.algorithms.gf_integrator import GFIntegrator
from ega.algorithms.brute_force import BFGFIntegrator



class PlaceboSeparationGFIntegrator(GFIntegrator):
    """ 
    This class implements the algorithm for integrating graph fields based on balanced separations. 
    The constructor of the objects of this class takes extra arguments (in addition to standard graph description), namely:
        
    separation_finder: 
        an object of the class SeparationFinder responsible for finding balanced separators,
    unit_size: 
        a scalar used to produce the quantized versions of shortest-path-distances according to the formula: 
            quantized_path_length = floor(path_length / unit_size) 
            (i.e. express the shortest-path distances as natural numbers in pre-defined units).
    """
    def __init__(self, adjacency_lists: List[List[int]], weights_lists: List[List[int]], vertices: List[int], f_fun, 
                 separation_finder=LTSeparationFinder(), unit_size=1.0, nb_clusters=2, laplace=True, threshold_nb_vertices=500):
        super().__init__(adjacency_lists, weights_lists, vertices, f_fun)
        self._separation_finder = separation_finder
        self._unit_size = unit_size
        self._laplace = laplace
        self._f_fun = f_fun 
        self._graph_factorization = construct_placebo_graph_factorization(len(vertices), threshold_nb_vertices, nb_clusters)
           
    def integrate_graph_field(self, field: np.ndarray) -> np.ndarray:
        return integrate_factorized_graph_field(field, self._graph_factorization, self._f_fun, self._unit_size, self._laplace)
    
    

class SeparationGFIntegrator(GFIntegrator):
    """ 
    This class implements the algorithm for integrating graph fields based on balanced separations. 
    The constructor of the objects of this class takes extra arguments (in addition to standard graph description), namely:
        
    separation_finder: 
        an object of the class SeparationFinder responsible for finding balanced separators
    compute_clustering: 
        a function to specify the way to compute clusterings
    unit_size: 
        a scalar used to produce the quantized versions of shortest-path-distances according to the formula: 
            quantized_path_length = floor(path_length / unit_size) 
            (i.e. express the shortest-path distances as natural numbers in pre-defined units).
    threshold_nb_vertices: 
        if the number of vertices is smaller than this value, 
        then a brute-force node of the tree encoding graph factorization is constructed
    """
    
    def __init__(self, adjacency_lists: List[List[int]], weights_lists: List[List[int]], vertices: List[int], f_fun,
                 separation_finder=LTSeparationFinder(), compute_clustering=base_compute_clustering, unit_size=1.0, 
                 laplace=True, threshold_nb_vertices=500):
        
        super().__init__(adjacency_lists, weights_lists, vertices, f_fun)
        
        self._separation_finder = separation_finder
        self._compute_clustering = compute_clustering
        self._unit_size = unit_size
        self._laplace = laplace
        vertices = np.arange(len(adjacency_lists))
        
        self._graph_factorization = construct_graph_factorization(self._adjacency_lists, self._weights_lists, self._separation_finder, 
                                                                  self._compute_clustering, self._unit_size, threshold_nb_vertices, 
                                                                  vertices, self._f_fun)

    def integrate_graph_field(self, field):
        return integrate_factorized_graph_field(field, self._graph_factorization, self._f_fun, self._unit_size, self._laplace)

    

