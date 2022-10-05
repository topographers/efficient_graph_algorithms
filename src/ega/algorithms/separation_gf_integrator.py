import time
import numpy as np 
from typing import List

from ega.util.separator_utils import construct_placebo_graph_factorization, integrate_factorized_graph_field
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
    
    



    