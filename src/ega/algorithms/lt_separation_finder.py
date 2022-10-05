import numpy as np
from ega.algorithms.separation_finder import SeparationFinder 

from planar_separators.lipton_tarjan import PlanarGraphEdges 
from planar_separators.lipton_tarjan import PlanarGraph 
from planar_separators.lipton_tarjan import Triangulator
from planar_separators.lipton_tarjan import PlanarGraphGenerator 
from planar_separators.lipton_tarjan import PlanarSeparator 
from planar_separators.lipton_tarjan import PlanarGraphConstructor
from planar_separators.lipton_tarjan import SeparationClass



class LTSeparationFinder(SeparationFinder):
    """Class for computing balanced separations using Lipton-Tarjan algorithm.
    """
    def __init__(self):
        pass
    def compute_separation(self, adjacency_lists):
        triangulated_planar_graph = PlanarGraphConstructor.construct_from_ordered_adjacencies(adjacency_lists)
        separation = PlanarSeparator.mark_separation(triangulated_planar_graph)
        separation = [np.int32(e) for e in separation]
        return separation  

