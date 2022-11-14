from typing import List
import numpy as np
from ega.algorithms.gf_integrator import GFIntegrator


class Interpolator:
    """
    given an integrator (for example, brute force), as well as the position of points to be interpolated, 
    this function will do interpolation for these points based on other known points on the mesh graph data 
    """

    def __init__(self, integrator: GFIntegrator, vertices_known: List[int], vertices_interpolate: List[int]):
        """ 
        integrator: the integrator to be used. For example: brute_force integrator 
        vertices_known: a list of integers representing the vertices known on the mesh graph, 
                        which are used to predict the fields of the vertices to be interpolated 
        vertices_interpolate: a list of intergers representing the vertices with unknown fields to be interpolated 
        """
        self.integrator = integrator
        self.vertices_known = vertices_known
        self.vertices_interpolate = vertices_interpolate

    def interpolate(self, field: np.ndarray) -> np.ndarray:
        """ 
        this function predicts the fields for the vertices to be interpolated from existing vertices on the mesh graph data 
        
        inputs: field is an numpy ndarray (for example, a matrix with size N by d representing node features). 
                It can also be a numpy nparray with more than dimension of 2 as long as we can use it as input to integrate_graph_field function 
                    in the integrator.
        """
        # since fields for vertices to be interpolated are unknown, we initialize them as zeros 
        field[self.vertices_interpolate] = 0
        interpolated_fields = self.integrator.integrate_graph_field(field)[self.vertices_interpolate]
        return interpolated_fields
