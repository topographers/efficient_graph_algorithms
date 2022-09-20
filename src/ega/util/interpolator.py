from typing import List
import numpy as np 
from ega.algorithms.gf_integrator import GFIntegrator 

class Interpolator():
    """
    given an integrator (for example, brute force), as well as the position of points to be interpolated, 
    this function will do interpolation for these points based on other known points on the mesh graph data 
    """    
    def __init__(self, integrator: GFIntegrator, vertices_known: List[int], vertices_interpolate: List[int]):
        """ 
        integrator: the integrator to be used. For example: brute_force integrator 
        vertices_known: the vertices known on the mesh graph, which are used to predict the fields of the vertices to be interpolated 
        vertices_interpolate: the vertices with unknown fields to be interpolated 
        """
        self.integrator = integrator 
        self.vertices_known = vertices_known
        self.vertices_interpolate = vertices_interpolate

        # since we don't need to predict fields for known vertices, we mask these rows in _m_matrix as zeros 
        self.integrator._m_matrix[vertices_known] = 0 
        
        
    def interpolate(self, field: np.ndarray) -> np.ndarray:
        """ this function predicts the fields for the vertices to be interpolated from existing vertices on the mesh graph data """
        # since fields for vertices to be interpolated are unknown, we initialize them as zeros 
        field[self.vertices_interpolate] = 0 
        interpolated_fields = self.integrator.integrate_graph_field(field)[self.vertices_interpolate]
        return interpolated_fields


