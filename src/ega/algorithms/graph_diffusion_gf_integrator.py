import numpy as np 
from scipy import linalg 
from ega.util.mesh_utils import construct_random_features
from typing import Callable

class DFGFIntegrator(object):
    """
    The class supporting fast (approximate) multiplication of the diffusion kernel matrix
    M_lambda \in R ^ {N * N} defined as M_lambda = exp(lambda Adj(G)), 
    where Adj(G) stands for the adjacency matrix of the undirected graph G, 
    with a given tensor field \in R^{N * d_1 * ... * d_l}.
    The fast multiplication procedure is based on the low-rank decomposition of Adj(G).
    The decomposition is obtained by constructing random features leveraging: 
        (a) Fourier Transform of the function defining edges between nodes of the graph 
            (see: detailed explanation below) 
        and (b) random projections
        
    We assume that G is the so-called epsilon-neighbor graph, where vertices correspond 
    to the points in the d-dimensional space (in applications involving point cloud data 
    we have: d=3 ) and edges exist between nodes within distance <= epsilon 
    (for a given epsilon > 0) (the distance is with respect to the particular metric, 
    the class supports different metrics via different choices of the fourier_transform 
    argument of the constructors of its objects; this argument is a FT of the function  
    f: R^3 -> R such that the weight of the edge between a node with coordinates x 
    and a node with coordinates y is f(x−y) ). 
    We refer to the tensor M_lambda * field  as the graph field integration tensor with 
    the weight matrix given by M_lambda.
    
    
    parameters definition:
    
    positions: 2d numpy array of the shape [N, d] encoding positions of the N 
               given points from the  d -dimensional space
               
    epsilon: the parameter epsilon defining (together with positions and a particular 
                                             metric applied) a graph G
    
    lambda_par: parameter defining (together with graph G) a graph diffusion kernel matrix
    
    num_rand_features: number of random features used to low-rank-decompose 
                       graph adjacency matrix
                       
    dim: dimensionality of the points of the graph
    
    random_projection_creator: a function N*N -> R^{N*N} that constructs random projections 
                               used to construct low-rank decomposition of the adjacency 
                               matrix; for the input: (num_rand_features, dim), a matrix of 
                               num_rand_features rows and dim columns is created (with 
                               different rows corresponding to different projections)
                               
    density_function: density function of the probabilistic distribution applied by the 
                      above random_projection_creator to construct projections
                      
    fourier_transform: Fourier Transform of the function defining edges between graph nodes.
    """
    
    def __init__(self, positions: np.ndarray, 
                 epsilon: float, 
                 lambda_par: float, 
                 num_rand_features: int, 
                 dim: int,
                 random_projection_creator: Callable, 
                 density_function: Callable, 
                 fourier_transform: Callable):
        """
        _a_matrix:         matrix of shape N x M
        _bt_matrix:        matrix of shape M x N
        _invbta_matrix:    matrix of shape M x M
        _expbta_matrix:    matrix of shape M x M
        """
        self._positions = positions
        self._a_matrix = construct_random_features(positions, 
                                                   random_projection_creator, 
                                                   density_function,
                                                   num_rand_features,
                                                   fourier_transform,
                                                   epsilon)          
        self._bt_matrix = construct_random_features(-positions, 
                                                    random_projection_creator,
                                                    density_function, 
                                                    num_rand_features,
                                                    fourier_transform,
                                                    epsilon)
        self._bt_matrix = np.transpose(self._bt_matrix)
        bta = np.matmul(self._bt_matrix, self._a_matrix)
        self._invbta_matrix = linalg.inv(bta)     
        self._expbta_matrix = linalg.expm(lambda_par * bta)
        self._expbta_matrix -= np.identity(num_rand_features)  
         
    def integrate_graph_field(self, field: np.ndarray):
        res = np.einsum('mn,n...->m...',self._bt_matrix, field)
        res = np.einsum('am,m...->a...', self._invbta_matrix, res)
        res = np.einsum('am,m...->a...', self._expbta_matrix, res)
        res = np.einsum('nm,m...->n...', self._a_matrix, res)
        return np.real(field + res)

