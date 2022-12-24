import numpy as np
import scipy
from sksparse.cholmod import cholesky

def blur_on_mesh(signal: float, mesh_dictionary: dict):
    '''
    efficient matrix vector multiplication using cotangent Laplacian and Cholesky factorization.
    the procedure is described in (Solomon et al, 2015) section 5.
    
    inputs:
        signal is the vector that we want to multiply by a matrix
        mesh_dictionary stores information of the mesh, including pre-computed cotangent Laplacian,
            which is a representation of the matrix

    output:
        a 1d vector, the result of the matrix vector multplication  
    '''
    result = signal
    area_weights = mesh_dictionary['area_weights']
    if mesh_dictionary['sparse']:
        factor = mesh_dictionary['factor']
        for _ in range(mesh_dictionary['steps']):
            b = np.multiply(result.T, area_weights).T
            result = factor(b)
    else:
        L = mesh_dictionary['L']
        for _ in range(mesh_dictionary['steps']):
            b = np.multiply(result.T, area_weights).T
            result = scipy.linalg.cho_solve((L, True), b)
    return result



