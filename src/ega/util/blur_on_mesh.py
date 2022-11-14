import numpy as np
import scipy

from ega.data_structures.mesh_data import MeshData


def blur_on_mesh(signal: float, mesh_data: MeshData, time: float, steps: int):
    """
    efficient matrix vector multiplication using cotangent Laplacian and Cholesky factorization.
    the procedure is described in (Solomon et al, 2015) section 5.

    inputs:
        signal is the vector that we want to multiply by a matrix
        mesh_dictionary stores information of the mesh, including pre-computed cotangent Laplacian,
            which is a representation of the matrix

    output:
        a 1d vector, the result of the matrix vector multplication
    """
    area_weights = mesh_data.area_weights
    if steps <= 0:
        raise ValueError('steps must be greater than zero')
    matrix_to_pre_factor = np.diag(area_weights) + time / steps * mesh_data.cotangent_laplacian
    # use cholesky factorization to efficiently solve systems of linear equations
    L = np.linalg.cholesky(matrix_to_pre_factor)
    result = signal
    for _ in range(steps):
        result = scipy.linalg.cho_solve((L, True), np.multiply(result.T, area_weights).T)
    return result
