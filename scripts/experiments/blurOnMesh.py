import numpy as np
import scipy

def blur_on_mesh(signal: float, mesh_dictionary: dict, time: float,
                 steps: int):
    area_weights = mesh_dictionary['area_weights']
    matrix_to_pre_factor = np.diag(area_weights) + time / steps * mesh_dictionary['cot_laplacian']
    # use cholesky factorization to efficiently solve systems of linear equations
    L = np.linalg.cholesky(matrix_to_pre_factor)
    result = signal
    for _ in range(steps):
        result = scipy.linalg.cho_solve((L,True), np.multiply(result.T, area_weights).T)
    return result



