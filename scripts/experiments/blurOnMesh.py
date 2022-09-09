import numpy as np


def blur_on_mesh(signal: float, mesh_dictionary: dict, time: float,
                 steps: int, transpose = 0):
    area_weight_np = np.array(mesh_dictionary['area_weights'].todense()).T[0]
    blur_inverse = np.diag(area_weight_np) - time / steps * mesh_dictionary['cot_laplacian']
    result = signal
    if not transpose:
        for _ in range(steps):
            # solve system of equations
            result = np.linalg.solve(blur_inverse,
                     np.multiply(result.T, area_weight_np).T)
    else:
        for _ in range(steps):
            result = np.multiply(area_weight_np,
                     np.linalg.solve(blur_inverse.T, result).T).T
    return result



