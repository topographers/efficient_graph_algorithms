import numpy as np


def blurOnMesh(signal,M,time,steps,transpose=0):
    h = time/steps
    areaweight_np = np.array(M.areaWeights.todense()).T[0]
    blurInverse = np.diag(areaweight_np) - h*M.cotLaplacian
    result = signal
    if not transpose:
        for _ in range(steps):
            # solve system of equations
            result = np.linalg.solve(blurInverse,
                     np.multiply(result.T, areaweight_np).T)
    else:
        for _ in range(steps):
            result = np.multiply(areaweight_np,
                     np.linalg.solve(blurInverse.T, result).T).T
    return result



