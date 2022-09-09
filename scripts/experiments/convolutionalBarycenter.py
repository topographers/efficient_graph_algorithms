import numpy as np
from scipy.optimize import root_scalar
from typing import Callable

def convolutional_barycenter(distributions_arr: float, alpha: float, area_weights: float,
        kernel: Callable, kernel_transpose: Callable, options: dict, entropyLimit=None):

    if kernel_transpose is None:
        kernel_transpose = kernel

    area_weights_np = np.array(area_weights.todense()).T[0]

    if area_weights_np is None:
        area_weights_np = np.ones(np.size(distributions_arr,1),1)

    niter = options['niter'] #1500
    tol = options['tol'] # 1e-7
    verb = options['verb'] # 1
    displayFunction = options['disp'] # []
    v = options['initial_v'] # ones(np.size(distributions_arr)))
    barycenter = options['initial_barycenter'] # np.ones(np.size(distributions_arr,1),1)
    unit_area_projection = options['unit_area_projection'] # 0

    alpha = alpha/np.sum(alpha)

    """

    Wasserstein barycenter using iterated Bregman projection. Refer to Algorithm 2 in (Solomon et al, 2015)

    """

    for j in range(niter):
        old_barycenter = barycenter

        """
        
        vectors v, w, and d: instead of computing a transportation plan matrix, we can alternatively compute vectors v, w, and d
            reducing the number of unknowns

        """
        w = np.divide(distributions_arr, kernel_transpose(np.multiply(v.T,area_weights_np).T))
        
        if unit_area_projection == 1:
            integrals = np.sum(np.multiply(area_weights_np,np.multiple(v, kernel(np.multiply(w.T,area_weights_np).T)),1))
            w = w/integrals
        
        d = np.multiply(v, kernel(np.multiply(w.T,area_weights_np).T))

        d[d<1e-300] = 1e-300

        barycenter = np.exp(np.sum(np.multiply(alpha,np.log(d)),axis=1))

        entropy = -np.sum(np.multiply(area_weights_np, np.multiply(barycenter,np.log(barycenter))))

        if j>1 and (entropyLimit is not None) and (entropy>entropyLimit):
            """

            Entropic-Sharpening algorithm. Refer to Algorithm 3 in (Solomon et al, 2015)
            
            """
            fn = lambda x: -np.sum(x*np.multiply(area_weights_np, np.multiply(np.power(barycenter, x), np.log(barycenter)))) - entropyLimit
            try:
                sol = root_scalar(fn, args=(), method='toms748', bracket=[0.5, 3])
                beta = sol.root 
                if verb==1:
                    print('\ta = %g\n', beta)
            except:
                beta = 1
                print('\tProjection failed.\n')
            
            barycenter = np.power(barycenter, beta)

        v = np.multiply(v.T,barycenter).T/d
        
        if unit_area_projection == 1:
            integrals = np.sum(np.multiply(area_weights_np, np.multiply(v, kernel(np.multiply(w,area_weights))),1));
            v = np.divide(v,integrals)
        

        change = np.sum(np.multiply(np.absolute(old_barycenter-barycenter), area_weights_np))
        area = np.sum(np.multiply(barycenter, area_weights_np))
        
        if verb==1:
            print('Iteration {0}:  change = {1}, area = {2}\n'.format(j,change,area))
        if j>1 and change < tol:
            return barycenter, v
    return barycenter, v
