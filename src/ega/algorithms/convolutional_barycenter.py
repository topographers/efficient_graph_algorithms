import numpy as np
from scipy.optimize import root_scalar
from typing import Callable

class ConvolutionalBarycenter():
    def __init__(self, niter=1500, tol=1e-5, verb=True, normalization=False):
        super(ConvolutionalBarycenter, self).__init__()
        self.niter = niter
        self.tol = tol
        self.verb = verb
        self.normalization = normalization
    
    def get_convolutional_barycenter(
        self, 
        distributions_arr: float,
        alpha: float,
        graph_field_integrator: Callable,
        area_weights: float=None,
        initial_v: float=None,
        initial_barycenter: float=None
    ):
        '''
        inputs:
            graph_field_integrator is blackbox functions that take in a vector x, and return H_t * x, where H_t is the heat kernel.
                since the heat kernel H_t is the same throughout the computations, we assume it is already computed
                and stored in the blackbox function. i.e. the only input of the blackbox function is the vector x
                The blackbox can be one of the following choices:
                (1) explicit materialize the matrix H_t by doing explicit pairwise distance computation (Cuturi, 2013).
                (2) use pre-factored Cholesky decomposition (Solomon et al, 2015)
                (3) any efficient matrix-vector multiplication method

            distributions_arr is a 2d numpy array that contains distributions for which we wish to find the wasserstein_barycenter
            alpha is a 1d array that contains a set of weights, the length is equal to the number of input distributions
            area weights are lumped areas proportional to the sum of triangle areas adjacent to a given vertex

        output:
            wasserstein barycenter as a 1d array        
        '''
        
        area_weights = np.ones(np.size(distributions_arr, 1), 1) if area_weights is None else area_weights
        v = np.ones(distributions_arr.shape) if initial_v is None else initial_v
        barycenter = np.ones((distributions_arr.shape[0], 1)) if initial_barycenter is None else initial_barycenter

        alpha = alpha / np.sum(alpha)

        # Wasserstein barycenter using iterated Bregman projection. Refer to Algorithm 2 in (Solomon et al, 2015)
        for j in range(self.niter):
            old_barycenter = barycenter

            """
            vectors v, w, and d: instead of computing a transportation plan matrix, we can alternatively compute vectors v, w, and d
                reducing the number of unknowns
            """
            integrator_input = np.multiply(v.T, area_weights).T
            integrator_output = graph_field_integrator(integrator_input)
            if self.normalization:
                 integrator_output = integrator_output / graph_field_integrator(np.ones(integrator_input.shape))
            w = np.divide(distributions_arr, integrator_output)

            integrator_input = np.multiply(w.T, area_weights).T
            integrator_output = graph_field_integrator(integrator_input)
            if self.normalization:
                 integrator_output = integrator_output / graph_field_integrator(np.ones(integrator_input.shape))
            d = np.multiply(v, integrator_output)

            d[d < 1e-300] = 1e-300

            barycenter = np.exp(np.sum(np.multiply(alpha, np.log(d)), axis=1))

            entropy = -np.sum(np.multiply(area_weights, np.multiply(barycenter, np.log(barycenter))))

            v = np.multiply(v.T, barycenter).T / d

            change = np.sum(np.multiply(np.absolute(old_barycenter - barycenter), area_weights))
            area = np.sum(np.multiply(barycenter, area_weights))
            
            if self.verb:
                print('Iteration {0}:  change = {1}, area = {2}\n'.format(j, change, area))
            if j > 1 and change < self.tol:
                return barycenter
        return barycenter
