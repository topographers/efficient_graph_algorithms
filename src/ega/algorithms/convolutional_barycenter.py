import numpy as np
from scipy.optimize import root_scalar
from typing import Callable


class ConvolutionalBarycenter:
    def __init__(self, number_of_iterations=1500, tolerance=1e-7, verbose=1, unit_area_projection=0):
        super(ConvolutionalBarycenter, self).__init__()
        self.number_of_iterations = number_of_iterations
        self.verbose = verbose
        self.tolerance = tolerance
        self.unit_area_projection = unit_area_projection

    def get_convolutional_barycenter(
            self,
            distributions_arr: np.ndarray,
            alpha: float,
            graph_field_integrator: Callable,
            area_weights: float = None,
            entropy_limit: float = None,
            initial_v: float = None,
            initial_barycenter: float = None,
    ):
        """
        inputs:
            graph_field_integrator is blackbox functions that take in a vector x, and return H_t * x, where H_t is the
            heat kernel. since the heat kernel H_t is the same throughout the computations, we assume it is already
            computed and stored in the blackbox function. i.e. the only input of the blackbox function is the vector x
            The blackbox can be one of the following choices:
            (1) explicit materialize the matrix H_t by doing explicit pairwise distance computation (Cuturi, 2013).
            (2) use pre-factored Cholesky decomposition (Solomon et al, 2015)
            (3) any efficient matrix-vector multiplication method

            distributions_arr is a 2d numpy array that contains distributions for which we wish to find the
            wasserstein_barycenter alpha is a 1d array that contains a set of weights, the length is equal to the number
            of input distributions area weights are lumped areas proportional to the sum of triangle areas adjacent to
            a given vertex

        output:
            wasserstein barycenter as a 1d array
        """

        area_weights = np.ones(np.size(distributions_arr, 1), 1) if area_weights is None else area_weights
        v = np.ones(distributions_arr.shape) if initial_v is None else initial_v
        barycenter = np.ones((distributions_arr.shape[0], 1)) if initial_barycenter is None else initial_barycenter

        alpha = alpha / np.sum(alpha)

        # Wasserstein barycenter using iterated Bregman projection. Refer to Algorithm 2 in (Solomon et al, 2015)
        for j in range(self.number_of_iterations):
            old_barycenter = barycenter

            """
            vectors v, w, and d: instead of computing a transportation plan matrix, we can alternatively compute vectors
             v, w, and d reducing the number of unknowns
            """
            w = np.divide(distributions_arr, graph_field_integrator(np.multiply(v.T, area_weights).T))

            if self.unit_area_projection == 1:
                matrix_vector_product = graph_field_integrator(np.multiply(w.T, area_weights).T)
                integrals = np.sum(np.multiply(area_weights, np.multiply(v, matrix_vector_product), 1))
                w = w / integrals

            d = np.multiply(v, graph_field_integrator(np.multiply(w.T, area_weights).T))

            d[d < 1e-300] = 1e-300

            barycenter = np.exp(np.sum(np.multiply(alpha, np.log(d)), axis=1))

            entropy = -np.sum(np.multiply(area_weights, np.multiply(barycenter, np.log(barycenter))))

            if j > 1 and (entropy_limit is not None) and (entropy > entropy_limit):
                # Entropic-Sharpening algorithm. Refer to Algorithm 3 in (Solomon et al, 2015)
                def entropic_sharpening(x):
                    return -np.sum(x *
                                   np.multiply(area_weights, np.multiply(np.power(barycenter, x), np.log(barycenter)))
                                   ) - entropy_limit

                try:
                    sol = root_scalar(entropic_sharpening, args=(), method='toms748', bracket=[0.5, 3])
                    beta = sol.root
                    if self.verbose:
                        print('\ta = %g\n', beta)
                except ValueError as _:
                    beta = 1
                    if self.verbose:
                        print('\tProjection failed.\n')

                barycenter = np.power(barycenter, beta)

            v = np.multiply(v.T, barycenter).T / d

            if self.unit_area_projection == 1:
                matrix_vector_product = graph_field_integrator(np.multiply(w, area_weights))
                integrals = np.sum(np.multiply(area_weights, np.multiply(v, matrix_vector_product), 1))
                v = np.divide(v, integrals)

            change = np.sum(np.multiply(np.absolute(old_barycenter - barycenter), area_weights))
            area = np.sum(np.multiply(barycenter, area_weights))

            if self.verbose:
                print('Iteration {0}:  change = {1}, area = {2}\n'.format(j, change, area))
            if j > 1 and change < self.tolerance:
                return barycenter
        return barycenter
