import numpy as np
from typing import Callable


class ConvolutionalDistance:
    def __init__(self, niter=100, tolerance=1e-6, verbose=1):
        super(ConvolutionalDistance, self).__init__()
        self.niter = niter
        self.tolerance = tolerance
        self.verbose = verbose

    def get_convolutional_distance(
            self,
            distribution_1: np.ndarray,
            distribution_2: np.ndarray,
            area_weights: np.ndarray,
            graph_field_integrator: Callable[[np.ndarray], np.ndarray]
    ):
        """
        inputs:
            graph_field_integrator is blackbox functions that take in a vector x, and return H_t * x, where H_t is the
            heat graph_field_integrator.
                since the heat graph_field_integrator H_t is the same throughout the computations, we assume it is
                 already computed and stored in the blackbox function. i.e. the only input of the blackbox function is
                 the vector x
                The blackbox can be one of the following choices:
                (1) explicit materialize the matrix H_t by doing explicit pairwise distance computation (Cuturi, 2013).
                (2) use pre-factored Cholesky decomposition (Solomon et al, 2015)
                (3) any efficient matrix-vector multiplication method

            distribution_1, distribution_2 are 2d arrays of shape (n_vertices, n_vertices)
            area weights are lumped areas proportional to the sum of triangle areas adjacent to a given vertex

        output:
            wasserstein distance as a 1d array
        """

        epsilon = 2.2204e-16

        if area_weights is None:
            area_weights = np.ones(distribution_1.shape[0])

        distribution_1 += epsilon
        distribution_2 += epsilon

        '''
        vectors v, w: instead of computing a transportation plan matrix, we can alternatively compute vectors v, w, 
            reducing the number of unknowns
        aw: element wise product of area_weights and w     
        '''

        v = np.ones(distribution_1.shape)
        w = np.ones(distribution_2.shape)
        aw = np.multiply(area_weights, w)

        distances = np.zeros(v.shape[1])

        # Wasserstein distance using Sinkhorn iterations. Refer to Algorithm 1 in (Solomon et al, 2015)
        for i in range(self.niter):
            """
            av: element wise product of area_weights and v
            aw: element wise product of area_weights and w
            lv: element wise product of av and log of v
            lw: element wise product of aw and log of w
            """
            v = np.divide(distribution_2, graph_field_integrator(aw))
            av = np.multiply(area_weights, v)
            w = np.divide(distribution_1, graph_field_integrator(av))
            aw = np.multiply(area_weights, w)

            old_distances = distances

            lv = np.multiply(av, np.log(v))
            lw = np.multiply(aw, np.log(w))
            distances = \
                np.sum(np.multiply(np.log(np.sum(area_weights)) * av + lv, graph_field_integrator(aw)), axis=1) \
                + np.sum(np.multiply(av, graph_field_integrator(lw)), axis=1)

            change = np.linalg.norm(old_distances - distances)

            if self.verbose:
                print("Iteration {}: {}\n".format(i, change))

            if change < self.tolerance and i > 2:
                break

        distances = np.sqrt(np.maximum(distances, np.zeros(v.shape[1])))

        return distances
