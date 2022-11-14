import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall, shortest_path

from ega.algorithms.gf_integrator import GFIntegrator
from tqdm import tqdm


class BFGFIntegrator(GFIntegrator):
    def __init__(self, adjacency_lists, weights_lists, vertices, f_fun):
        super().__init__(adjacency_lists, weights_lists, vertices, f_fun)
        self._m_matrix = self.get_kernel_graph()

    def get_kernel_graph(self):
        n = len(self._adjacency_lists)
        edges = np.zeros((n, n))
        for i in range(n):
            for j_idx, j in enumerate(self._adjacency_lists[i]):
                w = self._weights_lists[i][j_idx]
                edges[i, j] = w
                edges[j, i] = w
        csr_adjacency = csr_matrix(edges)

        # dist_G = floyd_warshall(csgraph=csr_adjacency, directed=False)
        dist_G = shortest_path(csgraph=csr_adjacency, directed=False)

        return self._f_fun(dist_G)

    def integrate_graph_field(self, field):
        return np.einsum('ij,j...->i...', self._m_matrix, field)
