import abc

class GFIntegrator(object):
    """Abstract class modeling fields defined on the topological objects.

    The main functionality of the class is to provide fast (approximate)
    multiplication of the following matrix-tensor-product expression:

                    v = Mx for a matrix M \in R^NxN defined as:
                    M[i][j] = f(dist/sim(i,j)),

    where:

    * x is a given input vector with the first dimension of length N and
    indexed by the columns of M,

    * dist/sim is a fixed distance or similarity function defined on the points
    (i,j) \in V x V of the fixed undirected weighted graph G(V, E, W) constituting
    discrete approximation of the topological domain under consideration,

    * f: R -> R is a fixed function.
    """ 
    
    __metaclass__ = abc.ABCMeta

    def __init__(self, adjacency_lists, weights_lists, vertices, f_fun):
        self._adjacency_lists = adjacency_lists
        self._weights_lists = weights_lists
        self._vertices = vertices
        self._f_fun = f_fun

    def vertices(self):
        return self._vertices

    @abc.abstractmethod
    def integrate_graph_field(self, field):
        raise NotImplementedError("Abstract method")
    