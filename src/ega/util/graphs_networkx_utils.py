import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from typing import Optional, List, Tuple


def dist_matrix_mst(adjacency_lists:List[List[int]], \
                    weights_lists:List[List[float]]) -> Tuple[np.ndarray,float]:
    """
    Distance matrix of MST of a graph
    """
    A = numpy_adjacency_matrix(adjacency_lists, weights_lists)
    G = nx.from_numpy_matrix(A)
    T = nx.minimum_spanning_tree(G)
    TA_srs = nx.to_scipy_sparse_array(T)
    TA = np.array(TA_srs.todense())
    csr_adjacency = csr_matrix(TA)
    dist_TA = shortest_path(csgraph=csr_adjacency, directed=False)
    return (dist_TA, TA.sum()/2)


def numpy_adjacency_matrix(adjacency_lists:List[List[int]], \
                           weights_lists:List[List[float]]) -> np.ndarray:
    n = len(adjacency_lists)
    A = np.zeros((n,n))
    for i in range(n):
        for j_idx, j in enumerate(adjacency_lists[i]):
            w = weights_lists[i][j_idx]
            A[i,j] = w
            A[j,i] = w
    return A


def get_adjacency_lists_from_A(A:np.ndarray):
    n = A.shape[0]
    adjacency_lists = [[] for _ in range(n)]
    weights_lists = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if A[i,j] != 0:
                adjacency_lists[i] += [j]
                adjacency_lists[j] += [i]
                weights_lists[i] += [A[i,j]]
                weights_lists[j] += [A[i,j]]
    return adjacency_lists, weights_lists


def get_adjacency_nx(G:nx.Graph) -> np.ndarray:
    A_srs = nx.to_scipy_sparse_array(G)
    A = np.array(A_srs.todense())
    print(f"{A[A > 0].min()=}, {A[A > 0].max()=}")
    return A


def get_rel_diff(a:np.ndarray, b:Optional[np.ndarray]=None, den:Optional[np.ndarray]=None) -> float:
    if not type(a).__module__ == np.__name__:
        a = np.array([a])
        b = np.array([b])  
    if a.ndim==0 or a.shape[0] == a.size or a.ndim>1 and a.shape[1] == a.size: 
        order = None
    else: 
        order = 'fro'
    if den is None:
        return np.linalg.norm(a-b, ord=order) / min(np.linalg.norm(a, ord=order), np.linalg.norm(b, ord=order))
    else:
        return np.linalg.norm(a-den, ord=order) / np.linalg.norm(den, ord=order)


def print_subopt_ratios(dist_T:np.ndarray, dist_G:np.ndarray):
    n = dist_T.shape[0]
    ratios =  np.divide(dist_T, dist_G + np.eye(n))
    ratios = ratios[np.ones(dist_T.shape) - np.eye(n) == 1]
    prob, vals = np.histogram(ratios, 5, density=True)
    print(f"{prob=}")
    print(f"{vals=}, {ratios.min()=}, {ratios.max()=}")
