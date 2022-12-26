import numpy as np
import networkx as nx

def get_adjacency_lists_from_A(A):
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

def get_adjacency_nx(G):
    A_srs = nx.to_scipy_sparse_array(G)
    A = np.array(A_srs.todense())
    print(f"{A[A > 0].min() = }, {A[A > 0].max() = }")
    return A

def get_rel_diff(a, b=None, den=None):
    if not type(a).__module__ == np.__name__:
        a = np.array([a])
        b = np.array([b])  
    if a.ndim==0 or a.shape[0] == a.size or a.ndim>1 and a.shape[1] == a.size: ord = None
    else: ord = 'fro'
    if den is None:
        return np.linalg.norm(a-b, ord=ord) / min(np.linalg.norm(a, ord=ord), np.linalg.norm(b, ord=ord))
    else:
        return np.linalg.norm(a-den, ord=ord) / np.linalg.norm(den, ord=ord)


def print_subopt_ratios(dist_T, dist_G):
    n = dist_T.shape[0]
    ratios =  np.divide(dist_T, dist_G + np.eye(n))
    ratios = ratios[np.ones(dist_T.shape) - np.eye(n) == 1]
    prob, vals = np.histogram(ratios, 5, density=True)
    print(f"{prob = }")
    print(f"{vals = }, {ratios.min() = }, {ratios.max() = }")
