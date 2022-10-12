from scipy.spatial.distance import cdist
import numpy as np

def dist(x1, x2=None, metric='sqeuclidean'):
    """Compute distance between samples in x1 and x2 using function scipy.spatial.distance.cdist
    Parameters
    ----------
    x1 : np.array (n1,d)
        matrix with n1 samples of size d
    x2 : np.array (n2,d), optional
        matrix with n2 samples of size d (if None then x2=x1)
    metric : str, fun, optional
        name of the metric to be computed (full list in the doc of scipy),  If a string,
        the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
        'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
        'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
        'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    Returns
    -------
    M : np.array (n1,n2)
        distance matrix computed with given metric
    """
    if x2 is None:
        x2 = x1

    return cdist(x1, x2, metric=metric)

def reshaper(x):
    x=np.array(x)
    try:
        a=x.shape[1]
        return x
    except IndexError:
        return x.reshape(-1,1)