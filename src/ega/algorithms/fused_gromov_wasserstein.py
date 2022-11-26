# Code copied from Optimal Transport for structured data with application on graphs
# https://github.com/tvayer/FGW

import numpy as np
import ot
import optimization
from ega.util.wasserstein_utils import dist, reshaper
from scipy import stats
from scipy.sparse import random


class StopError(Exception):
    pass


def fast_multiply_matrix_square(integrator, field):
    """
    Fast mutiplication with Hadamard square of a matrix and a vector
    Args : integrator : fast graph field integrator to compute einsum with a vector
    """
    assert field.shape[1] == 1
    partial_field = integrator.integrate_graph_field(np.diag(field.squeeze())).T
    return np.diag(integrator.integrate_graph_field(partial_field)).reshape(-1, 1)


def init_matrix(
    C1,
    C2,
    p,
    q,
    loss_fun="square_loss",
    method_type=None,
    source_integrator=None,
    target_integrator=None,
):
    """Return loss matrices and tensors for Gromov-Wasserstein fast computation
    Returns the value of \mathcal{L}(C1,C2) \otimes T with the selected loss
    function as the loss function of Gromow-Wasserstein discrepancy.
    The matrices are computed as described in Proposition 1 in [1]
    Where :
        * C1 : Metric cost matrix in the source space
        * C2 : Metric cost matrix in the target space
        * T : A coupling between those two spaces
    The square-loss function L(a,b)=(1/2)*|a-b|^2 is read as :
        L(a,b) = f1(a)+f2(b)-h1(a)*h2(b) with :
            * f1(a)=(a^2)
            * f2(b)=(b^2)
            * h1(a)=a
            * h2(b)=2b
    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
         Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
         Metric costfr matrix in the target space
    T :  ndarray, shape (ns, nt)
         Coupling between source and target spaces
    p : ndarray, shape (ns,)
    method_type : (str) Choose one of [None, "diffusion", "separator"]
    source_integrator : Callable , fast graph field integrator for source points
    target_integrator : Callable , fast graph field integrator for target points
    Returns
    -------
    constC : ndarray, shape (ns, nt)
           Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
           h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
           h2(C) matrix in Eq. (6)
    References
    ----------
    .. [1] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
    "Gromov-Wasserstein averaging of kernel and distance matrices."
    International Conference on Machine Learning (ICML). 2016.
    """

    if loss_fun == "square_loss":

        def f1(a):
            return a**2

        def f2(b):
            return b**2

        def h1(a):
            return a

        def h2(b):
            return 2 * b

    elif loss_fun == "kl_loss":

        def f1(a):
            return a * np.log(a + 1e-15) - a

        def f2(b):
            return b

        def h1(a):
            return a

        def h2(b):
            return np.log(b + 1e-15)

    if method_type is None:
        constC1 = np.dot(
            np.dot(f1(C1), p.reshape(-1, 1)), np.ones(len(q)).reshape(1, -1)
        )
        constC2 = np.dot(
            np.ones(len(p)).reshape(-1, 1), np.dot(q.reshape(1, -1), f2(C2).T)
        )
    elif method_type == 'diffusion':
        if loss_fun == "square_loss":
            constC1 = np.dot(
                fast_multiply_matrix_square(source_integrator, p.reshape(-1, 1)),
                np.ones(len(q)).reshape(1, -1),
            )
            constC2 = np.dot(
                np.ones(len(p)).reshape(-1, 1),
                fast_multiply_matrix_square(target_integrator, q.reshape(-1, 1)).T,
            )

        elif loss_fun == "kl_loss":
            constC1 = np.dot(
                np.dot(f1(C1), p.reshape(-1, 1)), np.ones(len(q)).reshape(1, -1)
            )  # no idea how to make it faster
            constC2_partial = (
                target_integrator.integrate_graph_field(q.reshape(1, -1).T)
            ).T
            constC2 = np.dot(np.ones(len(p)).reshape(-1, 1), constC2_partial)
        else:
            raise ValueError("Unsupported combination of loss and methods")
    elif method_type == 'separator' :
        pass 
    else : 
        raise ValueError("Unsupported method type")
    constC = constC1 + constC2

    if method_type is None:
        hC1 = h1(C1)
        hC2 = h2(C2)
    else:
        if loss_fun == "square_loss":
            hC1, hC2 = None, None
        else:
            hC1, hC2 = None, h2(C2)

    return constC, hC1, hC2


def tensor_product(
    constC,
    hC1,
    hC2,
    T,
    method_type=None,
    loss_fun=None,
    source_integrator=None,
    target_integrator=None,
):

    """Return the tensor for Gromov-Wasserstein fast computation
    The tensor is computed as described in Proposition 1 Eq. (6) in [1].
    Parameters
    ----------
    constC : ndarray, shape (ns, nt)
           Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
           h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
           h2(C) matrix in Eq. (6)
    T : ndarray shape (ns,nt) coupling matrix between source and target
    Optional :
    method_type : str None defaults to brute force
    source_integrator : Callable function that does fast matrix multplication for source graph
    target_integrator : Callable function that does fast matrix multplication for target graph
    Returns
    -------
    tens : ndarray, shape (ns, nt)
           \mathcal{L}(C1,C2) \otimes T tensor-matrix multiplication result
    References
    ----------
    .. [1] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
    "Gromov-Wasserstein averaging of kernel and distance matrices."
    International Conference on Machine Learning (ICML). 2016.
    """

    if method_type is None:
        A = -np.dot(np.dot(hC1, T), hC2.T)
    else:
        if loss_fun == "square_loss":
            partial_prod = source_integrator.integrate_graph_field(T)
            A = -2 * (target_integrator.integrate_graph_field(partial_prod.T)).T
            del partial_prod
        elif loss_fun == "kl_loss":
            partial_prod = source_integrator.integrate_graph_field(T)
            A = -np.dot(partial_prod, hC2.T)
            del partial_prod
        else:
            raise NotImplementedError(
                "Other types of losses are not currently supported."
            )
    tens = constC + A

    return tens


def gwloss(
    constC,
    hC1,
    hC2,
    T,
    method_type=None,
    loss_fun=None,
    source_integrator=None,
    target_integrator=None,
):

    """Return the Loss for Gromov-Wasserstein
    The loss is computed as described in Proposition 1 Eq. (6) in [1].
    Parameters
    ----------
    constC : ndarray, shape (ns, nt)
           Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
           h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
           h2(C) matrix in Eq. (6)
    T : ndarray, shape (ns, nt)
           Current value of transport matrix T
    Optional :
    method_type : str None defaults to brute force
    source_integrator : Callable function that does fast matrix multplication for source graph
    target_integrator : Callable function that does fast matrix multplication for target graph
    Returns
    -------
    loss : float
           Gromov Wasserstein loss
    References
    ----------
    .. [1] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
    "Gromov-Wasserstein averaging of kernel and distance matrices."
    International Conference on Machine Learning (ICML). 2016.
    """

    tens = tensor_product(
        constC,
        hC1,
        hC2,
        T,
        method_type=method_type,
        loss_fun=loss_fun,
        source_integrator=source_integrator,
        target_integrator=target_integrator,
    )
    return np.sum(tens * T)


def gwggrad(
    constC,
    hC1,
    hC2,
    T,
    method_type=None,
    loss_fun=None,
    source_integrator=None,
    target_integrator=None,
):

    """Return the gradient for Gromov-Wasserstein
    The gradient is computed as described in Proposition 2 in [1].
    Parameters
    ----------
    constC : ndarray, shape (ns, nt)
           Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
           h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
           h2(C) matrix in Eq. (6)
    T : ndarray, shape (ns, nt)
           Current value of transport matrix T
    Optional :
    method_type : str None defaults to brute force
    source_integrator : Callable function that does fast matrix multplication for source graph
    target_integrator : Callable function that does fast matrix multplication for target graph
    Returns
    -------
    grad : ndarray, shape (ns, nt)
           Gromov Wasserstein gradient
    References
    ----------
    .. [1] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
    "Gromov-Wasserstein averaging of kernel and distance matrices."
    International Conference on Machine Learning (ICML). 2016.
    """

    return 2 * tensor_product(
        constC,
        hC1,
        hC2,
        T,
        method_type=method_type,
        loss_fun=loss_fun,
        source_integrator=source_integrator,
        target_integrator=target_integrator,
    )


def gw_lp(
    C1,
    C2,
    p,
    q,
    loss_fun="square_loss",
    alpha=1,
    armijo=True,
    G0=None,
    log=True,
    method_type=None,
    source_positions=None,
    target_positions: np.ndarray = None,
    source_epsilon: float = None,
    target_epsilon: float = None,
    source_lambda_par: float = None,
    target_lambda_par: float = None,
    num_rand_features: int = None,
    dim: int = None,
    verbose=False,
):

    """
    Returns the gromov-wasserstein transport between (C1,p) and (C2,q)
    The function solves the following optimization problem:
    .. math::
        \GW_Dist = \min_T \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}
    Where :
        C1 : Metric cost matrix in the source space
        C2 : Metric cost matrix in the target space
        p  : distribution in the source space
        q  : distribution in the target space
        L  : loss function to account for the misfit between the similarity matrices
        H  : entropy
    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
         Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
         Metric costfr matrix in the target space
    p :  ndarray, shape (ns,)
         distribution in the source space
    q :  ndarray, shape (nt,)
         distribution in the target space
    loss_fun :  string
        loss function used for the solver
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    armijo : bool, optional
        If True the step of the line-search is found via an armijo research. Else closed form is used.
        If there is convergence issues use False.
     G0: ndarray, shape (ns,nt), optional
        If None the initial transport plan of the solver is pq^T.
        Otherwise G0 must satisfy marginal constraints and will be used as initial transport of the solver.
    The rest of the parameters are optional and only used for fast matrix vector multiplication.
        method_type : Anything other than None defaults to fast matrix vector multiplication
        source_positions : (n_s, dim) location of points in d-dim Euclidean space.
        target_positions : (n_t, dim) location of points in d-dim Euclidean space.
        source_epsilon : parameter that controls the epsilon neighbor of source points
        target_epsilon : parameter that controls the epsilon neighbor of target points
        source_lambda_par : diffusion parameter for source graph.
        target_lambda_par : diffusion parameter for target graph.
        num_rand_features : Number of random features
        dim : Input dimensionality of the data
    verbose : bool, optional
        If true returns logs/errors in each iteration
    Returns
    -------
    T : ndarray, shape (ns, nt)
        coupling between the two spaces that minimizes :
            \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}
    log : dict
        convergence information and loss
    References
    ----------
    .. [1] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    .. [2] Mémoli, Facundo. Gromov–Wasserstein distances and the
        metric approach to object matching. Foundations of computational
        mathematics 11.4 (2011): 417-487.
    """

    if method_type is not None:
        dfgf_s_integrator = DFGFIntegrator(
            source_positions,
            source_epsilon,
            source_lambda_par,
            num_rand_features,
            dim,
            random_projection_creator,
            density_function,
            fourier_transform,
        )
        dfgf_t_integrator = DFGFIntegrator(
            target_positions,
            target_epsilon,
            target_lambda_par,
            num_rand_features,
            dim,
            random_projection_creator,
            density_function,
            fourier_transform,
        )
        if loss_fun == "square_loss":
            constC, hC1, hC2 = init_matrix(
                C1,
                C2,
                p,
                q,
                loss_fun,
                method_type=method_type,
                source_integrator=dfgf_s_integrator,
                target_integrator=dfgf_t_integrator,
            )
        elif loss_fun == "kl_loss":
            constC, hC1, hC2 = init_matrix(
                C1,
                C2,
                p,
                q,
                loss_fun,
                method_type=method_type,
                source_integrator=None,
                target_integrator=dfgf_t_integrator,
            )
        else:
            raise ValueError("incorrect loss function used")
    else:
        dfgf_s_integrator = None
        dfgf_t_integrator = None
        constC, hC1, hC2 = init_matrix(
            C1,
            C2,
            p,
            q,
            loss_fun,
            method_type=method_type,
            source_integrator=dfgf_s_integrator,
            target_integrator=dfgf_t_integrator,
        )

    if method_type is None:
        M = np.zeros((C1.shape[0], C2.shape[0]))
    else:
        M = np.zeros((p.shape[0], q.shape[0]))

    if G0 is None:
        G0 = p[:, None] * q[None, :]
    else:  # check marginals
        np.testing.assert_allclose(G0.sum(axis=1), p, atol=1e-08)
        np.testing.assert_allclose(G0.sum(axis=0), q, atol=1e-08)

    def f(G):
        return gwloss(
            constC,
            hC1,
            hC2,
            G,
            method_type=method_type,
            loss_fun=loss_fun,
            source_integrator=dfgf_s_integrator,
            target_integrator=dfgf_t_integrator,
        )

    def df(G):
        return gwggrad(
            constC,
            hC1,
            hC2,
            G,
            method_type=method_type,
            loss_fun=loss_fun,
            source_integrator=dfgf_s_integrator,
            target_integrator=dfgf_t_integrator,
        )

    if log:
        res, log0 = optimization.cg(
            a=p,
            b=q,
            M=M,
            reg=alpha,
            f=f,
            df=df,
            G0=G0,
            armijo=armijo,
            C1=C1,
            C2=C2,
            constC=constC,
            log=log,
            alpha_min=0,
            alpha_max=1,
            method_type=method_type,
            source_integrator=dfgf_s_integrator,
            target_integrator=dfgf_t_integrator,
            verbose=verbose,
        )
        log0["gw_dist"] = gwloss(
            constC,
            hC1,
            hC2,
            res,
            method_type=method_type,
            loss_fun=loss_fun,
            source_integrator=dfgf_s_integrator,
            target_integrator=dfgf_t_integrator,
        )
        return res, log0
    else:
        res = optimization.cg(
            a=p,
            b=q,
            M=M,
            reg=alpha,
            f=f,
            df=df,
            G0=G0,
            armijo=armijo,
            C1=C1,
            C2=C2,
            constC=constC,
            log=log,
            alpha_min=0,
            alpha_max=1,
            method_type=method_type,
            source_integrator=dfgf_s_integrator,
            target_integrator=dfgf_t_integrator,
            verbose=verbose,
        )
        return res
