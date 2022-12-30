# Base Code copied from Optimal Transport for structured data with application on graphs with some modification coming from the POT library
# https://github.com/tvayer/FGW

import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist
import ot
import ega.algorithms.optimization
from ega.algorithms.graph_diffusion_gf_integrator import DFGFIntegrator
from ega.algorithms.separation_gf_integrator import SeparationGFIntegrator
from ega.util.mesh_utils import (
    random_projection_creator,
    density_function,
    fourier_transform,
)


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

    elif method_type == "diffusion":
        if loss_fun == "square_loss":
            if source_integrator is not None and target_integrator is not None:
                constC1 = np.dot(
                    fast_multiply_matrix_square(source_integrator, p.reshape(-1, 1)),
                    np.ones(len(q)).reshape(1, -1),
                )
                constC2 = np.dot(
                    np.ones(len(p)).reshape(-1, 1),
                    fast_multiply_matrix_square(target_integrator, q.reshape(-1, 1)).T,
                )
            elif target_integrator is None and source_integrator is not None:
                constC1 = np.dot(
                    fast_multiply_matrix_square(source_integrator, p.reshape(-1, 1)),
                    np.ones(len(q)).reshape(1, -1),
                )
                constC2 = np.dot(
                np.ones(len(p)).reshape(-1, 1), np.dot(q.reshape(1, -1), f2(C2).T)
            )
            elif source_integrator is None and target_integrator is not None:
                constC1 = np.dot(
                np.dot(f1(C1), p.reshape(-1, 1)), np.ones(len(q)).reshape(1, -1)
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

    elif method_type == "separator":
        if loss_fun == "square_loss":
            constC1 = np.dot(
                source_integrator.integrate_graph_field(p.reshape(-1, 1)),
                np.ones(len(q)).reshape(1, -1),
            )
            constC2 = np.dot(
                np.ones(len(p)).reshape(-1, 1),
                target_integrator.integrate_graph_field(q.reshape(-1, 1)).T,
            )

        else:
            raise NotImplementedError("KL div loss is not implemented")

    else:
        raise ValueError("Unsupported method type")

    constC = constC1 + constC2

    if method_type is None:
        hC1 = h1(C1)
        hC2 = h2(C2)
    else:
        if loss_fun == "square_loss":
            if C1 is None and C2 is None :
                hC1, hC2 = None, None
            elif C1 is None and C2 is not None :
                hC1, hC2 = None, h2(C2)
            elif C2 is None and C1 is not None :
                hC1, hC2 = h1(C1), None
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
            if source_integrator is not None and target_integrator is not None:
                partial_prod = source_integrator.integrate_graph_field(T)
                A = -2 * (target_integrator.integrate_graph_field(partial_prod.T)).T
                del partial_prod
            elif source_integrator is not None and target_integrator is None:
                partial_prod = source_integrator.integrate_graph_field(T)
                A = -np.dot(partial_prod, hC2.T)
                del partial_prod
            elif target_integrator is not None and source_integrator is None :
                partial_prod = np.dot(hC1, T)
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
    C1=None,
    C2=None,
    p=None,
    q=None,
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
    source_adjacency_lists=None,
    source_weights_lists=None,
    source_vertices=None,
    source_unit_size=None,
    threshold_nb_vertices=None,
    target_adjacency_lists=None,
    target_weights_lists=None,
    target_vertices=None,
    target_unit_size=None,
    verbose=False,
    source_integrator=None,
    target_integrator=None,
    max_iter=1000,
    stopThr=1e-9,
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
        method_type : Choose from [None, "diffusion", "separator"]
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

    if method_type == "diffusion":
        if source_integrator is None and target_integrator is None:
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
        else:
            dfgf_s_integrator = source_integrator
            dfgf_t_integrator = target_integrator

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

    elif method_type is None:
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

    elif method_type == "separator":
        if loss_fun == "square_loss":
            f_fun_s = lambda x: np.exp(-source_lambda_par * x)
            f_fun_t = lambda x: np.exp(-target_lambda_par * x)
            f_fun_s_sq = lambda x: np.exp(-2 * source_lambda_par * x)
            f_fun_t_sq = lambda x: np.exp(-2 * target_lambda_par * x)
            dfgf_s_sq_integrator = SeparationGFIntegrator(
                adjacency_lists=source_adjacency_lists,
                weights_lists=source_weights_lists,
                vertices=source_vertices,
                f_fun=f_fun_s_sq,
                unit_size=source_unit_size,
                threshold_nb_vertices=threshold_nb_vertices,
            )
            dfgf_t_sq_integrator = SeparationGFIntegrator(
                adjacency_lists=target_adjacency_lists,
                weights_lists=target_weights_lists,
                vertices=target_vertices,
                f_fun=f_fun_t_sq,
                unit_size=target_unit_size,
                threshold_nb_vertices=threshold_nb_vertices,
            )
            dfgf_s_integrator = SeparationGFIntegrator(
                adjacency_lists=source_adjacency_lists,
                weights_lists=source_weights_lists,
                vertices=source_vertices,
                f_fun=f_fun_s,
                unit_size=source_unit_size,
                threshold_nb_vertices=threshold_nb_vertices,
            )
            dfgf_t_integrator = SeparationGFIntegrator(
                adjacency_lists=target_adjacency_lists,
                weights_lists=target_weights_lists,
                vertices=target_vertices,
                f_fun=f_fun_t,
                unit_size=source_unit_size,
                threshold_nb_vertices=threshold_nb_vertices,
            )
            constC, hC1, hC2 = init_matrix(
                C1,
                C2,
                p,
                q,
                loss_fun,
                method_type=method_type,
                source_integrator=dfgf_s_sq_integrator,
                target_integrator=dfgf_t_sq_integrator,
            )
        else:
            raise NotImplementedError("KL Div Loss is not implemented")
    else:
        raise ValueError("Unsupported method type")

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
        res, log0 = ega.algorithms.optimization.cg(
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
            numItermax=max_iter,
            stopThr=stopThr,
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
        res = ega.algorithms.optimization.cg(
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
            numItermax=max_iter,
            stopThr=stopThr,
        )
        return res


def fgw_lp(
    M,
    C1=None,
    C2=None,
    p=None,
    q=None,
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
    source_adjacency_lists=None,
    source_weights_lists=None,
    source_vertices=None,
    source_unit_size=None,
    threshold_nb_vertices=None,
    target_adjacency_lists=None,
    target_weights_lists=None,
    target_vertices=None,
    target_unit_size=None,
    verbose=False,
    source_integrator=None,
    target_integrator=None,
    max_iter=1000,
    stopThr=1e-9,
):
    """
    Computes the FGW distance between two graphs see [3]
    .. math::
        \gamma = arg\min_\gamma (1-\alpha)*<\gamma,M>_F + alpha* \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}
        s.t. \gamma 1 = p
             \gamma^T 1= q
             \gamma\geq 0
    where :
    - M is the (ns,nt) metric cost matrix
    - :math:`f` is the regularization term ( and df is its gradient)
    - a and b are source and target weights (sum to 1)
    The algorithm used for solving the problem is conditional gradient as discussed in  [1]_
    Parameters
    ----------
    M  : ndarray, shape (ns, nt)
         Metric cost matrix between features across domains
    C1 : ndarray, shape (ns, ns)
         Metric cost matrix respresentative of the structure in the source space
    C2 : ndarray, shape (nt, nt)
         Metric cost matrix espresentative of the structure in the target space
    p :  ndarray, shape (ns,)
         distribution in the source space
    q :  ndarray, shape (nt,)
         distribution in the target space
    loss_fun :  string,optionnal
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
        If True the steps of the line-search is found via an armijo research. Else closed form is used.
        If there is convergence issues use False.
    The rest of the parameters are optional and only used for fast matrix vector multiplication.
        method_type : Choose from [None, "diffusion", "separator"]
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
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters
    References
    ----------
    .. [3] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """

    if method_type == "diffusion":
        if source_integrator is None and target_integrator is None:
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
        else:
            dfgf_s_integrator = source_integrator
            dfgf_t_integrator = target_integrator

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

    elif method_type is None:
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

    elif method_type == "separator":
        if loss_fun == "square_loss":
            f_fun_s = lambda x: np.exp(-source_lambda_par * x)
            f_fun_t = lambda x: np.exp(-target_lambda_par * x)
            f_fun_s_sq = lambda x: np.exp(-2 * source_lambda_par * x)
            f_fun_t_sq = lambda x: np.exp(-2 * target_lambda_par * x)
            dfgf_s_sq_integrator = SeparationGFIntegrator(
                adjacency_lists=source_adjacency_lists,
                weights_lists=source_weights_lists,
                vertices=source_vertices,
                f_fun=f_fun_s_sq,
                unit_size=source_unit_size,
                threshold_nb_vertices=threshold_nb_vertices,
            )
            dfgf_t_sq_integrator = SeparationGFIntegrator(
                adjacency_lists=target_adjacency_lists,
                weights_lists=target_weights_lists,
                vertices=target_vertices,
                f_fun=f_fun_t_sq,
                unit_size=target_unit_size,
                threshold_nb_vertices=threshold_nb_vertices,
            )
            dfgf_s_integrator = SeparationGFIntegrator(
                adjacency_lists=source_adjacency_lists,
                weights_lists=source_weights_lists,
                vertices=source_vertices,
                f_fun=f_fun_s,
                unit_size=source_unit_size,
                threshold_nb_vertices=threshold_nb_vertices,
            )
            dfgf_t_integrator = SeparationGFIntegrator(
                adjacency_lists=target_adjacency_lists,
                weights_lists=target_weights_lists,
                vertices=target_vertices,
                f_fun=f_fun_t,
                unit_size=source_unit_size,
                threshold_nb_vertices=threshold_nb_vertices,
            )
            constC, hC1, hC2 = init_matrix(
                C1,
                C2,
                p,
                q,
                loss_fun,
                method_type=method_type,
                source_integrator=dfgf_s_sq_integrator,
                target_integrator=dfgf_t_sq_integrator,
            )
        else:
            raise NotImplementedError("KL Div Loss is not implemented")
    else:
        raise ValueError("Unsupported method type")

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
        res, log0 = ega.algorithms.optimization.cg(
            a=p,
            b=q,
            M=(1 - alpha) * M,
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
            numItermax=max_iter,
            stopThr=stopThr,
        )
        fgw_dist = log0["loss"][-1]
        log0["fgw_dist"] = fgw_dist
        return res, log0
    else:
        res = ega.algorithms.optimization.cg(
            a=p,
            b=q,
            M=(1 - alpha) * M,
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
            numItermax=max_iter,
            stopThr=stopThr,
        )
        return res


def reshaper(x):
    x = np.array(x)
    try:
        a = x.shape[1]
        return x
    except IndexError:
        return x.reshape(-1, 1)


def update_square_loss(p, lambdas, T, Cs, method_type):
    """
    Updates C according to the L2 Loss kernel with the S Ts couplings
    calculated at each iteration
    Parameters
    ----------
    p  : ndarray, shape (N,)
         masses in the targeted barycenter
    lambdas : list of float
              list of the S spaces' weights
    T : list of S np.ndarray(ns,N)
        the S Ts couplings calculated at each iteration
    Cs : list of S ndarray, shape(ns,ns)
         Metric cost matrices
    Returns
    ----------
    C : ndarray, shape (nt,nt)
        updated C matrix
    """
    if method_type is None:
        tmpsum = sum(
            [lambdas[s] * np.dot(np.dot(T[s].T, Cs[s]), T[s]) for s in range(len(T))]
        )
    else:
        tmpsum = sum(
            [
                lambdas[s] * np.dot(T[s].T, Cs[s].integrate_graph_field(T[s]))
                for s in range(len(T))
            ]
        )
    ppt = np.outer(p, p)

    return tmpsum / ppt


def update_kl_loss(p, lambdas, T, Cs, method_type):
    r"""
    Updates :math:`\mathbf{C}` according to the KL Loss kernel with the `S` :math:`\mathbf{T}_s` couplings calculated at each iteration
    Parameters
    ----------
    p  : array-like, shape (N,)
        Weights in the targeted barycenter.
    lambdas : list of float
        List of the `S` spaces' weights
    T : list of S array-like of shape (ns,N)
        The `S` :math:`\mathbf{T}_s` couplings calculated at each iteration.
    Cs : list of S array-like, shape(ns,ns)
        Metric cost matrices.
    Returns
    ----------
    C : array-like, shape (`ns`, `ns`)
        updated :math:`\mathbf{C}` matrix
    """
    if method_type is None:
        tmpsum = sum(
            [lambdas[s] * np.dot(np.dot(T[s].T, Cs[s]), T[s]) for s in range(len(T))]
        )
    else:
        raise NotImplementedError("KL loss is not implemented using fast variants.")
    ppt = np.outer(p, p)

    return np.exp(tmpsum / ppt)


def update_cross_feature_matrix(X, Y, metric="sqeuclidean", p=2, w=None):

    """
    Updates M the distance matrix between the features
    calculated at each iteration
    ----------
    X : ndarray, shape (N,d)
        First features matrix, N: number of samples, d: dimension of the features
    Y : ndarray, shape (M,d)
        Second features matrix, N: number of samples, d: dimension of the features
    Returns
    ----------
    M : ndarray, shape (N,M)
    """

    if metric == "dirac":
        f = lambda x, y: x != y
        return cdist(X, Y, metric=f)
    elif metric == "hamming":
        raise ValueError("Hammming will produce worng results. Do not use.")
    else:
        return ot.dist(X, Y, metric=metric, p=p, w=w)


def update_Ms(X, Ys, metric, p, w):

    l = [
        update_cross_feature_matrix(X, Ys[s], metric=metric, p=p, w=w)
        for s in range(len(Ys))
    ]

    return l


def update_feature_matrix(lambdas, Ys, Ts, p):

    """
    Updates the feature with respect to the S Ts couplings. See "Solving the barycenter problem with Block Coordinate Descent (BCD)" in [3]
    calculated at each iteration
    Parameters
    ----------
    p  : ndarray, shape (N,)
         masses in the targeted barycenter
    lambdas : list of float
              list of the S spaces' weights
    Ts : list of S np.ndarray(ns,N)
        the S Ts couplings calculated at each iteration
    Ys : list of S ndarray, shape(d,ns)
         The features
    Returns
    ----------
    X : ndarray, shape (d,N)
    References
    ----------
    .. [3] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """

    p = 1.0 / p
    tmpsum = sum(
        [lambdas[s] * np.dot(Ys[s], Ts[s].T) * p[None, :] for s in range(len(Ts))]
    )

    return tmpsum


def gw_barycenters(
    N,
    Cs,
    ps,
    q,
    lambdas,
    loss_fun,
    max_iter=1000,
    tol=1e-9,
    verbose=False,
    log=False,
    init_C=None,
    random_state=None,
    alpha=0.5,
    armijo=True,
    method_type="diffusion",
    integrators=None,
    stopThr=1e-9,
):
    if Cs is not None:
        S = len(Cs)
    else:
        S = len(integrators)

    # Initialization of C : random SPD matrix (if not provided by user)
    if init_C is None:
        np.random.seed(random_state)
        xalea = np.random.rand(N, 2)
        C = sp.spatial.distance.cdist(xalea,xalea, 'minkowski', p=1)
        C /= C.max()
    else:
        C = init_C #choosing proper init_C can lead to better results

    cpt = 0
    err = 1

    error = []

    while err > tol and cpt < max_iter:
        Cprev = C

        if method_type is None:
            T = [
                gw_lp(
                    Cs[s],
                    C,
                    ps[s],
                    q,
                    loss_fun,
                    max_iter=max_iter,
                    stopThr=stopThr,
                    log=False,
                    armijo=armijo,
                    alpha=alpha,
                    method_type=method_type,
                )
                for s in range(S)
            ]
        elif method_type == "diffusion":
            T = [
                gw_lp(
                    C1=None,
                    C2=C,
                    p=ps[s],
                    q=q,
                    loss_fun=loss_fun,
                    alpha=alpha,
                    armijo=armijo,
                    G0=None,
                    log=False,
                    method_type=method_type,
                    max_iter=max_iter,
                    stopThr=stopThr,
                    source_integrator=integrators[s],
                    target_integrator=None,
                )
                for s in range(S)
            ]
        else:
            raise NotImplementedError("KL loss is not yet implemented")

        if method_type is None:
            if loss_fun == "square_loss":
                C = update_square_loss(q, lambdas, T, Cs, method_type)

            elif loss_fun == "kl_loss":
                C = update_kl_loss(q, lambdas, T, Cs, method_type)

        elif method_type == "diffusion":
            if loss_fun == "square_loss":
                C = update_square_loss(q, lambdas, T, integrators, method_type)

            elif loss_fun == "kl_loss":
                raise NotImplementedError("KL loss is not supported yet")

        else:
            raise NotImplementedError("Other FM method is not supported yet")

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = np.linalg.norm(C - Cprev)
            error.append(err)

            if verbose:
                if cpt % 200 == 0:
                    print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                print("{:5d}|{:8e}|".format(cpt, err))

        cpt += 1

    if log:
        return C, {"err": error}
    else:
        return C


def fgw_barycenters(
    N,
    Ys,
    Cs,
    ps,
    lambdas,
    alpha,
    fixed_structure=False,
    fixed_features=False,
    p=None,
    loss_fun="square_loss",
    max_iter=1000,
    tol=1e-9,
    verbose=False,
    random_seed=42,
    log=False,
    init_C=None,
    init_X=None,
    armijo=True,
    integrators=None,
    method_type=None,
    metric="dirac",
    p_norm=None,
    w=None,
    stopThr=1e-9,
):
    """
    Compute the fgw barycenter as presented eq (5) in [3].
    ----------
    N : integer
        Desired number of samples of the target barycenter
    Ys: list of ndarray, each element has shape (ns,d)
        Features of all samples
    Cs : list of ndarray, each element has shape (ns,ns)
         Structure matrices of all samples
    ps : list of ndarray, each element has shape (ns,)
        masses of all samples
    lambdas : list of float
              list of the S spaces' weights
    alpha : float
            Alpha parameter for the fgw distance
    fixed_structure :  bool
                       Wether to fix the structure of the barycenter during the updates
    fixed_features :  bool
                       Wether to fix the feature of the barycenter during the updates
    init_C :  ndarray, shape (N,N), optional
              initialization for the barycenters' structure matrix. If not set random init
    init_X :  ndarray, shape (N,d), optional
              initialization for the barycenters' features. If not set random init
    metric : str | callable, optional
        'sqeuclidean' or 'euclidean' on all backends. On numpy the function also
        accepts  from the scipy.spatial.distance.cdist function : 'braycurtis',
        'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice',
        'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
        'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
        'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    Returns
    ----------
    X : ndarray, shape (N,d)
        Barycenters' features
    C : ndarray, shape (N,N)
        Barycenters' structure matrix
    log_:
        T : list of (N,ns) transport matrices
        Ms : all distance matrices between the feature of the barycenter and the other features dist(X,Ys) shape (N,ns)
    References
    ----------
    .. [3] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """
    np.random.seed(random_seed)

    if Cs is not None:
        S = len(Cs)
    else:
        S = len(integrators)

    d = reshaper(Ys[0]).shape[1]  # dimension on the node features
    if p is None:
        p = np.ones(N) / N

    if fixed_structure: 
        if init_C is None:
            raise ValueError('If C is fixed it must be initialized')
        else:
            C = init_C
    else:
        if init_C is None:
            xalea = np.random.randn(N, 2)
            C = sp.spatial.distance.cdist(xalea,xalea, 'minkowski', p=1)
            C /= C.max()
        else :
            C = init_C # Note that choosing a proper init_C can lead to improved performance

    if fixed_features:
        if init_X is None:
            raise ValueError("If X is fixed it must be initialized")
        else:
            X = init_X
    else:
        if init_X is None:
            X = np.zeros((N, d))
        else:
            X = init_X

    T = [np.outer(p, q) for q in ps]  # aligning with POT code
    # T=[random_gamma_init(p,q) for q in ps]

    # X is N,d
    # Ys is ns,d
    Ms = update_Ms(X, Ys, metric=metric, p=p_norm, w=w)
    # Ms is N,ns

    cpt = 0
    err_feature = 1
    err_structure = 1

    if log:
        log_ = {}
        log_["err_feature"] = []
        log_["err_structure"] = []
        log_["Ts_iter"] = []

    while (err_feature > tol or err_structure > tol) and cpt < max_iter:
        Cprev = C
        Xprev = X

        if not fixed_features:
            Ys_temp = [y.T for y in Ys]
            X = update_feature_matrix(lambdas, Ys_temp, T, p)

        # X must be N,d
        # Ys must be ns,d
        Ms = update_Ms(X.T, Ys, metric=metric, p=p_norm, w=w)

        if not fixed_structure:
            if loss_fun == "square_loss":
                # T must be ns,N
                # Cs must be ns,ns
                # p must be N,1
                T_temp = [t.T for t in T]
                if method_type is None:
                    C = update_square_loss(p, lambdas, T_temp, Cs, method_type)
                elif method_type == "diffusion":
                    C = update_square_loss(p, lambdas, T_temp, integrators, method_type)
                else:
                    raise NotImplementedError("Other methods are not implemented")

        # Ys must be d,ns
        # Ts must be N,ns
        # p must be N,1
        # Ms is N,ns
        # C is N,N
        # Cs is ns,ns
        # p is N,1
        # ps is ns,1
        if method_type is None:
            T = [
                fgw_lp(
                    Ms[s],
                    C,
                    Cs[s],
                    p,
                    ps[s],
                    loss_fun,
                    alpha,
                    max_iter=max_iter,
                    stopThr=stopThr,
                    verbose=verbose,
                    armijo=armijo,
                    log=False,
                )
                for s in range(S)
            ]
        elif method_type == "diffusion":
            T = [
                fgw_lp(
                    Ms[s],
                    C1=C,
                    C2=None,
                    p=p,
                    q=ps[s],
                    loss_fun=loss_fun,
                    alpha=alpha,
                    armijo=armijo,
                    log=False,
                    method_type=method_type,
                    source_positions=None,
                    target_positions=None,
                    source_epsilon=None,
                    target_epsilon=None,
                    source_lambda_par=None,
                    target_lambda_par=None,
                    num_rand_features=None,
                    dim=None,
                    source_integrator=None,
                    target_integrator=integrators[s],
                    max_iter=max_iter,
                    stopThr=stopThr,
                )
                for s in range(S)
            ]

        # T is N,ns
        if X.shape != Xprev.shape:
            err_feature = np.linalg.norm(X - Xprev.reshape(d, N))  # hack
        else:
            err_feature = np.linalg.norm(X - Xprev)
        err_structure = np.linalg.norm(C - Cprev)

        if log:
            log_["err_feature"].append(err_feature)
            log_["err_structure"].append(err_structure)
            log_["Ts_iter"].append(T)

        if verbose:
            if cpt % 200 == 0:
                print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
            print("{:5d}|{:8e}|".format(cpt, err_structure))
            print("{:5d}|{:8e}|".format(cpt, err_feature))

        cpt += 1
    if log:
        log_["T"] = T  # ce sont les matrices du barycentre de la target vers les Ys
        log_["p"] = p
        log_["Ms"] = Ms  # Ms sont de tailles N,ns

    if log:
        return X.T, C, log_
    else:
        return X.T, C
