import numpy as np
from scipy.optimize.linesearch import scalar_search_armijo
from ot.lp import emd


class StopError(Exception):
    pass


class NonConvergenceError(Exception):
    pass


def solve_1d_linesearch_quad_funct(a, b, c):
    # solve min f(x)=a*x**2+b*x+c sur 0,1
    f0 = c
    df0 = b
    f1 = a + f0 + df0

    if a > 0:  # convex
        minimum = min(1, max(0, -b / (2 * a)))
        # print('entrelesdeux')
        return minimum
    else:  # non convexe donc sur les coins
        if f0 > f1:
            # print('sur1 f(1)={}'.format(f(1)))
            return 1
        else:
            # print('sur0 f(0)={}'.format(f(0)))
            return 0


def line_search_armijo(
    f,
    xk,
    pk,
    gfk,
    old_fval,
    args=(),
    c1=1e-4,
    alpha0=0.99,
    alpha_min=None,
    alpha_max=None,
):
    """
    Armijo linesearch function that works with matrices
    find an approximate minimum of f(xk+alpha*pk) that satifies the
    armijo conditions.
    Parameters
    ----------
    f : function
        loss function
    xk : np.ndarray
        initial position
    pk : np.ndarray
        descent direction
    gfk : np.ndarray
        gradient of f at xk
    old_fval : float
        loss value at xk
    args : tuple, optional
        arguments given to f
    c1 : float, optional
        c1 const in armijo rule (>0)
    alpha0 : float, optional
        initial step (>0)
    alpha_min : float, optional
        minimum value for alpha
    alpha_max : float, optional
        maximum value for alpha
    Returns
    -------
    alpha : float
        step that satisfy armijo conditions
    fc : int
        nb of function call
    fa : float
        loss value at step alpha
    """
    xk = np.atleast_1d(xk)
    fc = [0]

    def phi(alpha1):
        fc[0] += 1
        return f(xk + alpha1 * pk, *args)

    if old_fval is None:
        phi0 = phi(0.0)
    else:
        phi0 = old_fval

    derphi0 = np.sum(pk * gfk)  # Quickfix for matrices
    alpha, phi1 = scalar_search_armijo(phi, phi0, derphi0, c1=c1, alpha0=alpha0)

    if alpha is None:
        return 0.0, fc[0], phi0
    else:
        if alpha_min is not None or alpha_max is not None:
            alpha = np.clip(alpha, alpha_min, alpha_max)
        return float(alpha), fc[0], phi1


def do_linesearch(
    cost,
    G,
    deltaG,
    Mi,
    f_val,
    amijo=True,
    C1=None,
    C2=None,
    reg=None,
    Gc=None,
    constC=None,
    M=None,
    alpha_min=None,
    alpha_max=None,
    method_type=None,
    source_integrator=None,
    target_integrator=None,
):

    """
    Solve the linesearch in the FW iterations
    Parameters
    ----------
    cost : method
        The FGW cost
    G : ndarray, shape(ns,nt)
        The transport map at a given iteration of the FW
    deltaG : ndarray (ns,nt)
        Difference between the optimal map found by linearization in the FW algorithm and the value at a given iteration
    Mi : ndarray (ns,nt)
        Cost matrix of the linearized transport problem. Corresponds to the gradient of the cost
    f_val :  float
        Value of the cost at G
    amijo : bool, optionnal
            If True the steps of the line-search is found via an amijo research. Else closed form is used.
            If there is convergence issues use False.
    C1 : ndarray (ns,ns), optionnal
        Structure matrix in the source domain. Only used when amijo=False
    C2 : ndarray (nt,nt), optionnal
        Structure matrix in the target domain. Only used when amijo=False
    reg : float, optionnal
          Regularization parameter. Corresponds to the alpha parameter of FGW. Only used when amijo=False
    Gc : ndarray (ns,nt)
        Optimal map found by linearization in the FW algorithm. Only used when amijo=False
    constC : ndarray (ns,nt)
             Constant for the gromov cost. See [3]. Only used when amijo=False
    M : ndarray (ns,nt), optional
        Cost matrix between the features. Only used when amijo=False,
    Optional:
    method_type : str None defaults to brute force
    source_integrator : Callable function that does fast matrix multplication for source graph
    target_integrator : Callable function that does fast matrix multplication for target graph
    Returns
    -------
    alpha : float
            The optimal step size of the FW
    fc : useless here
    f_val :  float
             The value of the cost for the next iteration
    References
    ----------
    .. [3] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """
    if amijo:
        alpha, fc, f_val = line_search_armijo(
            cost, G, deltaG, Mi, f_val, alpha_min=alpha_min, alpha_max=alpha_max
        )
    else:  # need sym matrices
        if method_type is None:
            dot = np.dot(np.dot(C1, deltaG), C2)
            a = (
                -2 * reg * np.sum(dot * deltaG)
            )  # -2*alpha*<C1 dt C2,dt> si qqlun est pas bon c'est lui
            b = np.sum((M + reg * constC) * deltaG) - 2 * reg * (
                np.sum(dot * G) + np.sum(np.dot(np.dot(C1, G), C2) * deltaG)
            )
            c = cost(G)  # f(xt)

        else:
            partial_dcost = source_integrator.integrate_graph_field(deltaG)
            dot = (
                target_integrator.integrate_graph_field(partial_dcost.T)
            ).T  # use symmetry here
            del partial_dcost
            a = (
                -2 * reg * np.sum(dot * deltaG)
            )  # -2*alpha*<C1 dt C2,dt> si qqlun est pas bon c'est lui
            partial_cost = source_integrator.integrate_graph_field(G)
            b1 = (target_integrator.integrate_graph_field(partial_cost.T)).T
            del partial_cost
            b = np.sum((M + reg * constC) * deltaG) - 2 * reg * (
                np.sum(dot * G) + np.sum(b1 * deltaG)
            )
            del b1
            c = cost(G)

        alpha = solve_1d_linesearch_quad_funct(a, b, c)
        if alpha_min is not None or alpha_max is not None:
            alpha = np.clip(alpha, alpha_min, alpha_max)
        fc = None
        f_val = cost(G + alpha * deltaG)

    return alpha, fc, f_val


def cg(
    a,
    b,
    M,
    reg,
    f,
    df,
    G0=None,
    numItermax=500,
    numItermaxEmd=100000,
    stopThr=1e-09,
    stopThr2=1e-9,
    verbose=False,
    log=False,
    amijo=True,
    C1=None,
    C2=None,
    constC=None,
    alpha_min=0.0,
    alpha_max=1.0,
    method_type=None,
    source_integrator=None,
    target_integrator=None,
):
    """
    Solve the general regularized OT problem with conditional gradient
        The function solves the following optimization problem:
    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg*f(\gamma)
        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma\geq 0
    where :
    - M is the (ns,nt) metric cost matrix
    - :math:`f` is the regularization term ( and df is its gradient)
    - a and b are source and target weights (sum to 1)
    The algorithm used for solving the problem is conditional gradient as discussed in  [1]_
    Parameters
    ----------
    a : np.ndarray (ns,)
        samples weights in the source domain
    b : np.ndarray (nt,)
        samples in the target domain
    M : np.ndarray (ns,nt)
        loss matrix
    reg : float
        Regularization term >0
    G0 :  np.ndarray (ns,nt), optional
        initial guess (default is indep joint density)
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    Optional:
    method_type : str None defaults to brute force
    source_integrator : Callable function that does fast matrix multplication for source graph
    target_integrator : Callable function that does fast matrix multplication for target graph
    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters
    References
    ----------
    .. [1] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014). Regularized discrete optimal transport. SIAM Journal on Imaging Sciences, 7(3), 1853-1882.
    See Also
    --------
    ot.lp.emd : Unregularized optimal ransport
    ot.bregman.sinkhorn : Entropic regularized optimal transport
    """

    loop = 1

    if log:
        log = {"loss": [], "delta_fval": []}

    if G0 is None:
        G = np.outer(a, b)
    else:
        G = G0

    def cost(G):
        return np.sum(M * G) + reg * f(G)

    f_val = cost(G)  # f(xt)

    if log:
        log["loss"].append(f_val)

    it = 0

    if verbose:
        print(
            "{:5s}|{:12s}|{:8s}".format("It.", "Loss", "Delta loss") + "\n" + "-" * 32
        )
        print("{:5d}|{:8e}|{:8e}".format(it, f_val, 0))

    while loop:

        it += 1
        old_fval = f_val
        # G=xt
        # problem linearization
        Mi = M + reg * df(G)  # Gradient(xt)
        # set M positive
        Mi += Mi.min()

        # solve linear program
        Gc, logemd = emd(a, b, Mi, numItermax=numItermaxEmd, log=True)  # st

        deltaG = Gc - G  # dt

        # argmin_alpha f(xt+alpha dt)
        alpha, fc, f_val = do_linesearch(
            cost=cost,
            G=G,
            deltaG=deltaG,
            Mi=Mi,
            f_val=f_val,
            amijo=amijo,
            constC=constC,
            C1=C1,
            C2=C2,
            reg=reg,
            Gc=Gc,
            M=M,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            method_type=method_type,
            source_integrator=source_integrator,
            target_integrator=target_integrator,
        )

        if alpha is None or np.isnan(alpha):
            raise NonConvergenceError("Alpha is not found")
        else:
            G = G + alpha * deltaG  # xt+1=xt +alpha dt

        # test convergence
        if it >= numItermax:
            loop = 0

        delta_fval = f_val - old_fval
        abs_delta_fval = abs(f_val - old_fval)

        relative_delta_fval = abs_delta_fval / abs(f_val)
        if relative_delta_fval < stopThr or abs_delta_fval < stopThr2:
            loop = 0

        if log:
            log["loss"].append(f_val)
            log["delta_fval"].append(delta_fval)

        if verbose:
            if it % 20 == 0:
                print(
                    "{:5s}|{:12s}|{:8s}".format("It.", "Loss", "Delta loss")
                    + "\n"
                    + "-" * 32
                )
            print("{:5d}|{:8e}|{:8e}|{:5e}".format(it, f_val, delta_fval, alpha))

    if log:
        log.update(logemd)
        return G, log
    else:
        return G
