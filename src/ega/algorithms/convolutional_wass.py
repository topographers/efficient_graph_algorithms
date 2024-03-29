from logging import raiseExceptions
import warnings
import math
import torch
import torch.nn.functional as F

small_constant = 1e-12


def convolve_images(imgs, kernel_x, kernel_y=None):
    """
    Simple convolution. First it blurs along x-axis and then y-axis
    """
    if kernel_y is None:
        kernel_y = kernel_x
    kx = torch.einsum("...ij,kjl->kil", kernel_x, imgs)
    kxy = torch.einsum("...ij,klj->kli", kernel_y, kx)
    return kxy


def convolve_clouds(cloud, kernel):
    """
    Convolution for 3d point clouds. First 1d convolution along x-axis, then y-axis and finally z-axis.
    """
    kx = torch.einsum("ij,rjlk->rilk", kernel, cloud)
    kxy = torch.einsum("ij,rkjl->rkil", kernel, kx)
    kxyz = torch.einsum("ij,rlkj->rlki", kernel, kxy)
    return kxyz


def convolutional_wasserstein_barycenter_2d(
    hist_images,
    reg,
    weights=None,
    max_iterations=10000,
    threshold=1e-4,
    conv_method="simple",
    conv_operator=None,
    iter_every=10,
    first_check=9,
):
    r"""Compute the entropic regularized wasserstein barycenter of distributions :math:`\mathbf{A}`
    where :math:`\mathbf{A}` is a collection of 2D images.
     The function solves the following optimization problem:
    .. math::
       \mathbf{a} = \mathop{\arg \min}_\mathbf{a} \quad \sum_i W_{reg}(\mathbf{a},\mathbf{a}_i)
    where :
    - :math:`W_{reg}(\cdot,\cdot)` is the entropic regularized Wasserstein
      distance (see :py:func:`ot.bregman.sinkhorn`)
    - :math:`\mathbf{a}_i` are training distributions (2D images) in the last two dimensions
      of matrix :math:`\mathbf{A}`
    - `reg` is the regularization strength scalar value

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm
    as proposed in :ref:`[21] <references-convolutional-barycenter-2d>`
    Parameters
    ----------
    hist_images : array-like, shape (n_hists, width, height)
        `n` distributions (2D images) of size `width` x `height`. Each image is considered a distribution over the pixels.
    reg : float
        Regularization term >0
    weights : array-like, shape (n_hists,)
        Weights of each image on the simplex (barycentric coodinates)
    max_iterations : int, optional
        Max number of iterations
    threshold : float, optional
        Stop threshold on error (> 0)
    conv_method : str, optional, default='simple'
        type of method to be used to compute the action of the kernel on a matrix
    conv_operator : Tuple(float, float) or float, optional.
        The matrix to be used for convolutions.
    iter_every : int, optional
        Check for errors in Sinkhorn convergence after every x number of iterations
    first_check : int, optional
        The first time the convergence for Sinkhorn iterations is tested
    Returns
    -------
    a : array-like, shape (width, height)
        2D Wasserstein barycenter

    .. _references-convolutional-barycenter-2d:
    References
    ----------
    .. [21] Solomon, J., De Goes, F., Peyré, G., Cuturi, M., Butscher,
        A., Nguyen, A. & Guibas, L. (2015).     Convolutional wasserstein distances:
        Efficient optimal transportation on geometric domains. ACM Transactions
        on Graphics (TOG), 34(4), 66
    .. [37] Janati, H., Cuturi, M., Gramfort, A. Proceedings of the 37th
        International Conference on Machine Learning, PMLR 119:4692-4701, 2020
    """

    if weights is None:
        weights = (torch.ones(hist_images.shape[0]) / hist_images.shape[0]).to(
            hist_images.device
        )
    else:
        assert len(weights) == hist_images.shape[0]

    bar = torch.ones(
        hist_images.shape[1:], dtype=hist_images.dtype, device=hist_images.device
    )
    bar /= torch.sum(bar)
    U = torch.ones(
        hist_images.shape, dtype=hist_images.dtype, device=hist_images.device
    )
    V = torch.ones(
        hist_images.shape, dtype=hist_images.dtype, device=hist_images.device
    )
    err = 1

    # build the convolution operator
    # this is equivalent to blurring on horizontal then vertical directions, unlike a traditional 2d Gaussian filter
    if conv_operator is None:
        if hist_images.shape[1] != hist_images.shape[2]:
            t = torch.linspace(0, 1, hist_images.shape[1]).to(hist_images.device)
            [Y, X] = torch.meshgrid(t, t)
            K1 = torch.exp(-((X - Y) ** 2) / reg)

            t = torch.linspace(0, 1, hist_images.shape[2]).to(hist_images.device)
            [Y, X] = torch.meshgrid(t, t)
            K2 = torch.exp(-((X - Y) ** 2) / reg)

        else:
            t = torch.linspace(0, 1, hist_images.shape[1]).to(hist_images.device)
            [Y, X] = torch.meshgrid(t, t)
            K1 = torch.exp(-((X - Y) ** 2) / reg)
    else:
        if isinstance(conv_operator, tuple):
            K1, K2 = conv_operator[0], conv_operator[1]
        else:
            K1, K2 = conv_operator

    if conv_method != "simple":
        raise NotImplementedError("Only simple 1D convolutions are implemeted.")

    if hist_images.shape[1] == hist_images.shape[2]:
        KU = convolve_images(U, K1, K1)
    else:
        KU = convolve_images(U, K1, K2)

    for ii in range(max_iterations):
        V = bar[None] / (KU + small_constant)
        if hist_images.shape[1] == hist_images.shape[2]:
            KV = convolve_images(V, K1, K1)
        else:
            KV = convolve_images(V, K1, K2)

        U = hist_images / (KV + small_constant)
        if hist_images.shape[1] == hist_images.shape[2]:
            KU = convolve_images(U, K1, K1)
        else:
            KU = convolve_images(U, K1, K2)
        bar = torch.exp(
            torch.sum(weights[:, None, None] * torch.log(KU + small_constant), dim=0)
        )
        if ii % iter_every == first_check:
            err = torch.sum(torch.std(V * KU, dim=0))

            if err < threshold:
                break
        return bar


def convolutional_wasserstein_barycenter_pt_cloud(
    hist_clouds,
    reg=None,
    weights=None,
    max_iterations=10000,
    threshold=1e-4,
    conv_method="simple",
    conv_operator=None,
    return_log=False,
):
    r"""Compute the entropic regularized wasserstein barycenter of distributions :math:`\mathbf{A}`
    where :math:`\mathbf{A}` is a collection of point clouds.
     The function solves the following optimization problem:
    .. math::
       \mathbf{a} = \mathop{\arg \min}_\mathbf{a} \quad \sum_i W_{reg}(\mathbf{a},\mathbf{a}_i)
    where :
    - :math:`W_{reg}(\cdot,\cdot)` is the entropic regularized Wasserstein
      distance (see :py:func:`ot.bregman.sinkhorn`)
    - :math:`\mathbf{a}_i` are training distributions (point clouds) in the last 3 dimensions
      of matrix :math:`\mathbf{A}`
    - `reg` is the regularization strength scalar value

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm
    as proposed in :ref:`[21] <references-convolutional-barycenter-2d>`
    Parameters
    ----------
    hist_clouds : array-like, shape (n_hists, width, height, depth)
        `n` distributions (3D point clouds) of size `width` x `height x depth`.  A cube, i.e [-1,1]^3 is binned into subcubes and the point cloud is placed in the cube.
    reg : float
        Regularization term >0
    weights : array-like, shape (n_hists,)
        Weights of each image on the simplex (barycentric coodinates)
    max_iterations : int, optional
        Max number of iterations
    threshold : float, optional
        Stop threshold on error (> 0)
    conv_method : str, optional, default='simple'
        type of method to be used to compute the action of the kernel on a matrix
    conv_operator : float, optional.
        The matrix to be used for convolutions.
    return_log : bool, optional
        Whether to return logs

    Returns
    -------
    a : array-like, shape (width, height, depth)
        3D Wasserstein barycenter

    log: bool optional
        records logs

    .. _references-convolutional-barycenter-3d:
    References
    ----------
    .. [21] Solomon, J., De Goes, F., Peyré, G., Cuturi, M., Butscher,
        A., Nguyen, A. & Guibas, L. (2015).     Convolutional wasserstein distances:
        Efficient optimal transportation on geometric domains. ACM Transactions
        on Graphics (TOG), 34(4), 66
    .. [37] Janati, H., Cuturi, M., Gramfort, A. Proceedings of the 37th
        International Conference on Machine Learning, PMLR 119:4692-4701, 2020
        Also : https://github.com/hichamjanati/debiased-ot-barycenters
    """

    if weights is None:
        weights = (torch.ones(hist_clouds.shape[0]) / hist_clouds.shape[0]).to(
            hist_clouds.device
        )
    else:
        assert len(weights) == hist_clouds.shape[0]

    if reg is None and conv_operator is None:
        raise ValueError(
            "At least one of the regularizer or conv_operator must be given"
        )

    n_hists, width, _, _ = hist_clouds.shape

    # build the convolution operator
    # this is equivalent to blurring on horizontal then vertical directions.
    if conv_operator is None:
        grid = torch.linspace(-1.0, 1.0, width)
        M = (grid[:, None] - grid[None, :]) ** 2
        conv_operator = torch.exp(-M / reg)

    b = torch.ones_like(hist_clouds, requires_grad=False)
    q = torch.ones(
        (width, width, width), device=hist_clouds.device, dtype=hist_clouds.dtype
    )

    if conv_method != "simple":
        raise NotImplementedError("Only 1D convolutions are supported")

    Kb = convolve_clouds(b, conv_operator)

    log = {"err": [], "a": [], "b": [], "q": []}
    err = 1

    for ii in range(max_iterations):
        if torch.isnan(q).any():
            break

        q_old = q.clone()
        a = hist_clouds / Kb
        Ka = convolve_clouds(a, conv_operator.t())
        q = torch.exp(
            torch.sum(
                weights[:, None, None, None] * torch.log(Ka + small_constant), dim=0
            )
        )
        Q = q[None, :]
        b = Q / Ka
        Kb = convolve_clouds(b, conv_operator)
        err = torch.abs(q - q_old).max()

        if err < threshold and ii > 10:
            break
    print("Barycenter 3d | err = ", err)

    if return_log:
        log["err"].append(err)
        log["a"] = a
        log["q"] = q
        log["b"] = b

    if ii == max_iterations - 1:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(err))

    if return_log:
        return q, log
    return q


############################################################################
# 1D conv wass barycenter. This is added purely for academic purposes
############################################################################


def create_gaussian_kernel_window(window_size, sigma=None) -> torch.Tensor:
    """
    Create 1D weights for a discrete gaussian kernel convolution.
    If sigma is not provided, it will be computed from the window size. Beware this is a a bad idea.
    """
    if not sigma:
        # Compute sigma in terms of pixels.
        # since sigma directly controls the error of the wasserstein approximations.
        # Expect this default sigma value to be blurry.
        sigmas_per_pixel = 1.5
        sigma = 0.5 * (window_size - 1) / sigmas_per_pixel

    gaussian_kernel = torch.tensor(
        [
            math.exp(-((x - 0.5 * (window_size - 1)) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ],
        dtype=torch.float,
        requires_grad=False,
    )
    gaussian_kernel = (gaussian_kernel / gaussian_kernel.sum()).unsqueeze(1)

    return gaussian_kernel


def wasserstein_distance(
    mu0, mu1, sigma, window_size, max_iter, threshold=1e-5, iter_every=10, first_check=9
):
    """
    Args:
    mu0, mu1 : n-array
        The source and target distributions
    sigma : float
        Controls the accuracy of the Sinkhorn iteration. Smaller is better but will require more iterations to converge
    window_size : int
        window_sizes for the 1d Gaussian kernel matrix
    max_iter : int
        Maximum number of Sinkhorn iterations
    threshold : float
        Stop threshold on error (> 0)
    iter_every : int, optional
        Check for errors in Sinkhorn convergence after every x number of iterations
    first_check : int, optional
        The first time the convergence for Sinkhorn iterations is tested

    Returns: Convolutional wasserstein distance between the two distributions
    """

    # TODO: This is a bit buggy code but essentially implements the algorithm as described in the paper.

    G = create_gaussian_kernel_window(window_size, sigma).to(mu0.device)
    G = G.reshape([1, 1, G.shape[0]])

    if window_size % 2 != 0:
        H = lambda x: F.conv1d(x, G, padding=window_size // 2)[0]
    else:
        raise NotImplementedError(
            "There is a shape mismatch error. The output shape is [1,n+1] for a given input tensor of shape [1,n] "
        )

    gamma = sigma**2

    err = 1
    a = 1.0 / mu0.shape[-1]

    v = torch.ones(mu0.shape).to(mu0.device)
    w = torch.ones(mu0.shape).to(mu0.device)

    for i in range(max_iter):

        v1 = v
        w1 = w
        v = mu0 / H(a * w)
        w = mu1 / H(a * v)

        d = gamma * (a * (mu0 * v.log() + mu1 * w.log())).sum()

        if i % iter_every == first_check:
            err = (v - v1).abs().sum(-1).mean() + (w - w1).abs().sum(-1).mean()

            if err.item() < threshold:
                break

    return gamma * (a * (mu0 * v.log() + mu1 * w.log())).sum()
