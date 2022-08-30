from logging import raiseExceptions
import math
import torch
import torch.nn.functional as F

small_constant = 1e-9


def convolutional_wasserstein_barycenter_2d(
    A, reg, weights=None, numItermax=10000, stopThr=1e-4
):
    r"""Compute the entropic regularized wasserstein barycenter of distributions :math:`\mathbf{A}`
    where :math:`\mathbf{A}` is a collection of 2D images.
     The function solves the following optimization problem:
    .. math::
       \mathbf{a} = \mathop{\arg \min}_\mathbf{a} \quad \sum_i W_{reg}(\mathbf{a},\mathbf{a}_i)
    where :
    - :math:`W_{reg}(\cdot,\cdot)` is the entropic regularized Wasserstein
      distance (see :py:func:`ot.bregman.sinkhorn`)
    - :math:`\mathbf{a}_i` are training distributions (2D images) in the mast two dimensions
      of matrix :math:`\mathbf{A}`
    - `reg` is the regularization strength scalar value
    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm
    as proposed in :ref:`[21] <references-convolutional-barycenter-2d>`
    Parameters
    ----------
    A : array-like, shape (n_hists, width, height)
        `n` distributions (2D images) of size `width` x `height`. Each image is considered a distribution over the pixels.
    reg : float
        Regularization term >0
    weights : array-like, shape (n_hists,)
        Weights of each image on the simplex (barycentric coodinates)
    method : string, optional
        method used for the solver either 'sinkhorn' or 'sinkhorn_log'
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (> 0)
    stabThr : float, optional
        Stabilization threshold to avoid numerical precision issue

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
        weights = (torch.ones(A.shape[0]) / A.shape[0]).to(A.device)
    else:
        assert len(weights) == A.shape[0]

    bar = torch.ones(A.shape[1:], dtype=A.dtype, device=A.device)
    bar /= torch.sum(bar)
    U = torch.ones(A.shape, dtype=A.dtype, device=A.device)
    V = torch.ones(A.shape, dtype=A.dtype, device=A.device)
    err = 1

    # build the convolution operator
    # this is equivalent to blurring on horizontal then vertical directions, unlike a traditional 2d Gaussian filter
    t = torch.linspace(0, 1, A.shape[1]).to(A.device)
    [Y, X] = torch.meshgrid(t, t)
    K1 = torch.exp(-((X - Y) ** 2) / reg)

    t = torch.linspace(0, 1, A.shape[2]).to(A.device)
    [Y, X] = torch.meshgrid(t, t)
    K2 = torch.exp(-((X - Y) ** 2) / reg)

    def convol_imgs(imgs):
        kx = torch.einsum("...ij,kjl->kil", K1, imgs)
        kxy = torch.einsum("...ij,klj->kli", K2, kx)
        return kxy

    KU = convol_imgs(U)
    for ii in range(numItermax):
        V = bar[None] / (KU + small_constant)
        KV = convol_imgs(V)
        U = A / (KV + small_constant)
        KU = convol_imgs(U)
        bar = torch.exp(
            torch.sum(weights[:, None, None] * torch.log(KU + stopThr), dim=0)
        )
        if ii % 10 == 9:
            err = torch.sum(torch.std(V * KU, dim=0))

            if err < stopThr:
                break
        return bar


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
        sigma = 2.5
        sigma = 0.5 * (window_size - 1) / sigmas_per_pixel

    gauss_ker = torch.tensor(
        [
            math.exp(-((x - 0.5 * (window_size - 1)) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ],
        dtype=torch.float,
        requires_grad=False,
    )
    gauss_ker = (gauss_ker / gauss_ker.sum()).unsqueeze(1)

    return gauss_ker


def wasserstein_distance(mu0, mu1, sigma, window_size, max_iter, threshold=1e-5):
    """
    mu0, mu1: The source and target distributions
    sigma (float) : controls the accuracy of the Sinkhorn iteration. Smaller is better but will require more iterations to converge
    window_size (int) : window_sizes for the 1d Gaussian kernel matrix
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

        if i % 10 == 9:
            err = (v - v1).abs().sum(-1).mean() + (w - w1).abs().sum(-1).mean()

            if err.item() < threshold:
                break

    return gamma * (a * (mu0 * v.log() + mu1 * w.log())).sum()
