import torch

small_constant = 1e-9

def convolutional_wasserstein_barycenter_2d(A, reg, weights=None, numItermax=10000,
                               stopThr=1e-4):
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
        assert (len(weights) == A.shape[0])


    bar = torch.ones(A.shape[1:], dtype=A.dtype, device=A.device)
    bar /= torch.sum(bar)
    U = torch.ones(A.shape, dtype=A.dtype, device=A.device)
    V = torch.ones(A.shape, dtype=A.dtype, device=A.device)
    err = 1

    # build the convolution operator
    # this is equivalent to blurring on horizontal then vertical directions, unlike a traditional 2d Gaussian filter
    t = torch.linspace(0, 1, A.shape[1]).to(A.device)
    [Y, X] = torch.meshgrid(t, t)
    K1 = torch.exp(-(X - Y) ** 2 / reg)

    t = torch.linspace(0, 1, A.shape[2]).to(A.device)
    [Y, X] = torch.meshgrid(t, t)
    K2 = torch.exp(-(X - Y) ** 2 / reg)

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

###########################################
# 1D conv wass barycenter
############################################
