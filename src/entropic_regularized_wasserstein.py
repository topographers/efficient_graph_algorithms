import torch


class SinkhornDistance(torch.nn.Module):
    """
    Algorithm follows the famous work of Cuturi : https://papers.nips.cc/paper/2013/file/af21d0c97db2e27e13572cbf59eb343d-Paper.pdf
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
        cost matrix (array, optional) : (N x N) matrix of pairwise distances
        threshold (float) : some small conatnt for Sinkhorn iterations to converge if the error falls below this number
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(
        self, eps, max_iter, reduction="none", cost_matrix=None, threshold=1e-4
    ):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.cost_matrix = cost_matrix
        self.threshold = threshold

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        if self.cost_matrix is None:
            C = self._cost_matrix(x, y)  # Wasserstein cost function
        else:
            C = self.cost_matrix

        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = (
            torch.empty(batch_size, x_points, dtype=torch.float, requires_grad=False)
            .fill_(1.0 / x_points)
            .to(x.device)
            .squeeze()
        )
        nu = (
            torch.empty(batch_size, y_points, dtype=torch.float, requires_grad=False)
            .fill_(1.0 / y_points)
            .to(x.device)
            .squeeze()
        )

        u = torch.zeros_like(mu).to(x.device)
        v = torch.zeros_like(nu).to(x.device)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = (
                self.eps
                * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1))
                + u
            )  # TODO: add this in the code instead of hardcoding it like this
            v = (
                self.eps
                * (
                    torch.log(nu + 1e-8)
                    - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)
                )
                + v
            )
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < self.threshold:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == "mean":
            cost = cost.mean()
        elif self.reduction == "sum":
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        """
        Modified cost for logarithmic updates
        $M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$
        Provides more numerical stability.
        """
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        """
        Returns the matrix of $|x_i-y_j|^p$. Simple L^p distance but can be modified by more complicated distances.
        """
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, v, tau):
        """
        Barycenter subroutine, used by kinetic acceleration through extrapolation.
        """
        return tau * u + (1 - tau) * v
