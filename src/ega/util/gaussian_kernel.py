import torch


class GaussianKernel(object):
    """
    If the input M is a matrix, this function calculates exp( - sigma * dist(i,j)) for each (i,j)^th entry of matrix
    M elementwise
    """

    def __init__(self, sigma):
        super(GaussianKernel, self).__init__()
        self.sigma = sigma

    def __call__(self, M):
        return torch.exp(- self.sigma * M)


def main():
    f = GaussianKernel(0.1)


if __name__ == '__main__':
    main()
