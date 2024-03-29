#import torch
import numpy as np


class GaussianKernel(object):
    """
    If the input M is a matrix, this function calculates exp( - sigma * dist(i,j)) for each (i,j)^th entry of matrix
    M elementwise
    """

    def __init__(self, sigma):
        super(GaussianKernel, self).__init__()
        self.sigma = sigma

    def __call__(self, M):
        #return torch.exp(- self.sigma * M)
        return np.exp(-self.sigma*M)
