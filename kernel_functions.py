
import argparse
import trimesh 
import os 
import numpy as np 
import torch 
from distutils.util import strtobool
import random 
import open3d as o3d
from torch import nn


class GaussianKernel():
    """
    If the input M is a matrix, this function calculates exp( - sigma * dist(i,j)) for each (i,j)^th entry of matrix M elementwise 
    
    """
    def __init__(self, sigma):
        super(GaussianKernel, self).__init__() 
        self.sigma = sigma 
        
    def __call__(self, M):
        return torch.exp( - self.sigma * M)
        




if __name__ == '__main__':

    f = GaussianKernel(0.1)


    
    