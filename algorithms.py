

import os 
import numpy as np 
import torch 
import random 
from torch import nn
from memory_profiler import profile


class BruteForce(nn.Module):
    """
    some definitions:
        n: number of points in the graph 
        d: feature vector dimension for each point in the graph 
        
    inputs: 
        f: a function that applies to each entry of the distance matrix. 
        x: n by d feature matrix. Each row represents the feature vector of a point in the graph. 
        
    class description: 
        this BruteForce class takes function f and feature matrix x, and outputs: Mx, 
        where M[i,j] = f(dist(i,j))
    """
    def __init__(self, device, evaluator):
        super(BruteForce, self).__init__()
        self.device = device
        self.evaluator = evaluator 

    @profile
    def forward(self, f, M, x):
        
        self.evaluator.start_time()
        
        Mx =  f(M) @ x  
        
        self.evaluator.stop_time() 
        self.evaluator.record_memory_consumption(Mx)
        
        return Mx 
    

    
    