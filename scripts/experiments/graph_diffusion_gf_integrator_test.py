#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 21:55:51 2022

@author: hanlin
"""

import time 
from ega.algorithms.graph_diffusion_gf_integrator import DFGFIntegrator
import numpy as np
from ega.util.mesh_utils import random_projection_creator, density_function, fourier_transform


def main():    
    """
    the following example is from GraphFieldIntegrator.ipynb
    """
        
    number_points = 100
    dim = 3
    epsilon = 0.1
    lambda_par = 0.1
    num_rand_features = 32
    field_dim = 5
    
    positions = np.random.normal(size=(number_points, dim))
    field = np.random.normal(size=(number_points, field_dim))
    
    # CONSTRUCT THE OBJECT OF THE CLASS
    start = time.time()
    dfgf_integrator = DFGFIntegrator(positions,  epsilon, lambda_par, 
                                     num_rand_features, dim, 
                                     random_projection_creator, density_function, 
                                     fourier_transform)
    end = time.time()
    print("Constructor for Graph Diffusion (DF) GF Integrator: ", end - start)
    
    # INTEGRATE GRAPH FIELD
    start = time.time()
    result_df = dfgf_integrator.integrate_graph_field(field)
    end = time.time()
    print("Graph field integration for Graph Diffusion (DF) GF Integrator: ", end - start)


if __name__ == '__main__':
    main()
   