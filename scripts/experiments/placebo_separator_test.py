import numpy as np 
import time 

from ega.algorithms.brute_force import BFGFIntegrator
from ega.algorithms.separation_gf_integrator import PlaceboSeparationGFIntegrator



def main():    
    
    """
    the following example is from GraphFieldIntegrator.ipynb
    """
    # GENERAL PARAMETERS
    nb_vertices = 8000
    threshold_nb_vertices = 500
    nb_clusters = 2
    lambda_par = 1.0
    feature_dim = 10
    field = np.ones((nb_vertices, feature_dim))
    f_fun = lambda x: np.exp(-lambda_par * x)
    adjacency_lists = [[np.arange(nb_vertices)] for _ in range(nb_vertices)]  
    weights_lists = [np.ones(nb_vertices) for _ in range(nb_vertices)] 
    vertices = np.arange(nb_vertices)
    
    # SOME OF THE PARAMETERS OF THE BALANCED SEPARATOR APPROACH
    unit_size = 1.0
    laplace = True
    
    # PREPROCESSING (this step might need some time)
    psgf_integrator = PlaceboSeparationGFIntegrator(adjacency_lists, weights_lists, vertices, f_fun)
    bfgf_integrator = BFGFIntegrator(adjacency_lists, weights_lists, vertices, f_fun)
    
    # Placebo Seperator Integrator 
    start = time.time()
    result = psgf_integrator.integrate_graph_field(field)
    end = time.time()
    print("Processing time for the BS approach is: ", end - start)
    
    # Brute Force Integrator
    start = time.time()
    result_bf = bfgf_integrator.integrate_graph_field(field)
    end = time.time()
    print("Processing time for the BF approach is: ", end - start)


if __name__ == '__main__':
    main()
   


