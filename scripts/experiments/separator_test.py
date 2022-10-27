import numpy as np 
import time 
import trimesh 
import os 

from ega.algorithms.brute_force import BFGFIntegrator
from ega.algorithms.separation_gf_integrator import PlaceboSeparationGFIntegrator, SeparationGFIntegrator
from ega.util.mesh_utils import trimesh_to_adjacency_matrices 
from ega import default_trimesh_dataset_path


def main():    
    
    """
    the following example is from GraphFieldIntegrator.ipynb
    """
    
    # we load busted.STL datafile from default_trimesh_dataset_path. 
    trimesh_datapath = os.path.join(default_trimesh_dataset_path,'models/busted.STL')
    if os.path.exists(trimesh_datapath):
        mesh = trimesh.load(trimesh_datapath)
    else: 
        # in case you did not download trimesh data under this path, we also upload the datafile under data/trimesh/models directory.
        mesh = trimesh.load(os.path.join("data/trimesh/models", "busted.STL"))
        
        
    adjacency_lists = trimesh_to_adjacency_matrices(mesh)
    weights_lists = [[] for _ in range(len(adjacency_lists))]
    for i in range(len(adjacency_lists)):
        for j in range(len(adjacency_lists[i])):
            weights_lists[i].append(0.01)
    print(len(adjacency_lists))
    
    
    # GENERAL PARAMETERS
    lambda_par = 1.0
    feature_dim = 10
    field = np.ones((len(adjacency_lists), feature_dim))
    f_fun = lambda x: np.exp(-lambda_par * x)
    vertices = np.arange(len(adjacency_lists))
    
    
    # SOME OF THE PARAMETERS OF THE BALANCED SEPARATOR APPROACH
    laplace = True
    ##### parameters to be tuned #####
    threshold_nb_vertices = 800  # controls the threshold to apply brute force integrator
    unit_size = 1.0  # controls the bucket size for approximation
    ##########
    
    
    # PREPROCESSING
    start = time.time()
    sgf_integrator = SeparationGFIntegrator(adjacency_lists, weights_lists, vertices, f_fun, 
                                            unit_size=unit_size, threshold_nb_vertices=threshold_nb_vertices)
    end = time.time()
    print("Constructor for Balanced separator (BS) takes time: ", end - start)
    start = time.time()
    bfgf_integrator = BFGFIntegrator(adjacency_lists, weights_lists, vertices, f_fun)
    end = time.time()
    print("Constructor for Brute-force (BF) takes time: ", end - start)


    # Seperator Integrator 
    start = time.time()
    result = sgf_integrator.integrate_graph_field(field)
    end = time.time()
    print("Graph field integration for Balanced Separator (BS) takes time: ", end - start)
    
    # Brute Force Integrator
    start = time.time()
    result_bf = bfgf_integrator.integrate_graph_field(field)
    end = time.time()
    print("Graph field integration for Brute-force (BF) takes time: ", end - start)


if __name__ == '__main__':
    main()
   


