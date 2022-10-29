import networkx as nx
from ega.algorithms.graph_kernels import GraphKernel
import time

# TODO: TEST SPEED AND ACCURACY WITH https://github.com/BorgwardtLab/GraphKernels
# create a networkx graph
g = nx.erdos_renyi_graph(500, 0.2, seed=42, directed=False)
gk = GraphKernel(g, verbose_level=1)  # instantiates the kernel class

## generate kernels for "Random Walk", "Random Walk with Restart", "Diffusion State", "Heat", "ICN"


start_time = time.time()
rw = gk.eval_random_walk_kernel(2)  # 2 random walks
elapsed = time.time() - start_time
print("Time to generate Random Walk", elapsed)
##############################

start_time = time.time()
rwr = gk.eval_random_walk_with_restart_kernel(0.3)  # restart prob=.3
elapsed = time.time() - start_time
print("Time to generate Random Walk with Restart", elapsed)
##############

start_time = time.time()
dsd = gk.eval_diffusion_state_distance(3)  # number of random walks
elapsed = time.time() - start_time
print("Time to generate Diffusion state kernel", elapsed)
######################

start_time = time.time()
ht = gk.eval_heat_kernel(0.7)  # time step .7, can be greater than 1
elapsed = time.time() - start_time
print("Time to generate heat kernel", elapsed)
######################

start_time = time.time()
icn = gk.eval_interconnected_kernel()
elapsed = time.time() - start_time
print("Time to generate ICN", elapsed)
##########################

print(gk.kernels)  # get a dictionary of the computed kernels.
# the kernels are cached for later use.
