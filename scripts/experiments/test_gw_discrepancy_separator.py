import os
import time
import random
import argparse

import numpy as np
import scipy as sp
import trimesh
import ot
import networkx as nx

from ega import default_trimesh_dataset_path
from ega.util.mesh_utils import trimesh_to_adjacency_matrices
from ega.algorithms.gromov_wasserstein_graphs import gromov_wasserstein_discrepancy

parser = argparse.ArgumentParser(
    description="Testing computations of Gromov Wasserstein discrepancy using the Separator"
)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument(
    "--n_samples", default=3000, type=int, help="Number of random samples to draw"
)
parser.add_argument(
    "--lambda_par",
    default=1.0,
    type=float,
    help="Lambda for the smoothening of the kernel matrix",
)
parser.add_argument(
    "--number_random_feats",
    default=16,
    type=int,
    help="Number of orthogonal random features, 16 and 32 works best for us. 64 produces bad results.",
)
parser.add_argument(
    "--threshold_nb_vertices",
    default=800,
    type=int,
    help="Understand when to use brute force vs separator",
)
parser.add_argument(
    "--unit_size", default=1.0, type=float, help="unit size for balanced separator"
)


def main():
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    # we load busted.STL datafile from default_trimesh_dataset_path.
    source_datapath = os.path.join(default_trimesh_dataset_path, "models/busted.STL")
    target_datapath = os.path.join(
        default_trimesh_dataset_path, "models/unit_sphere.STL"
    )
    if os.path.exists(source_datapath):
        source_mesh = trimesh.load(source_datapath)
    else:
        # in case you did not download trimesh data under this path, we also upload the datafile under
        # data/trimesh/models directory.
        source_mesh = trimesh.load(os.path.join("data/trimesh/models", "busted.STL"))

    if os.path.exists(target_datapath):
        target_mesh = trimesh.load(target_datapath)
    else:
        # in case you did not download trimesh data under this path, we also upload the datafile under
        # data/trimesh/models directory.
        target_mesh = trimesh.load(
            os.path.join("data/trimesh/models", "unit_sphere.STL")
        )

    source_adj = nx.adjacency_matrix(trimesh.graph.vertex_adjacency_graph(source_mesh))
    target_adj = nx.adjacency_matrix(trimesh.graph.vertex_adjacency_graph(target_mesh))
    Cs = np.exp(
        -(args.lambda_par * source_adj.todense() * 0.01)
    )  # to coincide with Han
    Ct = np.exp(
        -(args.lambda_par * target_adj.todense() * 0.01)
    )  # @Han why are we doing this? see line 113 and 119
    del source_adj, target_adj

    p = ot.unif(len(source_vertices))
    q = ot.unif(len(target_vertices))

    # the key hyperparameters of GW distance
    ot_dict = {
        "loss_type": "L2",
        "ot_method": "proximal",
        "beta": 0.2,
        "outer_iteration": 1000,  # outer, inner iteration, error bound of optimal transport
        "iter_bound": 1e-10,
        "inner_iteration": 2,
        "sk_bound": 1e-10,
        "node_prior": 10,
        "max_iter": 5,  # iteration and error bound for calcuating barycenter
        "cost_bound": 1e-16,
        "update_p": False,  # optional updates of source distribution
        "lr": 0,
        "alpha": 0,
    }

    # test baseline method
    time_s = time.time()
    T, d, _ = gromov_wasserstein_discrepancy(
        Cs, Ct, p.reshape(-1, 1), q.reshape(-1, 1), ot_dict
    )
    elapsed_time = time.time() - time_s
    del Cs, Ct

    # test ours
    # Define the necessary objects and parameters
    source_adjacency_lists = trimesh_to_adjacency_matrices(source_mesh)
    source_weights_lists = [[] for _ in range(len(source_adjacency_lists))]
    for i in range(len(source_adjacency_lists)):
        for j in range(len(source_adjacency_lists[i])):
            source_weights_lists[i].append(0.01)  # @Han, why do you do this?

    target_adjacency_lists = trimesh_to_adjacency_matrices(target_mesh)
    target_weights_lists = [[] for _ in range(len(target_adjacency_lists))]
    for i in range(len(target_adjacency_lists)):
        for j in range(len(target_adjacency_lists[i])):
            target_weights_lists[i].append(0.01)  # @Han, why do you do this?

    source_vertices = np.arange(len(source_adjacency_lists))
    target_vertices = np.arange(len(target_adjacency_lists))

    time_s1 = time.time()
    T1, d1, _ = gromov_wasserstein_discrepancy(
        cost_s=None,
        cost_t=None,
        p_s=p.reshape(-1, 1),
        p_t=q.reshape(-1, 1),
        ot_hyperpara=ot_dict,
        trans0=None,
        method_type="separator",
        source_positions=None,
        target_positions=None,
        source_epsilon=None,
        source_lambda_par=args.lambda_par,
        num_rand_features=args.number_random_feats,
        dim=None,
        target_epsilon=None,
        target_lambda_par=args.lambda_par,
        source_adjacency_lists=source_adjacency_lists,
        source_weights_lists=source_weights_lists,
        source_vertices=source_vertices,
        source_unit_size=args.unit_size,
        threshold_nb_vertices=args.threshold_nb_vertices,
        target_adjacency_lists=target_adjacency_lists,
        target_weights_lists=target_weights_lists,
        target_vertices=target_vertices,
        target_unit_size=args.unit_size,
    )
    elapsed_time1 = time.time() - time_s1

    print(f"Ground truth transport cost is {d}")
    print(f"Using fast matrix multiplication, truth transport cost is {d1}")
    print(f"Time taken for computing Ground truth transport cost is {elapsed_time}")
    print(f"Time taken for computing fast transport cost is {elapsed_time1}")
    print(f"Difference between the transport matrices are {((T-T1)**2).sum()}")


if __name__ == "__main__":
    main()
