import os
import time
import random
import argparse

import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist
import trimesh
import ot
import networkx as nx
from scipy.sparse.csgraph import shortest_path

from ega import default_trimesh_dataset_path
from ega.util.mesh_utils import trimesh_to_adjacency_matrices
from ega.algorithms.fused_gromov_wasserstein import fgw_lp

parser = argparse.ArgumentParser(
    description="Testing computations of Gromov Wasserstein (POT) using the Separator"
)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument(
    "--lambda_par",
    default=-.01,
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
parser.add_argument(
    "--edge_weight",
    default=1.0,
    type=float,
    help="Edge weights for the adjacency matrices",
)
parser.add_argument(
    "--use_armijo",
    default=True,
    type=bool,
    help="Whether to armijo line search algorithms. Turn it off if there are numerical instabilities",
)


def main():
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    def hamming_dist(x,y):
        return len([i for i, j in zip(x, y) if i != j]) 

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

    source_graph = trimesh.graph.vertex_adjacency_graph(source_mesh)
    if args.edge_weight != 1.0:
        nx.set_edge_attributes(source_graph, values=args.edge_weight, name="weight")
    source_adj = nx.adjacency_matrix(source_graph)
    del source_graph

    target_graph = trimesh.graph.vertex_adjacency_graph(target_mesh)
    if args.edge_weight != 1.0:
        nx.set_edge_attributes(target_graph, values=args.edge_weight, name="weight")
    target_adj = nx.adjacency_matrix(target_graph)
    del target_graph

    # randomly label the vertices of source and target graphs 
    ys = (np.random.randint(2, size=source_adj.shape[0])).reshape(-1,1)
    yt = np.random.randint(2, size=target_adj.shape[0])
    np.random.shuffle(yt)
    yt = yt.reshape(-1,1)

    if args.features_metric=='dirac':
        f=lambda x,y: x!=y
        M=cdist(ys,yt,metric=f)
    elif args.features_metric=='hamming_dist': 
        f=lambda x,y: hamming_dist(x,y)
        M=cdist(yt,ys,metric=f)
    else: 
        raise NotImplementedError("Other feature distance metrics are not implemented yet.")


    Cs = shortest_path(csgraph=source_adj, directed=False)
    Cs = np.exp(-args.lambda_par * Cs)
    Cs = (Cs + Cs.T) / 2.0  # symmetrize
    Ct = shortest_path(csgraph=target_adj, directed=False)
    Ct = np.exp(-args.lambda_par * Ct)
    Ct = (Ct + Ct.T) / 2.0  # symmetrize
    del source_adj, target_adj

    p = ot.unif(Cs.shape[0])
    q = ot.unif(Ct.shape[0])

    if args.features_metric=='dirac':
        f=lambda x,y: x!=y
        M=cdist(ys,yt,metric=f)
    elif args.features_metric=='hamming_dist': 
        f=lambda x,y: hamming_dist(x,y)
        M=cdist(yt,ys,metric=f)
    else: 
        raise NotImplementedError("Other feature distance metrics are not implemented yet.")

    # test baseline
    time_s = time.time()
    trans0, log0 = fgw_lp(
        M=M,
        C1=Cs,
        C2=Ct,
        p=p,
        q=q,
        loss_fun="square_loss",
        alpha=0.5,
        armijo=args.use_armijo,
        G0=None,
        log=True,
    )
    elapsed_time = time.time() - time_s

    # test ours
    # Define the necessary objects and parameters
    source_adjacency_lists = trimesh_to_adjacency_matrices(source_mesh)
    source_weights_lists = [[] for _ in range(len(source_adjacency_lists))]
    for i in range(len(source_adjacency_lists)):
        for j in range(len(source_adjacency_lists[i])):
            source_weights_lists[i].append(args.edge_weight)

    target_adjacency_lists = trimesh_to_adjacency_matrices(target_mesh)
    target_weights_lists = [[] for _ in range(len(target_adjacency_lists))]
    for i in range(len(target_adjacency_lists)):
        for j in range(len(target_adjacency_lists[i])):
            target_weights_lists[i].append(args.edge_weight)

    source_vertices = np.arange(len(source_adjacency_lists))
    target_vertices = np.arange(len(target_adjacency_lists))
    time1 = time.time()
    trans1, log1 = fgw_lp(
        M=M,
        C1=None,
        C2=None,
        p=p,
        q=q,
        loss_fun="square_loss",
        alpha=0.5,
        armijo=args.use_armijo,
        G0=None,
        log=True,
        method_type="separator",
        source_positions=None,
        target_positions=None,
        source_epsilon=args.epsilon,
        target_epsilon=args.epsilon,
        source_lambda_par=args.lambda_par,
        target_lambda_par=args.lambda_par,
        num_rand_features=args.number_random_feats,
        dim=None,
        source_adjacency_lists=source_adjacency_lists,
        source_weights_lists=source_weights_lists,
        source_vertices=source_vertices,
        source_unit_size=args.unit_size,
        threshold_nb_vertices=args.threshold_nb_vertices,
        target_adjacency_lists=target_adjacency_lists,
        target_weights_lists=target_weights_lists,
        target_vertices=target_vertices,
        target_unit_size=args.unit_size,
        verbose=False,
    )
    elapsed_time1 = time.time() - time1

    print(f"Ground truth transport cost is {log0['fgw_dist']}")
    print(f"Using fast matrix multiplication, transport cost is {log1['fgw_dist']}")
    print(f"Time taken for computing Ground truth transport cost is {elapsed_time}")
    print(f"Time taken for computing fast transport cost is {elapsed_time1}")
    print(f"Difference between the transport matrices are {((trans0-trans1)**2).sum()}")


if __name__ == "__main__":
    main()