import scipy as sp
import numpy as np
import ot
import random
import time
import argparse
from ega.algorithms.gromov_wasserstein_graphs import gromov_wasserstein_discrepancy

parser = argparse.ArgumentParser(
    description="Testing computations of Gromov Wasserstein discrepancy"
)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument(
    "--n_samples", default=3000, type=int, help="Number of random samples to draw"
)
parser.add_argument(
    "--lambda_par",
    default=1e-5,
    type=float,
    help="Lambda for the smoothening of the kernel matrix",
)
parser.add_argument(
    "--epsilon", default=0.1, type=float, help="Neighborhood distance around a point"
)
parser.add_argument(
    "--number_random_feats",
    default=32,
    type=int,
    help="Number of orthogonal random features, 16 and 32 works best for us. 64 produces bad results.",
)


def main():
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    ## generate data
    mu_s = np.array([0, 0, 0])
    r_s = np.random.rand(3, 3)
    cov_s = np.matmul(r_s, r_s.T)

    mu_t = np.array([4, 4, 4])
    r_t = np.random.rand(3, 3)
    cov_t = np.matmul(r_t, r_t.T)

    Q = sp.linalg.sqrtm(cov_s)
    xs = np.random.randn(args.n_samples, 3).dot(Q) + mu_s
    P = sp.linalg.sqrtm(cov_t)
    xt = np.random.randn(args.n_samples, 3).dot(P) + mu_t

    # Replace xs and xt with points from meshes

    ### construct distance matrices
    Cs = sp.spatial.distance.cdist(xs, xs, "minkowski", p=1)
    Ct = sp.spatial.distance.cdist(xt, xt, "minkowski", p=1)

    # Replace with length of the point clouds
    p = ot.unif(args.n_samples)
    q = ot.unif(args.n_samples)

    ## sparsify the matrices
    Cs[Cs > args.epsilon] = 0
    Ct[Ct > args.epsilon] = 0

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
    Cs1 = sp.linalg.expm(args.lambda_par * Cs)
    Ct1 = sp.linalg.expm(args.lambda_par * Ct)
    T, d, _ = gromov_wasserstein_discrepancy(
        Cs1, Ct1, p.reshape(-1, 1), q.reshape(-1, 1), ot_dict
    )
    elapsed_time = time.time() - time_s
    del Cs1, Ct1

    # test ours
    time_s1 = time.time()
    T1, d1, _ = gromov_wasserstein_discrepancy(
        cost_s=None,
        cost_t=None,
        p_s=p.reshape(-1, 1),
        p_t=q.reshape(-1, 1),
        ot_hyperpara=ot_dict,
        trans0=None,
        method_type="diffusion",
        source_positions=xs,
        target_positions=xt,
        source_epsilon=args.epsilon,
        source_lambda_par=args.lambda_par,
        num_rand_features=args.number_random_feats,
        dim=3,
        target_epsilon=args.epsilon,
        target_lambda_par=args.lambda_par,
    )
    elapsed_time1 = time.time() - time_s1

    print(f"Ground truth transport cost is {d}")
    print(f"Using fast matrix multiplication, truth transport cost is {d1}")
    print(f"Time taken for computing Ground truth transport cost is {elapsed_time}")
    print(f"Time taken for computing fast transport cost is {elapsed_time1}")
    print(f"Difference between the transport matrices are {((T-T1)**2).sum()}")


if __name__ == "__main__":
    main()
