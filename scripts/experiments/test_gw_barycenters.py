import scipy as sp
import scipy.spatial
from scipy import linalg
import numpy as np
import ot
import random
import time
import argparse
import os

from ega.algorithms.fused_gromov_wasserstein import gw_barycenters
from ega.algorithms.graph_diffusion_gf_integrator import DFGFIntegrator
from ega.util.mesh_utils import (
    random_projection_creator,
    density_function,
    fourier_transform,
)

parser = argparse.ArgumentParser(
    description="Testing computations of Gromov Wasserstein Barycenters"
)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument(
    "--n_samples", default=1000, type=int, help="Number of random samples to draw"
)
parser.add_argument(
    "--lambda_par",
    default=0.75,
    type=float,
    help="Lambda for the smoothening of the kernel matrix",
)
parser.add_argument(
    "--epsilon", default=0.5, type=float, help="Neighborhood distance around a point"
)
parser.add_argument(
    "--number_random_feats",
    default=16,
    type=int,
    help="Number of orthogonal random features, 16 and 32 works best for us. 64 produces bad results.",
)
parser.add_argument(
    "--use_armijo",
    default=True,
    type=bool,
    help="Whether to armijo line search algorithms. Turn it off if there are numerical instabilities",
)
parser.add_argument(
    "--max_iter",
    default=1000,
    type=int,
    help="Max number of iterations",
)
parser.add_argument(
    "--tolerance",
    default=1e-9,
    type=float,
    help="Error bound to stop iterations",
)
parser.add_argument(
    "--output_dir",
    default="/output/",
    type=str,
    help="Folder to save the output",
)


def main():
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    mu_s = np.array([0, 0, 0])
    r_s = np.random.rand(3, 3)
    cov_s = np.matmul(r_s, r_s.T)

    mu_t = np.array([4, 4, 4])
    r_t = np.random.rand(3, 3) * (np.random.rand(3, 3) / 7)
    cov_t = np.matmul(r_t, r_t.T)

    mu = np.array([-1, -1, -1])
    r = np.random.rand(3, 3) * (np.random.rand(3, 3) * 3)
    cov = np.matmul(r, r.T)

    Q = sp.linalg.sqrtm(cov_s)
    x0 = np.random.randn(args.n_samples, 3).dot(Q) + mu_s
    P = sp.linalg.sqrtm(cov_t)
    x1 = np.random.randn(args.n_samples, 3).dot(P) + mu_t
    R = sp.linalg.sqrtm(cov)
    x2 = np.random.randn(args.n_samples, 3).dot(R) + mu

    C0 = sp.spatial.distance.cdist(x0, x0, "minkowski", p=1)
    C1 = sp.spatial.distance.cdist(x1, x1, "minkowski", p=1)
    C2 = sp.spatial.distance.cdist(x2, x2, "minkowski", p=1)

    C0[C0 > args.epsilon] = 0
    C1[C1 > args.epsilon] = 0
    C2[C2 > args.epsilon] = 0

    time_s = time.time()
    D0 = linalg.expm(args.lambda_par * C0)
    D0 = (D0 + D0.T) / 2
    D1 = linalg.expm(args.lambda_par * C1)
    D1 = (D1 + D1.T) / 2
    D2 = linalg.expm(args.lambda_par * C2)
    D2 = (D2 + D2.T) / 2

    Cs = [D0, D1, D2]
    ps = [ot.unif(args.n_samples), ot.unif(args.n_samples), ot.unif(args.n_samples)]
    p = ot.unif(args.n_samples)
    lambdas = [1 / 3, 1 / 3, 1 / 3]

    C, log = gw_barycenters(
        args.n_samples,
        Cs,
        ps,
        p,
        lambdas,
        "square_loss",
        max_iter=args.max_iter,
        tol=args.tolerance,
        verbose=False,
        log=True,
        init_C=None,
        random_state=args.seed,
        alpha=0.5,
        armijo=args.use_armijo,
        method_type=None,
        integrators=None,
    )
    elapsed_time = time.time() - time_s
    del C0, C1, C2, D0, D1, D2, Cs

    ### test diffusion variant
    time_s1 = time.time()
    dfgf_s0_integrator = DFGFIntegrator(
        positions=x0,
        epsilon=args.epsilon,
        lambda_par=args.lambda_par,
        num_rand_features=args.number_random_feats,
        dim=3,
        random_projection_creator=random_projection_creator,
        density_function=density_function,
        fourier_transform=fourier_transform,
    )
    dfgf_s1_integrator = DFGFIntegrator(
        positions=x1,
        epsilon=args.epsilon,
        lambda_par=args.lambda_par,
        num_rand_features=args.number_random_feats,
        dim=3,
        random_projection_creator=random_projection_creator,
        density_function=density_function,
        fourier_transform=fourier_transform,
    )
    dfgf_s2_integrator = DFGFIntegrator(
        positions=x2,
        epsilon=args.epsilon,
        lambda_par=args.lambda_par,
        num_rand_features=args.number_random_feats,
        dim=3,
        random_projection_creator=random_projection_creator,
        density_function=density_function,
        fourier_transform=fourier_transform,
    )

    Cs1 = [dfgf_s0_integrator, dfgf_s1_integrator, dfgf_s2_integrator]

    Cost1, log1 = gw_barycenters(
        N=args.n_samples,
        Cs=None,
        ps=ps,
        q=p,
        lambdas=lambdas,
        loss_fun="square_loss",
        max_iter=args.max_iter,
        tol=args.tolerance,
        verbose=False,
        log=True,
        init_C=None,
        random_state=42,
        alpha=0.5,
        armijo=args.use_armijo,
        method_type="diffusion",
        integrators=Cs1,
    )
    elapsed_time1 = time.time() - time_s1

    print("Saving the data for visualization later")
    with open(os.path.join(args.output_dir, "brute_force_cost_matrix.npy"), "w") as fp:
        np.save(fp, C)
    with open(
        os.path.join(args.output_dir, "diffusion_variant_cost_matrix.npy"), "w"
    ) as fp:
        np.save(fp, Cost1)
    print("Saving completed.")
    print(f"Time taken for computing Ground truth transport cost is {elapsed_time}")
    print(f"Time taken for computing fast transport cost is {elapsed_time1}")


if __name__ == "__main__":
    main()
