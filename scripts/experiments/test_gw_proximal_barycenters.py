import scipy as sp
import scipy.spatial
from scipy import linalg
import numpy as np
import ot
import random
import time
import argparse
import os
import pickle

from ega.algorithms.gromov_wasserstein_graphs import ( 
    gromov_wasserstein_barycenter, 
    estimate_target_distribution 
    )
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


    ot_dict = {'loss_type': 'L2',  # the key hyperparameters of GW distance
                           'ot_method': 'proximal',
                           'beta': 0.2,
                           'outer_iteration': 1000,  # outer, inner iteration, error bound of optimal transport
                           'iter_bound': 1e-10,
                           'inner_iteration': 2,
                           'sk_bound': 1e-10,
                           'node_prior': 10,
                           'max_iter': 5,  # iteration and error bound for calcuating barycenter
                           'cost_bound': 1e-16,
                           'update_p': False,  # optional updates of source distribution
                           'lr': 0,
                           'alpha': 0
                           }

    time_s = time.time()
    D0 = linalg.expm(args.lambda_par * C0)
    D0 = (D0 + D0.T) / 2
    D1 = linalg.expm(args.lambda_par * C1)
    D1 = (D1 + D1.T) / 2
    D2 = linalg.expm(args.lambda_par * C2)
    D2 = (D2 + D2.T) / 2

    Cost_dict = {0:D0, 
            1: D1, 
             2 : D2}
    ps_dict = {0:ot.unif(samp).reshape(-1,1), 1:ot.unif(samp).reshape(-1,1), 2:ot.unif(samp).reshape(-1,1)}
    p = ot.unif(samp) # alternatively can estimate the target distribution from the sources
    weights = {0:1/3, 1:1/3, 2:1/3}

    # brute force
    b, t, s = gromov_wasserstein_barycenter(
        costs=Cost_dict,
        p_s=ps_dict,
        ot_hyperpara=ot_dict,
        p_center=p.reshape(-1,1),
        weights=weights,
        method_type = None,
        N = None,
    )
    elapsed_time = time.time() - time_s
    del C0, C1, C2, D0, D1, D2, Cost_dict
    dict0 = {"barycenter": b, "transports": t, "distances": s, "time":elapsed_time}
    print("Saving the brute force data for visualization later")
    with open(os.path.join(args.output_dir, "brute_force_barycenter.pkl"), "wb") as fp:
        pickle.dump(dict0, fp, protocol=pickle.HIGHEST_PROTOCOL)
    del dict0, b, t, s

    # test our diffusion variant
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
    c_dict = {0:dfgf_s0_integrator, 1: dfgf_s1_integrator, 2:dfgf_s0_integrator }
    b1, t1, s1 = gromov_wasserstein_barycenter(
        costs=c_dict,
        p_s=ps_dict,
        ot_hyperpara=ot_dict,
        p_center=p.reshape(-1,1),
        weights=weights,
        method_type = "diffusion",
        N = None,
    )
    elapsed_time1 = time.time() - time_s1
    del c_dist
    dict1 = {"barycenter": b1, "transports": t1, "distances": s1, "time":elapsed_time1}
    print("Saving the diffusion barycenter data for visualization later")
    with open(os.path.join(args.output_dir, "diffusion_proximal_barycenter.pkl"), "wb") as fp:
        pickle.dump(dict1, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    if __name__ == "__main__":
        main()
