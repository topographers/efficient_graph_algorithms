import scipy as sp
from scipy.spatial.distance import cdist
import numpy as np
import ot
import random
import time
import argparse
from ega.algorithms.fused_gromov_wasserstein import fgw_lp

parser = argparse.ArgumentParser(
    description="Testing computations of Gromov Wasserstein discrepancy"
)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument(
    "--n_samples", default=3000, type=int, help="Number of random samples to draw"
)
parser.add_argument(
    "--lambda_par",
    default=1e-4,
    type=float,
    help="Lambda for the smoothening of the kernel matrix",
)
parser.add_argument(
    "--epsilon", default=0.1, type=float, help="Neighborhood distance around a point"
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
    "--features_metric",
    default='dirac',
    type=str,
    help="Choose from dirac and hamming. Other methods are not yet implemented",
)

parser.add_argument(
    "--different_random_features",
    default=False,
    type=bool,
    help="Different random features",
)


def main():
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    def hamming_dist(x, y):
        return len([i for i, j in zip(x, y) if i != j])

        # generate data

    mu_s = np.array([0, 0, 0])
    r_s = np.random.rand(3, 3)
    cov_s = np.matmul(r_s, r_s.T)

    mu_t = np.array([4, 4, 4])
    if args.different_random_features:
        r_t = np.array([
            [random.random() * 1.0, random.random() * 2.0, random.random() * 3.0],
            [random.random() * 4.0, random.random() * 5.0, random.random() * 6.0],
            [random.random() * 7.0, random.random() * 8.0, random.random() * 9.0],
        ])
    else:
        r_t = np.random.rand(3, 3)
    cov_t = np.matmul(r_t, r_t.T)

    Q = sp.linalg.sqrtm(cov_s)
    xs = np.random.randn(args.n_samples, 3).dot(Q) + mu_s
    # generate random labels
    ys = (np.random.randint(2, size=args.n_samples)).reshape(-1, 1)
    P = sp.linalg.sqrtm(cov_t)
    xt = np.random.randn(args.n_samples, 3).dot(P) + mu_t
    yt = np.random.randint(2, size=args.n_samples)
    np.random.shuffle(yt)
    yt = yt.reshape(-1, 1)

    # construct dissimilarity matrices
    Cs = sp.spatial.distance.cdist(xs, xs, 'minkowski', p=1)
    Ct = sp.spatial.distance.cdist(xt, xt, 'minkowski', p=1)

    p = ot.unif(args.n_samples)
    q = ot.unif(args.n_samples)

    if args.features_metric == 'dirac':
        f = lambda x, y: x != y
        M = cdist(ys, yt, metric=f)
    elif args.features_metric == 'hamming_dist':
        f = lambda x, y: hamming_dist(x, y)
        M = cdist(yt, ys, metric=f)
    else:
        raise NotImplementedError("Other feature distance metrics are not implemented yet.")

    # sparsify
    Cs[Cs > args.epsilon] = 0
    Ct[Ct > args.epsilon] = 0

    time_s = time.time()
    Cs1 = sp.linalg.expm(args.lambda_par * Cs)
    Cs1 = (
                  Cs1 + Cs1.T
          ) / 2  # have to symmetrize due to small numerical instabilities. depends on lambda again
    Ct1 = sp.linalg.expm(args.lambda_par * Ct)
    Ct1 = (Ct1 + Ct1.T) / 2
    trans0, log0 = fgw_lp(
        M=M,
        C1=Cs1,
        C2=Ct1,
        p=p,
        q=q,
        loss_fun="square_loss",
        alpha=0.5,
        armijo=args.use_armijo,
        G0=None,
        log=True,
    )
    elapsed_time = time.time() - time_s
    del Cs1, Ct1

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
        method_type="diffusion",
        source_positions=xs,
        target_positions=xt,
        source_epsilon=args.epsilon,
        target_epsilon=args.epsilon,
        source_lambda_par=args.lambda_par,
        target_lambda_par=args.lambda_par,
        num_rand_features=args.number_random_feats,
        dim=3,
    )
    elapsed_time1 = time.time() - time1
    rel_error = np.sum(np.abs(trans1 - trans0)) / np.sum(np.abs(trans0))

    print(f"Ground truth transport cost is {log0['fgw_dist']}")
    print(f"Using fast matrix multiplication, transport cost is {log1['fgw_dist']}")
    print(f"Time taken for computing Ground truth transport cost is {elapsed_time}")
    print(f"Time taken for computing fast transport cost is {elapsed_time1}")
    print(f"Difference between the transport matrices are {((trans0 - trans1) ** 2).sum()}")

    # make sure they are both 1
    print(f"Status message for GT computation is {log0['result_code']} and that of ours is {log1['result_code']}")
    print(f"Relative error between the transport matrices are {rel_error}")


if __name__ == "__main__":
    main()
