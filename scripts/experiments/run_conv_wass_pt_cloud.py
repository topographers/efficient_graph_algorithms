import argparse
import numpy as np
import pickle

from time import time
import torch
import pyvista as pv
from pyvista import examples

from ega.util.mesh_utils import rescale_mesh
from ega.algorithms.convolutional_wass import (
    convolutional_wasserstein_barycenter_pt_cloud,
)
from ega.visualization.point_cloud_visualization import render_pointcloud_still_np


def get_args_parser():
    parser = argparse.ArgumentParser(
        "convolutional_wasserstein_point_cloud_test", add_help=False
    )

    # Model parameters
    parser.add_argument(
        "--object_folder",
        default="./data/test_point_cloud_data/",
        type=str,
        help="""path for sample data.""",
    )
    parser.add_argument(
        "--x_rot",
        default=100,
        type=float,
        help="""rotate the point cloud along x-axis""",
    )
    parser.add_argument(
        "--y_rot",
        default=140,
        type=float,
        help="""rotate the point cloud along x-axis""",
    )
    parser.add_argument(
        "--z_rot",
        default=-20,
        type=float,
        help="""rotate the point cloud along x-axis""",
    )
    parser.add_argument(
        "--smoothing_factor",
        default=100,
        type=float,
        help="""Smoothing factor for the point clouds""",
    )
    parser.add_argument(
        "--relax_factor",
        default=0.1,
        type=float,
        help="""Relaxation factor for the point clouds""",
    )
    parser.add_argument(
        "--width", default=200, type=int, help="""Number of bins for the histogram"""
    )
    parser.add_argument(
        "--regularizer", default=0.01, type=float, help="""Entropic regularizer"""
    )
    parser.add_argument(
        "--scale_meshes",
        default=0.95,
        type=float,
        help="""Scaling the data in a given cube""",
    )
    parser.add_argument(
        "--rot_second_obj",
        default=90,
        type=float,
        help="""rotate the 2nd point cloud""",
    )
    parser.add_argument(
        "--error",
        default=1e-7,
        type=float,
        help="""stopping crietrion for Sinkhorn Knapp iterations""",
    )
    parser.add_argument(
        "--save", default=True, type=bool, help="""To save the generated point clouds"""
    )
    parser.add_argument(
        "--save_file",
        default="./data/test_point_cloud_data/interpolation-data.pkl",
        type=str,
        help="""File to save the data """,
    )

    return parser


def main():
    parser = argparse.ArgumentParser(
        "Test Convolutional Wasserstein Distance on Point Clouds",
        parents=[get_args_parser()],
    )

    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # generate examples
    beta = pv.ParametricTorus()
    beta = pv.PolyData(beta)

    alpha = examples.download_bunny()
    # rotate the data and apply some transformations
    alpha.rotate_x(args.x_rot)
    alpha.rotate_z(args.z_rot)
    alpha.rotate_y(args.y_rot)
    alpha = alpha.smooth(args.smoothing_factor, relaxation_factor=args.relax_factor)
    beta = beta.smooth(args.smoothing_factor, relaxation_factor=args.relax_factor)

    alpha = rescale_mesh(alpha, args.scale_meshes)
    beta = rescale_mesh(beta, args.scale_meshes)
    beta.rotate_y(args.rot_second_obj)

    # set up histograms and create empirical distributions

    n_features = args.width ** 3

    hist_grid = torch.linspace(-1.0, 1.0, args.width + 1)
    grid = torch.linspace(-1.0, 1.0, args.width)
    X, Y, Z = torch.meshgrid(grid, grid, grid)
    alpha_hist = np.histogramdd(alpha.points, bins=[hist_grid, hist_grid, hist_grid])[0]
    beta_hist = np.histogramdd(beta.points, bins=[hist_grid, hist_grid, hist_grid])[0]
    alpha_hist /= alpha_hist.sum()
    beta_hist /= beta_hist.sum()

    # prepare histograms to be fed into the model and set up the weights
    hists = np.stack((alpha_hist, beta_hist))
    hists += 1e-10
    hists /= hists.sum(axis=(1, 2, 3))[:, None, None, None]
    hists = torch.tensor(hists).type(torch.float32)
    hists = hists.to(device)
    interpolating_points = [0.0, 0.25, 0.5, 0.75, 1.0]  # interpolating points

    data = dict(ibp=dict(times=[], bars=[]))
    bars = []
    for ii, w in enumerate(interpolating_points):
        print("->>> Doing weight {} ... ".format(ii + 1))
        weights = torch.tensor([1.0 - w, w])
        t0 = time()
        bar_ibp = convolutional_wasserstein_barycenter_pt_cloud(
            hists, reg=args.regularizer, weights=weights, threshold=args.error
        )
        t1 = time()
        print("IBP done in ", t1 - t0)
        data["ibp"]["times"].append(t1 - t0)
        data["ibp"]["bars"].append(bar_ibp.cpu())

    # save data
    if args.save:
        with open(args.save_file, "wb") as ff:
            pickle.dump(data, ff)

    # VISUALIZE THESE OBJECTS and save them as png.
    for key in ["ibp"]:
        bars = data[key]["bars"]
        for ii, hist in enumerate(bars):
            print("->> creating mesh {} ... ".format(ii + 1))
            support = torch.where(hist > args.error)
            weights = hist[support].numpy()
            cloud = torch.stack((X[support], Y[support], Z[support])).t()
            render_pointcloud_still_np(
                cloud.numpy(),
                "test_interpolation" + str(ii) + ".png",
                scale_points=False,
                camera_position=(2.2, 2.2, 2.2),
            )


if __name__ == "__main__":
    main()
