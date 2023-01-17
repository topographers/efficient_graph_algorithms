import numpy as np
import os
import open3d as o3d
import pymesh
from typing import Optional

from ega.visualization.point_cloud_visualization import render_pointcloud_still_np, render_mesh_still

def handle_cloud(save_directory, filename, index, alpha=0.136, voxel_size: Optional[float] = 0.1):
    cloud = np.load(os.path.join(save_directory, filename))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    if voxel_size is None:
        downpcd = pcd
    else:
        downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)  # type: o3d.geometry.PointCloud
    print(f"There are {len(pcd.points)} points in cloud {index}")
    print(f"Reduced cloud to {len(downpcd.points)} points")
    render_pointcloud_still_np(
        np.asarray(downpcd.points),
        os.path.join(save_directory, f"small_test_interpolation_pc{index}.png"),
        scale_points=False,
        camera_position=(2.2, 2.2, 2.2),
    )
    np.save(os.path.join(save_directory, f"small_test_interpolation{index}.npy"), np.asarray(downpcd.points))
    downpcd.estimate_normals()

    knn_normal = 160
    downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn_normal), fast_normal_computation=False)

    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(downpcd)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(downpcd, alpha, tetra_mesh, pt_map)  # type: o3d.geometry.TriangleMesh
    mesh.compute_vertex_normals()

    # radii = [0.025, 0.05, 0.1, 0.2, 0.4, 0.8]
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(downpcd, o3d.utility.DoubleVector(radii))

    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(downpcd, depth=12)

    # o3d.visualization.draw_geometries([mesh, downpcd], mesh_show_back_face=True, point_show_normal=True)
    # o3d.visualization.draw_geometries([downpcd], mesh_show_back_face=True, point_show_normal=True)
    mesh_savepath = os.path.join(save_directory, f"small_test_interpolation{index}.ply")
    o3d.io.write_triangle_mesh(mesh_savepath, mesh)

    mesh = pymesh.load_mesh(mesh_savepath)
    render_mesh_still(mesh, save_file_path=os.path.join(save_directory, f"small_test_interpolation_mesh{index}.png"), camera_position=(2.2, 2.2, 2.2))


def main():
    save_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)), "process_conv_wass_pt_cloud_results")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    clouds = sorted([f for f in os.listdir(save_directory) if f.startswith("test_interpolation") and f.endswith(".npy")])
    # for index, filename in enumerate(clouds):
    #     handle_cloud(save_directory, filename, index)

    handle_cloud(save_directory, "small_test_interpolation_torus.npy", 8, alpha=0.5, voxel_size=None)


if __name__ == "__main__":
    main()
