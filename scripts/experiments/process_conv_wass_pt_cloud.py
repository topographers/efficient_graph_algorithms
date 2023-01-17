import numpy as np
import os
import open3d as o3d

from ega.visualization.point_cloud_visualization import render_pointcloud_still_np

clouds = [f for f in os.listdir(".") if f.startswith("test_interpolation") and f.endswith(".npy")]
for index, filename in enumerate(clouds):
    cloud = np.load(filename)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    downpcd = pcd.voxel_down_sample(voxel_size=0.1)
    print(f"There are {len(pcd.points)} points in cloud {index}")
    print(f"Reduced cloud to {len(downpcd.points)} points")
    render_pointcloud_still_np(
        np.asarray(downpcd.points),
        "small_test_interpolation" + str(index) + ".png",
        scale_points=False,
        camera_position=(2.2, 2.2, 2.2),
    )
    np.save(f"small_test_interpolation{index}.npy", np.asarray(downpcd.points))
