import numpy as np


def rescale_mesh(mesh, scale):
    """Function to rescale the meshes. The meshes used are pyvista meshes."""
    mesh.points -= np.array(mesh.center)[None, :]
    diameter = 0.5 * (mesh.points.max() - mesh.points.min())
    mesh.points *= scale / diameter
    return mesh
