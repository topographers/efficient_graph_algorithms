import numpy as np
import pyvista

def rescale_mesh(mesh: pyvista.PolyData, scale: float):
    """Function to rescale the meshes. The meshes used are pyvista meshes."""
    mesh.points -= np.array(mesh.center)[None, :]
    diameter = 0.5 * (mesh.points.max() - mesh.points.min())
    mesh.points *= scale / diameter
    return mesh
