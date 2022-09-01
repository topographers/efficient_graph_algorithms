import numpy as np

def rescale_data(mesh, scale):
    mesh.points -= np.array(mesh.center)[None, :]
    diameter = 0.5 * (mesh.points.max() - mesh.points.min())
    mesh.points *= scale / diameter
    return mesh