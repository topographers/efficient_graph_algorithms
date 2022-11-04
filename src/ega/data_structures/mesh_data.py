import numpy as np
import trimesh

from ega.algorithms.cotangent_laplacian import cotangent_laplacian


class MeshData:
    def __init__(self, mesh_file_path: str):
        self.mesh_file_path = mesh_file_path
        self.mesh = trimesh.load(mesh_file_path)

        self.vertices = vertices
        self.faces = faces
        self.cotangent_laplacian, self.area_weights = cotangent_laplacian(vertices, faces)
        self.area_weights = np.array(self.area_weights.todense()).T[0]
        self.num_vertices = len(vertices)
