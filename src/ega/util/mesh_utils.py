from typing import List, Callable

import numpy as np
import pyvista
import scipy
import trimesh
from scipy import linalg


def rescale_mesh(mesh: pyvista.PolyData, scale: float):
    """Function to rescale the meshes. The meshes used are pyvista meshes."""
    mesh.points -= np.array(mesh.center)[None, :]
    diameter = 0.5 * (mesh.points.max() - mesh.points.min())
    mesh.points *= scale / diameter
    return mesh


def random_circular_rotation(adjacency_lists: List[List[int]], seed: int) -> List[List[int]]:
    """ this function randomly shuffles the adjacency list """
    np.random.seed(seed)
    for index in range(len(adjacency_lists)):
        shift = np.random.randint(low=0, high=len(adjacency_lists[index]))
        rotated_adjacency_list = np.roll(np.array(adjacency_lists[index]),
                                         shift=shift).tolist()
        adjacency_lists[index] = rotated_adjacency_list
    return adjacency_lists


def neighbors_in_cyclic_order(face: List[int], vertex_index: int) -> List[int]:
    """ this function returns neighbors of current vertex in cyclic order. 
        Lipton-Tarjan algorithm only works when the neighbor node indices are ordered either clockwise or anti-clockwise
    """
    v1, v2, v3 = face
    if v1 == vertex_index:
        return [v2, v3]
    if v2 == vertex_index:
        return [v3, v1]
    return [v1, v2]


def trimesh_to_adjacency_matrices(mesh: trimesh.base, seed: int = 0) -> List[List[int]]:
    """
    given a trimesh object, this function will output the adjacency lists
    """
    graph = trimesh.graph.vertex_adjacency_graph(mesh)
    vertices = list(graph.nodes)
    faces = mesh.faces
    faces_adj_to_vertices = []
    adjacency_lists = []
    for _ in range(len(vertices)):
        faces_adj_to_vertices.append([])
        adjacency_lists.append([])
    for index in range(len(faces)):
        v1, v2, v3 = faces[index]
        faces_adj_to_vertices[v1].append(index)
        faces_adj_to_vertices[v2].append(index)
        faces_adj_to_vertices[v3].append(index)
    for index in range(len(faces_adj_to_vertices)):
        vertex_adjacency_list = []
        edge_dict = dict()
        first_vertex = 0
        for face_index in faces_adj_to_vertices[index]:
            x, y = neighbors_in_cyclic_order(faces[face_index], index)
            edge_dict[x] = y
            first_vertex = x
        next_vertex = first_vertex
        while True:
            vertex_adjacency_list.append(next_vertex)
            next_vertex = edge_dict[next_vertex]
            if next_vertex == first_vertex:
                break
        adjacency_lists[index] = vertex_adjacency_list
    return random_circular_rotation(adjacency_lists, seed)


def adjacency_list_to_sparse_matrix(adjacency_lists: List[List[int]]):
    """
    given an adjacency list, this function will return the scipy sparse matrix of it
    """
    row_idx_list = []
    rwo_content_list = []
    for i_idx, i_list in enumerate(adjacency_lists):
        for j_value in i_list:
            row_idx_list.append(i_idx)
            rwo_content_list.append(j_value)
    n_vertices = row_idx_list[-1] + 1
    V = [1] * len(row_idx_list)

    sparse_adjacency_matrix = scipy.sparse.coo_matrix((V, (row_idx_list, rwo_content_list)),
                                                      shape=(n_vertices, n_vertices))
    return sparse_adjacency_matrix


def generate_weights_from_adjacency_list(adjacency_lists: List[List[int]]) -> List[List[int]]:
    """
    given an adjacency list, this function will return the corresponding unweighted list 
    (every element equal to 1 in this unweighted list)
    """
    weight_lists = []
    for list_i in adjacency_lists:
        current_list = []
        for j in list_i:
            current_list.append(1)
        weight_lists.append(current_list)

    return weight_lists



def calculate_interpolation_metrics(true_fields: np.ndarray, interpolated_fields: np.ndarray, verbose: bool = False):
    """ 
    this function calculates the frobenius norma and cosine similarity between predicted and true graph fields. 
    
    true_fields and interpolated_fields: both should have dimension of N by d, with N represents number of nodes, 
                                         and d represents the dimension of graph field features.
    """

    frobenius_norm = np.linalg.norm(true_fields - interpolated_fields, ord='fro')
    cosine_similarity = np.mean((true_fields * interpolated_fields).sum(axis=-1) / \
                                np.linalg.norm(true_fields, axis=-1) / np.linalg.norm(interpolated_fields, axis=-1))
    if verbose:
        print("Frobenious Norm: {}\nCosine Similarity: {}".format(frobenius_norm, cosine_similarity))
    return frobenius_norm, cosine_similarity


def density_function(input_projection: np.ndarray) -> np.ndarray:
    """
    density function of the probabilistic distribution applied by the below 
    random_projection_creator to construct projections
    
    both the input and output are 1d numpy arrays
    """
    dim = len(input_projection)
    length = linalg.norm(input_projection)
    return (1.0 / np.power(2.0 * np.pi, dim / 2.0)) * np.exp(-length ** 2 / 2.0)


def random_projection_creator(num_random_features: int,
                              dim: int,
                              scaling: int = 0,
                              ortho: bool = True) -> np.ndarray:
    """
    this is a function  N * N -> R ^ {N * N}  that constructs random projections 
    used to construct low-rank decomposition of the adjacency matrix.
    
    for the input: (num_rand_features, dim), a matrix of num_rand_features rows and 
    dim columns is created (with different rows corresponding to different projections)
    
    ortho: 
        if True, then we use blocks of orthogonal random features as projection directions
        if False, then we use vanilla iid normals as projection directions
    """
    seed = 0
    np.random.seed(seed)
    if not ortho:
        final_matrix = np.random.normal(size=(num_random_features, dim))
    else:
        nb_full_blocks = int(num_random_features / dim)
        block_list = []
        for _ in range(nb_full_blocks):
            unstructured_block = np.random.normal(size=(dim, dim))
            q, _ = np.linalg.qr(unstructured_block)
            q = np.transpose(q)
            block_list.append(q)
        remaining_rows = num_random_features - nb_full_blocks * dim
        if remaining_rows > 0:
            unstructured_block = np.random.normal(size=(dim, dim))
            q, _ = np.linalg.qr(unstructured_block)
            q = np.transpose(q)
            block_list.append(q[0:remaining_rows])
        final_matrix = np.vstack(block_list)

    if scaling == 0:
        multiplier = np.linalg.norm(np.random.normal(size=(num_random_features, dim)), axis=1)
    elif scaling == 1:
        multiplier = np.sqrt(float(dim)) * np.ones(num_random_features)
    else:
        raise ValueError('Scaling must be one of {0, 1}. Was %s' % scaling)

    return np.matmul(np.diag(multiplier), final_matrix)


def fourier_transform(input_projection: np.ndarray, epsilon: float, norm_type='L1'):
    """
    Fourier Transform of the function defining edges between graph nodes.
    """
    if norm_type == 'L1':
        return np.prod(np.sin(2.0 * epsilon * input_projection) / input_projection)
    elif norm_type == 'L2':
        pass
    else:
        pass


def construct_random_features(positions: np.ndarray,
                              random_projection_creator: Callable,
                              density_function: Callable,
                              num_rand_features: int,
                              fourier_transform: Callable,
                              epsilon: float):
    """
    this function is used by graph diffusion GFIntegrator
    
    parameter definitions are the same as in graph_diffusion_gf_integrator
    """
    dim = len(positions[0])
    projection_matrix = random_projection_creator(num_rand_features, dim)
    projected_positions = np.einsum('md,nd->nm', projection_matrix, positions)
    exp_projected_positions = np.exp(2.0 * np.pi * 1j * projected_positions)

    def features_with_epsilon(x):
        return fourier_transform(x, epsilon)

    fts = np.apply_along_axis(features_with_epsilon, 1, projection_matrix)
    dens = np.apply_along_axis(density_function, 1, projection_matrix)
    renormalizers = fts / dens
    rfs = np.einsum('nm,m->nm', exp_projected_positions, renormalizers)
    return (1.0 / np.sqrt(num_rand_features)) * rfs
