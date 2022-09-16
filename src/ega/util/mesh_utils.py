
import trimesh 
import numpy as np 
import scipy

def random_circular_rotation(adjacency_lists, seed):
    np.random.seed(seed)
    for index in range(len(adjacency_lists)):
        shift = np.random.randint(low=0, high=len(adjacency_lists[index]))
        rotated_adjacency_list = np.roll(np.array(adjacency_lists[index]), 
                                         shift=shift).tolist()
        adjacency_lists[index] = rotated_adjacency_list
    return adjacency_lists


def neighbors_in_cyclic_order(face, vertex_index):
    v1, v2, v3 = face
    if v1 == vertex_index:
        return [v2, v3]
    if v2 == vertex_index:
        return [v3, v1]
    return [v1, v2]    


def trimesh_to_adjacency_matrices(mesh, seed=0):
    graph = trimesh.graph.vertex_adjacency_graph(mesh)
    vertices = list(graph.nodes)
    faces = mesh.faces
    faces_adj_to_vertices = []
    adjacency_lists = []
    for _  in range(len(vertices)):
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


def adjacency_list_to_sparse_matrix(adjacency_lists: list[list[int]]):
    row_idx_list = []
    rwo_content_list = []
    for i_idx, i_list in enumerate(adjacency_lists):
        for j_value in i_list:
            row_idx_list.append(i_idx)
            rwo_content_list.append(j_value)
    n_vertices = row_idx_list[-1] + 1
    V = [1] * len(row_idx_list)

    sparse_adjacency_matrix = scipy.sparse.coo_matrix((V,(row_idx_list,rwo_content_list)),shape=(n_vertices,n_vertices))
    return sparse_adjacency_matrix  


