import numpy as np
import scipy.sparse as sps
import scipy.linalg as splinalg

def cotangent_laplacian(vertices: float, faces: int):
    # Find orig edge lengths and angles
    num_vertices = len(vertices)
    num_faces = len(faces)
    # note index changes below, because matlab index starts with 1
    L1 = np.linalg.norm(vertices[faces[:,1],:]-vertices[faces[:,2],:], axis=1) 
    L2 = np.linalg.norm(vertices[faces[:,0],:]-vertices[faces[:,2],:], axis=1)
    L3 = np.linalg.norm(vertices[faces[:,0],:]-vertices[faces[:,1],:], axis=1)

    """

    explain what each of these intermediates are derived from

    """
    
    EL = np.array([L1,L2,L3]).T
    A1 = (L2**2 + L3**2 - L1**2) / (2*L2*L3)
    A2 = (L1**2 + L3**2 - L2**2) / (2*L1*L3)
    A3 = (L1**2 + L2**2 - L3**2) / (2*L1*L2)
    A = [A1,A2,A3]
    A = np.arccos(A).T

    # The Cot Laplacian 
    I = np.concatenate((faces[:,0], faces[:,1], faces[:,2]))
    J = np.concatenate((faces[:,1], faces[:,2], faces[:,0]))
    S = 0.5 / np.tan(np.concatenate([A[:,2],A[:,0],A[:,1]]))
    In = np.concatenate((I, J, I, J))
    Jn = np.concatenate((J, I, I, J))
    Sn = np.concatenate((-S, -S, S, S))

    # Compute the areas. Use mixed weights Voronoi areas
    cA = 0.5 / np.tan(A);
    vp1 = [1,2,0]; vp2 = [2,0,1];
    At = 1/4 * (EL[:,vp1]**2 * cA[:,vp1] + EL[:,vp2]**2 * cA[:,vp2])

    # Triangle areas
    N = np.cross(vertices[faces[:,0],:]-vertices[faces[:,1],:], vertices[faces[:,0],:] - vertices[faces[:,2],:]);
    Ar = np.linalg.norm(N, axis=1);

    # Use barycentric area when cot is negative
    locs = [i for i in range(len(cA[:,0])) if cA[i,0] < 0]
    At[locs,0] = Ar[locs]/4; At[locs,1] = Ar[locs]/8; At[locs,2] = Ar[locs]/8;
    locs = [i for i in range(len(cA[:,1])) if cA[i,1] < 0]
    At[locs,0] = Ar[locs]/8; At[locs,1] = Ar[locs]/4; At[locs,2] = Ar[locs]/8;
    locs = [i for i in range(len(cA[:,2])) if cA[i,2] < 0]
    At[locs,0] = Ar[locs]/8; At[locs,1] = Ar[locs]/8; At[locs,2] = Ar[locs]/4;

    # Vertex areas = sum triangles nearby
    J = np.zeros(len(I)).astype(int);
    S = np.concatenate((At[:,0], At[:,1], At[:,2]));
    area_weights = sps.csr_matrix((S, (I,J)), shape=(num_vertices,1)) 
    cot_laplacian = sps.csr_matrix((Sn, (In,Jn)), shape=(num_vertices, num_vertices))
    return cot_laplacian, area_weights

def get_mesh_data(vertices: float, faces: int,num_eigs = 10):
    mesh_dictionary = {}
    mesh_dictionary['vertices'] = vertices
    mesh_dictionary['faces'] = faces
    cot_laplacian, area_weights = cotangent_laplacian(vertices, faces)

    # Change to negative cot Laplacian and rescale to area = 1
    mesh_dictionary['area_weights'] = area_weights / np.sum(area_weights)
    mesh_dictionary['cot_laplacian'] = -cot_laplacian

    num_vertices = len(vertices)
    mesh_dictionary['num_vertices'] = num_vertices
    
    return mesh_dictionary
    
