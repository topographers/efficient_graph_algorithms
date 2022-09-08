import numpy as np
import scipy.sparse as sps
import scipy.linalg as splinalg

def cotLaplacian(X, T):
    # Find orig edge lengths and angles
    nv = len(X); nf = len(T)
    # note index changes below, because matlab index starts with 1
    L1 = np.linalg.norm(X[T[:,1],:]-X[T[:,2],:], axis=1) 
    L2 = np.linalg.norm(X[T[:,0],:]-X[T[:,2],:], axis=1)
    L3 = np.linalg.norm(X[T[:,0],:]-X[T[:,1],:], axis=1)

    EL = np.array([L1,L2,L3]).T
    A1 = (L2**2 + L3**2 - L1**2) / (2*L2*L3)
    A2 = (L1**2 + L3**2 - L2**2) / (2*L1*L3)
    A3 = (L1**2 + L2**2 - L3**2) / (2*L1*L2)
    A = [A1,A2,A3]
    A = np.arccos(A).T

    # The Cot Laplacian 
    I = np.concatenate((T[:,0], T[:,1], T[:,2]))
    J = np.concatenate((T[:,1], T[:,2], T[:,0]))
    S = 0.5 / np.tan(np.concatenate([A[:,2],A[:,0],A[:,1]]))
    In = np.concatenate((I, J, I, J))
    Jn = np.concatenate((J, I, I, J))
    Sn = np.concatenate((-S, -S, S, S))

    # Compute the areas. Use mixed weights Voronoi areas
    cA = 0.5 / np.tan(A);
    vp1 = [1,2,0]; vp2 = [2,0,1];
    At = 1/4 * (EL[:,vp1]**2 * cA[:,vp1] + EL[:,vp2]**2 * cA[:,vp2])

    # Triangle areas
    N = np.cross(X[T[:,0],:]-X[T[:,1],:], X[T[:,0],:] - X[T[:,2],:]);
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
    A = sps.csr_matrix((S, (I,J)), shape=(nv,1)) 
    W = sps.csr_matrix((Sn, (In,Jn)), shape=(nv,nv))
    W1 = np.zeros((nv,nv))
    return W, A

class getMeshData():
    def __init__(self,X,T,numEigs=10, name='mesh'):
        super(getMeshData, self).__init__()
        self.vertices = X; self.triangles = T; self.name = 'name'
        self.cotLaplacian, self.areaWeights = cotLaplacian(X,T)
        # Change to negative cot Laplacian and rescale to area = 1
        self.areaWeights = self.areaWeights / np.sum(self.areaWeights)
        self.cotLaplacian = -self.cotLaplacian
        T = T+1 
        self.numVertices= len(X); self.numTriangles = len(T)
        evec = np.vstack((T[:,0], T[:,1]))
        evec = np.hstack((evec, np.vstack((T[:,1], T[:,0]))))
        evec = np.hstack((evec, np.vstack((T[:,0], T[:,2]))))
        evec = np.hstack((evec, np.vstack((T[:,2], T[:,0]))))
        evec = np.hstack((evec, np.vstack((T[:,1], T[:,2]))))
        evec = np.hstack((evec, np.vstack((T[:,2], T[:,1]))))

        evec = np.unique(evec,axis=1).T
        orderedRows = [i for i in range(len(evec)) if
                       evec[i,0] < evec[i,1]]
        # orderedRows = matlab version - one, since python index starts at 0 
        self.edges = evec[orderedRows,:]
        self.numEdges = len(self.edges)

        # Compute LB eigenstuff
        nv = self.numVertices # for convenience
        areaWeights_np = np.array(self.areaWeights.todense()).T[0]
        areaMatrix = sps.csr_matrix((areaWeights_np,(np.arange(nv),np.arange(nv))))
        numEigs = max(numEigs, 1)

        cotLaplacian_np = np.array(self.cotLaplacian.todense())
        areaMatrix_np = np.array(areaMatrix.todense())
        evals, evecs = splinalg.eigh(cotLaplacian_np, areaMatrix_np,
                       subset_by_index = [nv - numEigs, nv - 1]);
        # eigenvectors match matlab output (note negate of eigenvector is still eigenvector associated with the same eigenvalue)
        self.laplaceBasis = evecs; self.eigenvalues = np.diag(evals)

        normalf = np.cross(self.vertices[self.triangles[:,1],:] -
                           self.vertices[self.triangles[:,0],:], 
                           self.vertices[self.triangles[:,2],:] -
                           self.vertices[self.triangles[:,0],:] )
        d_ = np.linalg.norm(normalf, axis=1)
        eps = 2.2204e-16 #floating point accuracy
        d_[d_<eps]=1
        d_tile = np.tile(d_,(3,1)).T
        self.faceNormals = (normalf / d_tile)
        
