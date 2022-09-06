import argparse
import numpy as np
import pdb
import getMeshData
import blurOnMesh
from convolutionalBarycenter import convolutionalBarycenter

def parse_arguments():
    parser = argparse.ArgumentParser(description='Convbarycenter Argument Parser')
    parser.add_argument('--niter', dest='niter', type=int, default=1500)
    parser.add_argument('--tol', dest='tol', type=float, default=1e-7)
    parser.add_argument('--verb', dest='verb', type=int, default=1)
    parser.add_argument('--disp_rate', dest='disp_rate', type=int, default=10)
    parser.add_argument('--initial_v', dest='initial_v', type=float, default=3e-4)
    return parser.parse_args()
 
if __name__ == '__main__':
    filepath = r'C:\Users\yunfan\Desktop\topography\2015-SIGGRAPH-convolutional-ot-master\2015-SIGGRAPH-convolutional-ot-master\data\meshes\moomoo_s0.off'
    X, T = getMeshData.readOff(filepath)
    W, A = getMeshData.cotLaplacian(X, T)
    M = getMeshData.getMeshData(X, T, 10)
    
    blurTime = .001 # if this gets too small, distances get noisy
    blurSteps = 3

    blur = lambda x: blurOnMesh.blurOnMesh(x,M,blurTime,blurSteps)
    blurTranspose = lambda x: blurOnMesh.blurOnMesh(x,M,blurTime,blurSteps,1)

    # Design a few functions to average

    centerVerts = [300, 100, 600]
    nFunctions = len(centerVerts)
    distributions = np.zeros((M.numVertices,nFunctions))
    areaweight_np = np.array(M.areaWeights.todense()).T[0]

    for i in range(nFunctions):
        distributions[centerVerts[i]-1,i] = 1 / areaweight_np[centerVerts[i]-1]
        # index - 1 because matlab index starts at 1
        distributions[:,i] =  blurOnMesh.blurOnMesh(
            distributions[:,i],M,blurTime,blurSteps)

    options={}
    options['niter'] = 1500
    options['tol'] = 1e-7
    options['verb'] = 1
    options['disp'] = []
    options['disp_rate'] = 10
    options['initial_v'] = np.ones(distributions.shape)
    options['initial_barycenter'] = np.ones((distributions.shape[0], 1))
    options['unit_area_projection'] = 0
    options['Display'] = None
    options['tolfun'] = 1e-4
    options['tolx'] = 1e-4


    alpha = np.ones(3)
    barycenter, _ = convolutionalBarycenter(distributions,alpha,M.areaWeights,blur,blurTranspose,options)
    print(barycenter)

