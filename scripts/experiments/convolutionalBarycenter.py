import numpy as np
from scipy.optimize import root_scalar

def convolutionalBarycenter(p, alpha, areaWeights, kernel, kernelTranspose, options, entropyLimit=None):

    if kernelTranspose is None:
        kernelTranspose = kernel

    areaWeights_np = np.array(areaWeights.todense()).T[0]

    if areaWeights_np is None:
        areaWeights_np = np.ones(np.size(p,1),1)


    niter = options['niter'] #1500
    tol = options['tol'] # 1e-7
    verb = options['verb'] # 1
    displayFunction = options['disp'] # []
    disp_rate = options['disp_rate'] # 10
    v = options['initial_v'] # ones(np.size(p)))
    barycenter = options['initial_barycenter'] # np.ones(np.size(p,1),1)
    unit_area_projection = options['unit_area_projection'] # 0

    alpha = alpha/np.sum(alpha)

    for j in range(niter):
        oldBarycenter = barycenter

        w = np.divide(p, kernelTranspose(np.multiply(v.T,areaWeights_np).T))
        
        if unit_area_projection == 1:
            integrals = np.sum(np.multiply(areaWeights_np,np.multiple(v, kernel(np.multiply(w.T,areaWeights_np).T)),1))
            w = w/integrals
        
        d = np.multiply(v, kernel(np.multiply(w.T,areaWeights_np).T))

        d[d<1e-300] = 1e-300

        barycenter = np.exp(np.sum(np.multiply(alpha,np.log(d)),axis=1))

        entropy = -np.sum(np.multiply(areaWeights_np, np.multiply(barycenter,np.log(barycenter))))

        if j>1 and (entropyLimit is not None) and (entropy>entropyLimit):
            fn = lambda x: -np.sum(x*np.multiply(areaWeights_np, np.multiply(np.power(barycenter, x), np.log(barycenter)))) - entropyLimit
            options['Display'] = None
            options['tolfun'] = 1e-4
            options['tolx'] = 1e-4
            try:
                sol = root_scalar(fn, args=(), method='toms748', bracket=[0.5, 3])
                a = sol.root 
                if verb==1:
                    print('\ta = %g\n',a)
            except:
                a = 1
                print('\tProjection failed.\n')
            
            barycenter = np.power(barycenter, a)

        v = np.multiply(v.T,barycenter).T/d
        
        if unit_area_projection == 1:
            integrals = np.sum(np.multiply(areaWeights_np, np.multiply(v, kernel(np.multiply(w,areaWeights))),1));
            v = np.divide(v,integrals)
        

        change = np.sum(np.multiply(np.absolute(oldBarycenter-barycenter), areaWeights_np))
        area = np.sum(np.multiply(barycenter, areaWeights_np))
        
        if verb==1:
            print('Iteration {0}:  change = {1}, area = {2}\n'.format(j,change,area))
        
    return barycenter, v
