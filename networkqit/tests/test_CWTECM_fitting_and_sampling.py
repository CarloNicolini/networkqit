#!/usr/bin/env python3

from pathlib import Path
home = str(Path.home())
import sys
import numpy as np
import matplotlib.pyplot as plt
import bct
import sys
sys.path.append(home + '/workspace/networkqit')
import networkqit as nq
from networkqit.graphtheory.models.MEModels import CWTECM, UBCM, UWCM,UECM3


if __name__=='__main__':

    filename = home + '//workspace/communityalg/data/GroupAverage_rsfMRI_unthr.adj'
    G = np.loadtxt(filename)[0:64, 0:64]
    t = 0.3
    G = bct.threshold_absolute(G,t)
    print(G)
    tstar = G[np.nonzero(G)].min()
    A = (G>0).astype(float)
    k = A.sum(axis=0)
    m = k.sum()
    s = G.sum(axis=0)
    Wtot = s.sum()
    n = len(G)
    pairs = n*(n-1)/2

    M = CWTECM(N=len(G),threshold=tstar)
    sol = M.fit(G=G, method='MLE',ftol=1E-12,verbose=0)
    
    print('Loglikelihood = ', M.loglikelihood(G,sol['x']))
    
    nq.plot_mle(G, M.expected_adjacency(sol['x']), M.expected_weighted_adjacency(sol['x']))
    plt.show()

    # nq.MLEOptimizer(W, x0=sol['x'], model=M)
    # sol = opt.run(method='saddle_point', xtol=1E-12, gtol=1E-9)
    # print('Loglikelihood = ', M.loglikelihood(G,sol['x']))
    # pij = M.expected_adjacency(sol['x'])
    # wij = M.expected_weighted_adjacency(sol['x'])
    # plot(W,pij,wij)

    # sol = opt.run(method='saddle_point', basinhopping = True, basin_hopping_niter=10, xtol=1E-9, gtol=1E-9)
    # #print('Gradient at Least squares solution=\n',grad(sol['x']))
    # print('Loglikelihood = ', M.loglikelihood(G,sol['x']))
    # pij = M.expected_adjacency(sol['x'])
    # wij = M.expected_weighted_adjacency(sol['x'])
    # plot(W,pij,wij)
