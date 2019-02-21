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
from networkqit.graphtheory.models.MEModels import CWTECM
from networkqit.utils.visualization import plot_mle

if __name__=='__main__':

    # #np.random.seed(0)
    # G  = np.triu(np.random.exponential(0.2, size=[64,64]), 1) *  10
    # G +=  G.T
    # t = 0.5
    # G = G * (G > t)
    # t = G[np.nonzero(G)].min()
    # A = G>t
    # print('threshold=',t)

    filename = home + '/workspace/communityalg/data/GroupAverage_rsfMRI_weighted.adj'
    G = np.loadtxt(filename)[0:64, 0:64]
    t = G[np.nonzero(G)].min()

    W = G
    A = (G>t).astype(float)
    k = A.sum(axis=0)
    s = G.sum(axis=0)
    Wtot = s.sum()
    
    n = len(A)
    m = k.sum()
    pairs = n*(n-1)/2

    M = CWTECM(N=len(G), threshold=t)
    sol = M.fit(G, method='MLE', model=M, verbose=0, maxiter=5000, gtol=1E-6)
    print('Loglikelihood = ', M.loglikelihood(G,sol['x']),'AIC = ', sol['AIC'])

    pij = M.expected_adjacency(sol['x'])
    wij = M.expected_weighted_adjacency(sol['x'])
    plot_mle(W,pij,wij,title='Loglikelihood LBFGS-B')
    plt.show()

    # TEST JACOBIAN
    from autograd import grad
    print('|grad|=',np.sqrt((grad(lambda z : M.loglikelihood(W,z))(sol['x'])**2).sum()))
    print('saddle_point=', np.sqrt((M.saddle_point(W,sol['x'])**2)).sum())

    # TEST SAMPLING
    Sd = (M.sample_adjacency(sol['x'], batch_size=1, with_grads=False)>0)[0,:,:]
    S = (M.sample_adjacency(sol['x'], batch_size=1, with_grads=False))[0,:,:]
    plot_mle(W, Sd.astype(float), S, title='Sampling')
    plt.show()

    # TEST SAMPLING WITH AVERAGING
    Sd = (M.sample_adjacency(sol['x'], batch_size=500, with_grads=False)>0).mean(axis=0)
    S = (M.sample_adjacency(sol['x'], batch_size=500, with_grads=False)).mean(axis=0)
    plot_mle(W, Sd.astype(float), S, title='Sampling averaging')
    plt.show()

    # TEST SADDLE POINT
    sol = M.fit(G, method='saddle_point', model=M, verbose=0, maxiter=5000, gtol=1E-6)
    print('Loglikelihood = ', M.loglikelihood(G,sol['x']))
    pij = M.expected_adjacency(sol['x'])
    wij = M.expected_weighted_adjacency(sol['x'])
    plot_mle(W,pij,wij,title='Saddle point method')
    plt.show()

    from autograd import grad
    print('|grad|=',np.sqrt((grad(lambda z : M.loglikelihood(W,z))(sol['x'])**2).sum()))
    print('saddle_point=', np.sqrt((M.saddle_point(W,sol['x'])**2)).sum())

    # TEST BASINHOPPING
    # opt = nq.MLEOptimizer(W, x0=x0, model=M)
    # sol = opt.run(method='saddle_point', verbose=2, basinhopping = True, basin_hopping_niter=10, xtol=1E-9, gtol=1E-9)
    # #print('Gradient at Least squares solution=\n',grad(sol['x']))
    # print('Loglikelihood = ', M.loglikelihood(G,sol['x']))
    # pij = M.expected_adjacency(sol['x'])
    # wij = M.expected_weighted_adjacency(sol['x'])
    # plot_mle(W,pij,wij)