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
from networkqit.graphtheory.models.MEModels import CWTECM, UBCM, UWCM, UECM3
from networkqit.utils.visualization import plot_mle

if __name__=='__main__':

    #np.random.seed(0)
    G  = np.triu(np.random.exponential(0.2, size=[64,64]), 1) *  10
    G +=  G.T
    t = 0.5
    G = G * (G > t)
    t = G[np.nonzero(G)].min()
    A = G>t
    print('threshold=',t)

    filename = home + '/workspace/communityalg/data/Coactivation_matrix_weighted.adj'
    G = np.loadtxt(filename)[0:64, 0:64]  *  10
    t = G[np.nonzero(G)].min()

    W = G
    A = (G>0).astype(float)
    k = A.sum(axis=0)
    s = G.sum(axis=0)
    Wtot = s.sum()
    
    n = len(A)
    m = k.sum()
    pairs = n*(n-1)/2

    M = CWTECM(N=len(G), threshold=t)
    x0 = (np.concatenate([k,s])) * 1E-4
    #x0 = np.random.random([2*len(k),])
    # # Optimize by L-BFGS-B
    opt = nq.MLEOptimizer(G, x0=x0, model=M)
    sol = opt.run(model=M, verbose=2, gtol=1E-6, method='MLE')
    print('Loglikelihood = ', M.loglikelihood(G,sol['x']))
    #print('Saddle point= ', M.saddle_point(G,sol['x']))
    # from autograd import grad
    # mygradient = grad(lambda z : M.loglikelihood(G,z))
    # print(mygradient(sol['x']))

    pij = M.expected_adjacency(sol['x'])
    wij = M.expected_weighted_adjacency(sol['x'])
    plot_mle(G,pij,wij)
    
    # opt = nq.MLEOptimizer(W, x0=x0, model=M)
    # sol = opt.run(method='saddle_point', xtol=1E-12, gtol=1E-9)
    # pij = M.expected_adjacency(sol['x'])
    # wij = M.expected_weighted_adjacency(sol['x'])
    # plot_mle(W,pij,wij)
    
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
