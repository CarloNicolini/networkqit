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

    #np.random.seed(0)
    G  = np.triu(np.random.exponential(0.2, size=[64,64]), 1) *  10
    G +=  G.T
    t = 0.5
    G = G * (G > t)
    t = G[np.nonzero(G)].min()
    A = G>t
    print('threshold=',t)

    filename = home + '/workspace/communityalg/data/GroupAverage_rsfMRI_weighted.adj'
    G = np.loadtxt(filename)#[0:64, 0:64]
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
    x0 = np.concatenate([k,s])
    x0 = np.clip(x0/x0.max(), np.finfo(float).eps, 1-np.finfo(float).eps ) # to make it in [0,1]
    # TEST LBFGS-B
    opt = nq.MLEOptimizer(W, x0=x0, model=M)
    sol = opt.run(method='MLE', model=M, maxiter=100, verbose=2, gtol=1E-6)
    print('Loglikelihood = ', M.loglikelihood(G,sol['x']))

    pij = M.expected_adjacency(sol['x'])
    wij = M.expected_weighted_adjacency(sol['x'])
    plot_mle(W,pij,wij,title='Loglikelihood LBFGS-B')

    # TEST JACOBIAN
    from autograd import grad
    print('|grad|=',np.sqrt((grad(lambda z : M.loglikelihood(G,z))(sol['x'])**2).sum()))
    print('saddle_point=', np.sqrt((M.saddle_point(G,sol['x'])**2)).sum())

    # TEST SAMPLING
    Sd = (M.sample_adjacency(sol['x'], batch_size=1, with_grads=False)>0)[0,:,:]#.mean(axis=0)
    S = (M.sample_adjacency(sol['x'], batch_size=1, with_grads=False))[0,:,:]#.mean(axis=0)
    plot_mle(W, Sd.astype(float), S, title='Sampling')

    # TEST SAMPLING
    Sd = (M.sample_adjacency(sol['x'], batch_size=1, with_grads=False)>0).mean(axis=0)
    S = (M.sample_adjacency(sol['x'], batch_size=1, with_grads=False)).mean(axis=0)
    plot_mle(W, Sd.astype(float), S, title='Sampling')
    

    # TEST SADDLE POINT
    opt = nq.MLEOptimizer(W, x0=x0, model=M)
    sol = opt.run(method='saddle_point', verbose=2, xtol=1E-9, gtol=1E-9)
    print('Loglikelihood = ', M.loglikelihood(G,sol['x']))
    pij = M.expected_adjacency(sol['x'])
    wij = M.expected_weighted_adjacency(sol['x'])
    plot_mle(W,pij,wij,title='Saddle point method')

    # TEST BASINHOPPING
    opt = nq.MLEOptimizer(W, x0=x0, model=M)
    sol = opt.run(method='saddle_point', verbose=2, basinhopping = True, basin_hopping_niter=10, xtol=1E-9, gtol=1E-9)
    #print('Gradient at Least squares solution=\n',grad(sol['x']))
    print('Loglikelihood = ', M.loglikelihood(G,sol['x']))
    pij = M.expected_adjacency(sol['x'])
    wij = M.expected_weighted_adjacency(sol['x'])
    plot_mle(W,pij,wij)