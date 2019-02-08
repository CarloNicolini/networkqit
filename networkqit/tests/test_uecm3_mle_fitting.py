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
from networkqit import plot_mle

if __name__=='__main__':

    filename = home + '/workspace/communityalg/data/Coactivation_matrix_weighted.adj'
    G = np.loadtxt(filename)[0:50,0:50]
    W = np.round(G*50)
    A = (G>0).astype(float)
    k = A.sum(axis=0)
    m = k.sum()
    s = W.sum(axis=0)
    Wtot = s.sum()
    n = len(W)
    pairs = n*(n-1)/2

    M = UECM3(N=len(W))
    x0 = np.concatenate([k,s])
    x0 = np.clip(x0/x0.max(), np.finfo(float).eps, 1-np.finfo(float).eps ) # to make it in [0,1]

    # TEST LBFGS-B
    opt = nq.MLEOptimizer(W, x0=x0, model=M)
    sol = opt.run(model=M, verbose=2, gtol=1E-8, method='MLE')
    print('Loglikelihood = ', M.loglikelihood(G,sol['x']))

    pij = M.expected_adjacency(sol['x'])
    wij = M.expected_weighted_adjacency(sol['x'])
    plot_mle(W,pij,wij,title='Loglikelihood method')

    # TEST JACOBIAN
    from autograd import grad
    print('|grad|=',np.sqrt((grad(lambda z : M.loglikelihood(G,z))(sol['x'])**2).sum()))
    print('saddle_point=', np.sqrt((M.saddle_point(G,sol['x'])**2)).sum())

    # TEST SAMPLING
    Sd = (M.sample_adjacency(sol['x'], batch_size=500, with_grads=False)>0).mean(axis=0)
    S = (M.sample_adjacency(sol['x'], batch_size=500, with_grads=False)).mean(axis=0)
    plot_mle(W, Sd.astype(float), S, title='Sampling')
    

    # TEST SADDLE POINT
    opt = nq.MLEOptimizer(W, x0=x0, model=M)
    sol = opt.run(method='saddle_point', gtol=1E-9)
    print('Loglikelihood = ', M.loglikelihood(G,sol['x']))
    pij = M.expected_adjacency(sol['x'])
    wij = M.expected_weighted_adjacency(sol['x'])
    plot_mle(W,pij,wij,title='Saddle point method')

    # TEST BASINHOPPING
    opt = nq.MLEOptimizer(W, x0=x0, model=M)
    sol = opt.run(method='saddle_point', basinhopping = True, basin_hopping_niter=10, xtol=1E-9, gtol=1E-9)
    #print('Gradient at Least squares solution=\n',grad(sol['x']))
    print('Loglikelihood = ', M.loglikelihood(G,sol['x']))
    pij = M.expected_adjacency(sol['x'])
    wij = M.expected_weighted_adjacency(sol['x'])
    plot_mle(W,pij,wij)
