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
from networkqit.graphtheory.models.MEModels import CWTECM, UBCM, UWCM
from networkqit import plot_mle
if __name__=='__main__':

    filename = home + '/workspace/communityalg/data/Coactivation_matrix_weighted.adj'
    G = np.loadtxt(filename)[0:64, 0:64]
    G = np.round(G*50)
    print(np.unique(G))
    W = G
    A = (G>0).astype(float)
    k = A.sum(axis=0)
    m = k.sum()
    s = W.sum(axis=0)
    Wtot = s.sum()
    n = len(W)
    pairs = n*(n-1)/2

    M = UWCM(N=len(W))
    x0 = (np.concatenate([s])+1E-5)*1E-3 #+ (np.random.random([2*len(W),])*2-1)*1E-5
    # Optimize by L-BFGS-B
    opt = nq.MLEOptimizer(W, x0=x0, model=M)
    sol = opt.run(model=M, verbose=0,gtol=1E-8, method='MLE')
    print('Loglikelihood = ', M.loglikelihood(G,sol['x']))

    # pij = M.expected_adjacency(sol['x'])
    # wij = M.expected_weighted_adjacency(sol['x'])
    # plot(W,pij,wij)

    # opt = nq.MLEOptimizer(W, x0=x0, model=M)
    # sol = opt.run(method='saddle_point', xtol=1E-12, gtol=1E-9)
    # print('Loglikelihood = ', M.loglikelihood(G,sol['x']))
    # pij = M.expected_adjacency(sol['x'])
    # wij = M.expected_weighted_adjacency(sol['x'])
    # plot(W,pij,wij)


    # TEST SAMPLING
    S = M.sample_adjacency(sol['x'],batch_size=10, with_grads=False).mean(axis=0)
    print(S)
    plot_mle(W,(S>0).astype(float),S)

    # sol = opt.run(method='saddle_point', basinhopping = True, basin_hopping_niter=10, xtol=1E-9, gtol=1E-9)
    # #print('Gradient at Least squares solution=\n',grad(sol['x']))
    # print('Loglikelihood = ', M.loglikelihood(G,sol['x']))
    # pij = M.expected_adjacency(sol['x'])
    # wij = M.expected_weighted_adjacency(sol['x'])
    # plot(W,pij,wij)
