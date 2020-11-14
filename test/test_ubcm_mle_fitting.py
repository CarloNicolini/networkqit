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
    filename = home + '/workspace/communityalg/data/karate.adj'
    G = np.loadtxt(filename)
    W = G
    A = (G>0).astype(float)
    k = A.sum(axis=0)
    m = k.sum()
    s = W.sum(axis=0)
    Wtot = s.sum()
    n = len(W)
    pairs = n*(n-1)/2

    M = UBCM(N=len(W))
    x0 = k / k.max()
        
    # TEST LBFGS-B
    opt = nq.MLEOptimizer(W, x0=x0, model=M)
    sol = opt.run(model=M, verbose=0, gtol=1E-8, method='MLE')
    print('Loglikelihood = ', M.loglikelihood(G,sol['x']))

    pij = M.expected_adjacency(sol['x'])
    plot_mle(W,pij,None,title='Loglikelihood method')

    # TEST SAMPLING
    S = M.sample_adjacency(sol['x'],batch_size=500, with_grads=False).mean(axis=0)
    plot_mle(W,S.astype(float),None, title='Sampling')

    # TEST SADDLE POINT
    opt = nq.MLEOptimizer(W, x0=np.random.random([34,]), model=M)
    sol = opt.run(method='saddle_point', verbose=2, gtol=1E-9)
    print('Loglikelihood = ', M.loglikelihood(G,sol['x']))
    pij = M.expected_adjacency(sol['x'])
    plot_mle(W,pij,None,title='Saddle point method')
