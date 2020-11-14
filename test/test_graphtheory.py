import pytest
import numpy as np
from networkqit.utils.visualization import plot_mle
import networkqit as nq
from networkqit.graphtheory.GraphModel import ErdosRenyi
from pkg_resources import resource_filename


def test_mem_models():
    filename = resource_filename('networkqit', 'networkqit/data/karate.adj.gz')
    G = np.loadtxt(filename)
    W = G
    A = (G > 0).astype(float)
    x0 = [0.5, ]
    M = ErdosRenyi(N=len(W))

    # TEST LBFGS-B
    opt = nq.MLEOptimizer(W, x0=x0, model=M)
    sol = opt.run(model=M, verbose=0, gtol=1E-8, method='MLE')
    loglike = M.loglikelihood(G, sol['x'])

    # TEST SAMPLING
    S = M.sample_adjacency(sol['x'], batch_size=500, with_grads=False).mean(axis=0)
    plot_mle(W, S.astype(float), None, title='Sampling')

    # TEST SADDLE POINT
    opt = nq.MLEOptimizer(W, x0=x0, model=M)
    sol = opt.run(method='saddle_point', verbose=2, gtol=1E-9)
    M.loglikelihood(G, sol['x'])
    pij = M.expected_adjacency(sol['x'])
    plot_mle(W, pij, None, title='Saddle point method')
