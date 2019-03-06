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
from networkqit.graphtheory.models.GraphModel import Edr
from networkqit.utils.visualization import plot_mle

if __name__=='__main__':

    G = np.loadtxt('/home/carlo2/workspace/communityalg/data/macaque_adj_wgh.txt')
    G = np.triu(G,1)
    G += G.T
    dij = np.loadtxt('/home/carlo2/workspace/communityalg/data/macaque_dist.txt')
    L = nq.graph_laplacian(G)
    
    M = Edr(N=len(G), dij=dij)
    # beta_range = np.logspace(3,-3,100)
    # Ws = M.sample_adjacency(theta=np.array([0.001,0.188]), batch_size=1000, with_grads=False)
    # plt.semilogx(beta_range,nq.batch_compute_vonneumann_entropy(L=L,beta_range=beta_range))
    # plt.semilogx(beta_range,nq.batch_compute_vonneumann_entropy(L=nq.graph_laplacian(Ws[0,:,:]),beta_range=beta_range))
    # plt.show()

    #x0 = np.random.random([2,])
    x0 = np.array([1,0.05])
    beta_sweep = [3]
    opt = nq.Adam(A=G, L=nq.graph_laplacian(G), x0=x0, beta_range=beta_sweep, model=M)
    sol = opt.run(refresh_frames=100, eta=0.001, max_iters=5000, gtol=1E-5, batch_size=256)
    print(sol)
    plt.pause(5)
    plt.show()
