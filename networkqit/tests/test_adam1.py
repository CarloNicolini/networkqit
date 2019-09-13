#!/usr/bin/python3
from pathlib import Path
home = str(Path.home())
import sys
sys.path.append(home + '/workspace/networkqit')
import matplotlib.pyplot as plt
import autograd
from autograd import numpy as np
import networkqit as nq
from networkqit import Adam

A = np.loadtxt(home + '/workspace/communityalg/data/karate.adj')
import networkx as nx
A = nq.ring_of_custom_cliques([40,20,40,20])

M = nq.UBCM(N=len(A))
L = nq.graph_laplacian(A)

beta_sweep = [100,]
plt.semilogx(np.logspace(3,-3,100),nq.entropy(L=L,beta_range=np.logspace(3,-3,100)))
plt.semilogx([beta_sweep[0]],nq.entropy(L=L,beta_range=[beta_sweep[0]]),'ko')
plt.show()

x0 = np.random.random([len(A),])
opt = Adam(G=A, L=L, x0=x0, beta_range=beta_sweep, model=M)
sol = opt.run(refresh_frames=50, eta=0.01, max_iters=1500, gtol=1E-5, batch_size=4)
plt.pause(5)
plt.show()
