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
A = nq.ring_of_custom_cliques([8,4,8,4])

N = len(A)
M = nq.UBCM(N=N)
p = A.sum() / (N*(N-1))
L = nq.graph_laplacian(A)

beta_sweep = np.logspace(1.2,-3,50)
plt.figure()
#plt.imshow(nq.compute_vonneuman_density(L=L, beta=1))
plt.semilogx(np.logspace(3,-3,100),nq.batch_compute_vonneumann_entropy(L=L,beta_range=np.logspace(3,-3,100)))
plt.semilogx([beta_sweep[0]],nq.batch_compute_vonneumann_entropy(L=L,beta_range=[beta_sweep[0]]),'ko')
plt.show()
#print(M.sample_adjacency(np.random.random([34,1])))

x0 = np.random.random([N,])
opt = Adam(A=A, L=L, x0=x0, beta_range=beta_sweep, model=M)
rho = nq.compute_vonneuman_density(L=L, beta=0.1)
sol = opt.run(refresh_frames=10, eta=0.01, max_iters=1500, gtol=1E-5, batch_size=32)
plt.pause(5)
plt.show()
