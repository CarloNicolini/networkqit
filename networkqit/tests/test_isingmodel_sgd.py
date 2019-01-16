from pathlib import Path
home = str(Path.home())
import sys
sys.path.append(home + '/workspace/networkqit')
import matplotlib.pyplot as plt
import autograd
from autograd import numpy as np
import networkqit as nq
from networkqit import Adam
import networkx as nx

#A = np.loadtxt('/home/carlo2/workspace/communityalg/data/karate.adj')

A = nq.ring_of_custom_cliques([12,8,4,2])
N = len(A)
M = nq.IsingModel(N=N)
p = A.sum() / (N*(N-1))
L = nq.graph_laplacian(A)

beta = 5.0
opt = Adam(A=A, L=L, x0=np.random.random([N*N,]), beta_range=np.logspace(1,-2,50))
rho = nq.VonNeumannDensity(A=None, L=L, beta=beta).density
opt.setup(model=M)

#G = autograd.grad(lambda x : np.sum(M.sample_adjacency(x)))
#x0 = np.random.random([34,])

#G = opt.gradient(x=np.random.random([N,1]), rho=rho, beta=beta, num_samples=1)
sol = opt.run(eta=1E-3, max_iters=np.inf, gtol=1E-5, batch_size=64)
plt.pause(5)
plt.show()