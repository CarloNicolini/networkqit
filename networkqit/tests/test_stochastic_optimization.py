from pathlib import Path
home = str(Path.home())
import sys
sys.path.append(home + '/workspace/networkqit')
import matplotlib.pyplot as plt
import autograd
from autograd import numpy as np
import networkqit as nq
from networkqit.algorithms.stochastic_optimize_autograd import StochasticGradientDescent, Adam

A = np.loadtxt(home + '/workspace/communityalg/data/karate.adj')
import networkx as nx
#A = np.loadtxt('/home/carlo2/workspace/communityalg/data/karate.adj')
A = nx.to_numpy_array(nq.ring_of_cliques(5,5))

N = len(A)
M = nq.UBCM(N=N)
p = A.sum() / (N*(N-1))
L = nq.graph_laplacian(A)

#print(M.sample_adjacency(np.random.random([34,1])))

beta = 0.1
x0 = np.random.random([N,])
opt = Adam(A=A, L=L, x0=x0, beta_range=[beta])
rho = nq.VonNeumannDensity(A=None, L=L, beta=beta).density
opt.setup(model=M)

x0 = np.random.random([34,])

#G = opt.gradient(x = np.random.random([N,]), rho=rho, beta=beta, batch_size=2)
#G(x0)
sol = opt.run(eta=1E-2, max_iters=10000, gtol=1E-3, batch_size=128)
plt.pause(5)
plt.show()
