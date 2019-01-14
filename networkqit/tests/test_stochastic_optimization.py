import sys
sys.path.append('/home/carlo/workspace/networkqit')
import matplotlib.pyplot as plt
import networkqit as nq
from autograd import numpy as np
from networkqit.algorithms.stochastic_optimize_autograd import StochasticGradientDescent, Adam

A = np.loadtxt('/home/carlo/workspace/communityalg/data/karate.adj')
N = len(A)
M = nq.UBCM(N=N)
p = A.sum() / (N*(N-1))
L = nq.graph_laplacian(A)

#print(M.sample_adjacency(np.random.random([34,1])))

beta = 1
opt = StochasticGradientDescent(A=A, L=L, x0=np.random.random([N,1]), beta_range=[beta])
rho = nq.VonNeumannDensity(A=None, L=L, beta=beta).density
opt.setup(model=M)
import autograd
G = autograd.grad(lambda x : np.sum(M.sample_adjacency(x)))
print(G(np.random.random([34,1])))
#G = opt.gradient(x=np.random.random([N,]), rho=rho, beta=beta, num_samples=1)
#sol = opt.run(eta=1E-3, max_iters=1000, gtol=1E-6)
# plt.plot([s['rel_entropy'] for s in sol])
# plt.show()