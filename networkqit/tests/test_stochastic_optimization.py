import sys
sys.path.append('/home/carlo/workspace/networkqit')
import matplotlib.pyplot as plt
import autograd
from autograd import numpy as np
import networkqit as nq
from networkqit.algorithms.stochastic_optimize_autograd import StochasticGradientDescent, Adam

A = np.loadtxt('/home/carlo/workspace/communityalg/data/karate.adj')
N = len(A)
M = nq.UBCM(N=N)
p = A.sum() / (N*(N-1))
L = nq.graph_laplacian(A)

#print(M.sample_adjacency(np.random.random([34,1])))

beta = 0.1
opt = Adam(A=A, L=L, x0=np.random.random([N,]), beta_range=np.logspace(1,-2,10))
rho = nq.VonNeumannDensity(A=None, L=L, beta=beta).density
opt.setup(model=M)

#G = autograd.grad(lambda x : np.sum(M.sample_adjacency(x)))
#x0 = np.random.random([34,])

#G = opt.gradient(x=np.random.random([N,1]), rho=rho, beta=beta, num_samples=1)
sol = opt.run(eta=1E-3, max_iters=1000, gtol=1E-5)
plt.imshow(M(sol['x']))
# plt.show()