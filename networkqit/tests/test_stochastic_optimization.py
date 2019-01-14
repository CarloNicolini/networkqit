import sys
sys.path.append('/home/carlo2/workspace/networkqit')
import matplotlib.pyplot as plt
import networkqit as nq
from autograd import numpy as np
from networkqit.algorithms.stochastic_optimize_autograd import StochasticGradientDescent

N = 100
M = nq.ErdosRenyi(N=N)
A = M.sample_adjacency(0.2)
p = A.sum()/(N*(N-1))
print('p=',p)
L = nq.graph_laplacian(A)
plt.imshow(A)
plt.show()

beta=0.1
opt = StochasticGradientDescent(A=A, L=L, x0=np.array([0.3,]), beta_range=[beta])
rho = nq.VonNeumannDensity(A=None, L=L, beta=beta).density

opt.setup(model=M)
#G = opt.gradient(x=np.array([p,]), rho=rho, beta=beta, num_samples=1)
opt.run(eta=1E-4,max_iters=2000)
