import sys
sys.path.append('/home/carlo2/workspace/networkqit')
import matplotlib.pyplot as plt
import networkqit as nq
import numpy as np

A = np.loadtxt('/home/carlo2/workspace/communityalg/data/karate.adj')
p = A.sum()/(34*33 -1)
print('p=',p)
L = nq.graph_laplacian(A)

from networkqit.algorithms.stochastic_optimize_autograd import StochasticGradientDescent

beta=1
opt = StochasticGradientDescent(A=A, L=L, x0=np.array([0.5,]), beta_range=[beta])
rho = nq.VonNeumannDensity(A=None, L=L, beta=beta).density

M = nq.ErdosRenyi(N=len(A))

opt.setup(model=M)
from autograd import numpy as anp

G = opt.gradient(x=anp.array([p,]), rho=rho, beta=beta, num_samples=1)
print(G)



# import autograd
# def getA(x):
#     rij = anp.random.random([5,5])
#     rij = anp.triu(rij,1)
#     rij += rij.T
#     slope = 1000
#     A = 1.0/(1.0 + anp.exp(-slope*(anp.ones([5,5])*x-rij)))
#     return A

# print(autograd.elementwise_grad(getA)(anp.array([0.5])))