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

#A = np.loadtxt('/home/carlo/workspace/communityalg/data/karate.adj')
A = nq.sbm(np.array([[1500,1500],[1500,1500]]),[64,64])
np.savetxt('matrix.dat',A)
print(A.shape)
#A = nq.ring_of_custom_cliques([24,12,8])
N = len(A)
M = nq.ErdosRenyi(N=N)
p = A.sum() / (N*(N-1))
print(p)
L = nq.graph_laplacian(A)

from scipy.optimize import OptimizeResult
sol = OptimizeResult(x=np.array([0.2]))

plt.semilogx(np.logspace(-3,3,100),nq.density2.entropy(L=L,beta_range=np.logspace(-3,3,100)))
plt.grid()
plt.title('Entropy')
plt.show()
opt = Adam(G=A, x0=sol['x'], model=M)
burnin = 1000
sample_iteration = 10

def get_beta_at_s(L,beta_range,Sval):
    return beta_range[np.where(nq.density2.entropy(nq.graph_laplacian(A),beta_range=beta_range) > Sval)[0][-1]]

minbeta = get_beta_at_s(nq.graph_laplacian(A),np.logspace(-3,3,200),0.05)
print(minbeta)

for i,beta in enumerate(np.linspace(0.5,0.01,100)):
	opt.sol['x'] = sol['x']
	sol = opt.run(beta, learning_rate=1E-3, batch_size=2, maxiter= burnin if i==0 else sample_iteration, gtol=1E-8)
