from pathlib import Path
home = str(Path.home())
import sys
sys.path.append(home + '/workspace/networkqit')
import matplotlib.pyplot as plt
from autograd import numpy as np
import networkqit as nq

#A = np.loadtxt('/home/linello/workspace/communityalg/data/karate.adj')

A = nq.ring_of_custom_cliques([16]*4)
N = len(A)
M = nq.IsingModel(N=N)
p = A.sum() / (N*(N-1))
L = nq.graph_laplacian(A)

Sstar = 1.0
beta = nq.find_beta_logc(L,np.exp(Sstar))
beta_range=np.logspace(-3,3,100)
S = nq.entropy(L,beta_range=beta_range)
plt.semilogx(beta_range,S)
plt.semilogx(beta,nq.entropy(L,beta_range=[beta]),'ro')
plt.show()

x0=np.random.random([N*N,])
for _ in range(100):
	opt = nq.Adam(G=A, L=L, x0=x0, model=M)
	rho = nq.density2.density(L=L, beta_range=[beta])
	sol = opt.run(beta, learning_rate=1E-3*5, batch_size=4, maxiter=200)
	x0=sol['x']
	nq.plot_mle(A, M.expected_adjacency(sol['x']) )
	plt.show()
