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
import networkx as nx

#A = np.loadtxt('/home/carlo/workspace/communityalg/data/karate.adj')

G = np.loadtxt('/home/carlo/workspace/communityalg/data/macaque_adj_wgh.txt')
dij = np.loadtxt('/home/carlo/workspace/communityalg/data/macaque_dist.txt')

N = len(G)
M = nq.EconomicalClustering(dij=dij, G=G, N=len(G))
L = nq.graph_laplacian(G)

beta = 1
opt = Adam(G=G, L=L, x0=np.array([1.0,1.0]), model=M)
rho = nq.compute_vonneuman_density(L=L, beta=beta)

for rep in range(10):
      sol = opt.run(beta, learning_rate=1E-3, batch_size=128, maxiter=100)
      nq.plot_mle(A, M.expected_adjacency(sol['x']))
      plt.show()

#sol = opt.run(refresh_frames=5000, quasi_hyperbolic=True, eta=1E-3, max_iters=5, gtol=1E-5, batch_size=64)


            #all_dkl.append(dkl)
            #print(np.hstack([t, x,beta,dkl]))
            #logfile.write( u"%g\t%g\t%g\n" % (x,beta,dkl) )
            # if t % refresh_frames == 0:
            #     frames += 1
            #     def draw_fig():
            #         plot_beta_range = np.logspace(-3,3,100)
            #         sol.append({'x': x.copy()})
            #         #plt.figure(figsize=(8, 8))
            #         A0 = np.mean(self.model.sample_adjacency(theta=x, batch_size=batch_size), axis=0)
            #         plt.subplot(2, 2, 1)
            #         im = plt.imshow(self.A)
            #         plt.colorbar(im)
            #         plt.title('Data')
            #         plt.subplot(2, 2, 2)
            #         im = plt.imshow(A0)
            #         plt.colorbar(im)
            #         plt.title('<Model>')
            #         plt.subplot(2, 2, 3)
            #         plt.plot(all_dkl)
            #         plt.xlabel('iteration')
            #         plt.ylabel('$S(\\rho,\\sigma)$')
            #         plt.subplot(2, 2, 4)
            #         plt.semilogx(plot_beta_range, batch_compute_vonneumann_entropy(self.L, plot_beta_range), '.-', label='data')
            #         plt.semilogx(plot_beta_range, batch_compute_vonneumann_entropy(graph_laplacian(A0), plot_beta_range), '.-', label='model')
            #         plt.plot(beta, batch_compute_vonneumann_entropy(graph_laplacian(A0), [beta]), 'ko', label='model')
            #         plt.xlabel('$\\beta$')
            #         plt.ylabel('$S$')
            #         plt.title('Entropy')
            #         plt.legend(loc='best')
            #         plt.suptitle('$\\beta=$' + '{0:0>3}'.format(beta))
            #         #plt.tight_layout()
            #     drawnow(draw_fig)


plt.pause(5)
plt.show()
