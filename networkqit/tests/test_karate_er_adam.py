import sys
sys.path.append('/home/carlo/workspace/networkqit/')
import networkqit as nq
import numpy as np

G = np.loadtxt('/home/carlo/workspace/communityalg/data/karate.adj')

M = nq.ErdosRenyi(N=len(G))
opt = nq.Adam(A=G,x0=np.array([0.4,]),model=M, beta_range=np.linspace(10,0.01,100))
sol = opt.run(refresh_frames=1000, eta=1E-5, max_iters=100, gtol=1E-5, batch_size=1024)