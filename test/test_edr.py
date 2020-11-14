from pathlib import Path
home = str(Path.home())
import numpy as np
import matplotlib.pyplot as plt
import networkqit as nq
from networkqit.graphtheory.models.GraphModel import Edr


if __name__=='__main__':
    import logging
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)

    G = np.loadtxt(home + '/workspace/communityalg/data/macaque_adj_wgh.txt')
    G = np.triu(G,1)
    G += G.T
    dij = np.loadtxt(home + '/workspace/communityalg/data/macaque_dist.txt')
    L = nq.graph_laplacian(G)
    
    M = Edr(N=len(G), dij=dij)
    x0 = np.array([0.8,])
    beta_sweep = np.linspace(0.1,10,100)
    opt = nq.Adam(A=G, L=nq.graph_laplacian(G), x0=x0, beta_range=beta_sweep, model=M)
    sol = opt.run(refresh_frames=100000, eta=0.001, max_iters=5000, gtol=1E-5, batch_size=64)
    print(sol)
    plt.pause(5)
    plt.show()
