from pathlib import Path
home = str(Path.home())
import sys
sys.path.append(home + '/workspace/networkqit')
import matplotlib.pyplot as plt
import autograd
from autograd import numpy as np
import networkqit as nq

N = 10
M = nq.UWCM(N=N)
# X  = M.sample_adjacency( np.array([0.1,]*N) )
# print(X.shape)
# plt.hist(np.ravel(np.squeeze(X[:,:,:])))
# plt.show()

x = np.random.geometric(0.5, size=[100000,])
plt.hist(x, 200)
plt.xticks(range(20))
plt.show()