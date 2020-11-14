from pathlib import Path
home = str(Path.home())
import matplotlib.pyplot as plt
from autograd import numpy as np
import networkqit as nq

ers = np.array([[50000]])
nr = np.array([1000])
print(2*ers[0]/(nr[0]**2))

eigs_range = np.linspace(0, 100, 10000)

allz,rho = nq.compute_rho(eigs_range, ers, nr, 'laplacian')
plt.plot(allz,rho)
plt.show()