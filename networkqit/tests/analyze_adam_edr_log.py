import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')

X = np.loadtxt('adam.log')
# these are the data at the change points of beta
Y = X[np.where(np.diff(X[:,1]))[0]+1,:]

plt.figure()
plt.plot(Y[:,1],Y[:,0],color='b',label='$\lambda$')
plt.axhline(y=0.174,color='r',label='$\lambda^*$')
plt.title('$\\lambda$')
plt.xlabel('$\\beta$')
plt.ylabel('$\\lambda$')
plt.legend()

plt.figure()
plt.plot(Y[:,1],Y[:,2],color='k',label='dkl')
plt.title('Relative entropy')
plt.xlabel('$\\beta$')
plt.ylabel('$S(\\rho \\| \\sigma )$')

plt.show()