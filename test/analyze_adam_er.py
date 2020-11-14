import numpy as np
import matplotlib.pyplot as plt

plt.style.use('default')
A = np.loadtxt('matrix.dat')
p = np.sum(A)/(len(A)*(len(A)-1))
print(p)
X = np.loadtxt('adam.log')

#densities = X[1000:,0]
#densities = densities.reshape([99,50])

#dkl = X[1000:,2]
#dkl = dkl.reshape([99,50])

# these are the data at the change points of beta
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(range(X.shape[0]),X[:,2],linewidth=0.5, color='b')
ax2.plot(range(X.shape[0]),X[:,0], linewidth=0.5,color='grey')


ax1.set_xlabel('Iteration')
ax1.set_ylabel('$\\beta^{-1} Tr[\\rho \\ \\log \\sigma]$')
ax2.set_ylabel('p')
#ax1.set_ylim([0,350])
ax2.set_ylim([0,1])
#ax2.axhline(y=0.13903743315508021, color='g')
ax2.axhline(y=p, color='r', LineStyle='-', linewidth=1)
#ax2.fill_between(x=np.arange(1000,X.shape[0],50),y1=p+np.std(densities,axis=1), y2=p-np.std(densities,axis=1),color='grey', step='mid', alpha=0.5, edgecolor=None)
#ax1.fill_between(x=np.arange(1000,X.shape[0],50),y1=np.mean(dkl,1)+100*np.std(dkl,axis=1), y2=np.mean(dkl,1)-100*np.std(dkl,axis=1),color='b',step='mid',edgecolor=None, alpha=0.5)
#ax2.set_ylim([0,1])

#for i in np.arange(1000,10000,1000):
#	ax2.axvline(x=i,color='k',alpha=0.2)
#std = np.std(X[1000:X.shape[0]:50])
#X[1000:].reshape([50,len(X[])])
plt.title('$N=128$')
plt.tight_layout()
plt.savefig('adam_er.pdf')

#import matplotlib2tikz
#matplotlib2tikz.save('adam_er_karate.tex')

#plt.errorbar(x=np.arange(1000,X.shape[0],50),y=np.zeros_like(np.arange(1000,X.shape[0],50)),yerr=np.std(Z,axis=1),barsabove=True)

# Y = X[np.where(np.diff(X[:,1]))[0]+1,:]
# plt.figure()
# plt.plot(Y[:,1],Y[:,0],color='b',label='$\lambda$')
# plt.savefig('adam_er_karate.pdf')