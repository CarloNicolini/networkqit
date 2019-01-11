#!/usr/bin/env python3

import sys
sys.path.append('/home/carlo/workspace/networkqit/')

import numpy as np
import matplotlib.pyplot as plt
import bct
import networkqit as nq
from networkqit.graphtheory.models.MEModels import cWECMt1, cWECMt2, UBCM, UWCM
filename = '/home/carlo/workspace/communityalg/data/Coactivation_matrix_weighted.adj'
W = np.loadtxt(filename)[0:16,0:16]
threshold = 0.0
W = bct.threshold_absolute(W, threshold)
A = (W>0).astype(float)
k = A.sum(axis=0)
m = k.sum()
s = W.sum(axis=0)
Wtot = s.sum()
n = len(W)
pairs = n*(n-1)/2

M = cWECMt2(N=len(W), threshold=threshold)
#x0 = np.random.random([2*len(W),])
x0 = np.concatenate([k,s]) #+ (np.random.random([2*len(W),])*2-1)*1E-5
#x0 = np.random.random([2*len(W),1])*1E-2
# Optimize part with basinhopping
opt = nq.MLEOptimizer(W, x0=x0)
sol = opt.runfsolve(model=M, basinhopping=True, verbose=2, xtol=1E-5, gtol=1E-5, basin_hopping_niter=100)
print(sol.keys())
print('Optimization done...')

pij = M.expected_adjacency(sol['x'])
wij = M.expected_weighted_adjacency(sol['x'])

############## PLOTTING PART ############## 

plt.figure(figsize=(12,8))
plt.subplot(2,3,1)
im = plt.imshow(pij)
plt.colorbar(im,fraction=0.046, pad=0.04)
plt.grid(False)
plt.title('$p_{ij}$')

plt.subplot(2,3,2)
im = plt.imshow(wij)
plt.colorbar(im,fraction=0.046, pad=0.04)
plt.grid(False)
plt.title('$<w_{ij}>$')

plt.subplot(2,3,3)
im = plt.imshow(W)
plt.colorbar(im,fraction=0.046, pad=0.04)
plt.grid(False)
plt.title('empirical matrix')

plt.subplot(2,3,4)
plt.plot((W>0).sum(axis=0),pij.sum(axis=0), 'b.')
plt.plot(np.linspace(0,pij.sum(axis=0).max()),np.linspace(0,pij.sum(axis=0).max()),'r-')
plt.grid(True)
plt.axis('equal')
plt.title('Degrees reconstruction')
plt.ylabel('model')
plt.xlabel('empirical')
#plt.xlim([0,min((W>0).sum(axis=0).max(),pij.sum(axis=0).max())])
#plt.ylim([0,min((W>0).sum(axis=0).max(),pij.sum(axis=0).max())])

plt.subplot(2,3,5)
plt.plot(W.sum(axis=0),wij.sum(axis=0), 'b.')
plt.plot(np.linspace(0,wij.sum(axis=0).max()),np.linspace(0,wij.sum(axis=0).max()),'r-')
plt.title('Strength reconstruction')
plt.axis('equal')
#plt.xlim([0,wij.sum(axis=0).max()])
#plt.ylim([0,wij.sum(axis=0).max()])
plt.grid(True)
plt.ylabel('model')
plt.xlabel('empirical')

plt.subplot(2,3,6)
R = np.random.random([len(W),len(W)])
R = (R+R.T)/2
As = (R < pij)
Ws = np.triu(np.random.exponential(wij),1)
Ws += Ws.T
Ws *= As
im = plt.imshow(Ws)
#print('Empirical Density',bct.density_und(W))
#print('Sampled Density',bct.density_und(Ws))
plt.colorbar(im,fraction=0.046, pad=0.04)
plt.title('Sampled matrix')
plt.tight_layout()
plt.show()