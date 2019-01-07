#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 14:41:36 2018

@author: carlo
"""

import sys
sys.path.append('/home/carlo/workspace/networkqit/')

import numpy as np
import matplotlib.pyplot as plt
import bct
import networkqit as nq
from networkqit.graphtheory.models.ExpectedGraphModel import UBCM, UWCM, UECM, cWECMt1, cWECMt2


W = np.loadtxt('/home/carlo/workspace/communityalg/data/GroupAverage_rsfMRI_unthr.adj')[0:32,0:32]
t = 0.4
W = bct.threshold_absolute(W,t)
A = (W>0).astype(float)
k = A.sum(axis=0)
m = k.sum()
s = W.sum(axis=0)
Wtot = s.sum()
n = len(W)
pairs = n*(n-1)/2

M = UBCM(N=len(W), threshold=t)
#x0 = np.concatenate([k,s])/pairs
x0 = k/pairs
#x0 = np.random.random([2*len(W),])

opt = nq.MLEOptimizer(W, x0=x0)
sol = opt.run(M,verbose=2)
#opt = nq.MLEOptimizer(W, x0=sol['x'])
sol = opt.runfsolve(model='UBCM', threshold=t, verbose=2)
print('Final cost=',sol['cost'])

pij = M.expected_adjacency(*sol['x'])
#wij = M.adjacency_weighted(*sol['x'])


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
print('Empirical Density',bct.density_und(W))
print('Sampled Density',bct.density_und(Ws))
plt.colorbar(im,fraction=0.046, pad=0.04)
plt.title('Sampled matrix')
plt.tight_layout()
plt.ion()
plt.show()