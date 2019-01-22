#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 23:02:25 2018

@author: carlo
"""

import sys
sys.path.append('/home/carlo/workspace/networkqit/')
import numpy as np
import matplotlib.pyplot as plt
import networkqit as nq
from networkqit.graphtheory.models.MEModels import UBCM,UWCM
import networkx as nx

#A = nx.to_numpy_array(nx.karate_club_graph())
filename = '/home/carlo/workspace/communityalg/data/Coactivation_matrix_weighted.adj'
A = np.loadtxt(filename)[0:128,0:128]
#A = (A>0).astype(float)
k = A.sum(axis=0)
n = len(A)
pairs = n*(n-1)/2
m = k.sum()

M = UWCM(N=len(A))
x0 = np.random.random([len(A),])
opt = nq.MLEOptimizer(A, x0=x0)
sol = opt.run(model=M)
print('Max likelihood solution=', M.loglikelihood(*sol['x']))

plt.figure()
pij = M.expected_adjacency(*sol['x'])
plt.plot(pij.sum(axis=0),A.sum(axis=0),'or')
#plt.plot(pij.sum(axis=0),pij.sum(axis=0),'-k')
plt.grid(True)
plt.show()

# RUN THE MAXIMUM LIKELIHOOD METHOD
sol = opt.runfsolve(model=M)
print('Saddle point solution=', M.loglikelihood(*sol['x']))
plt.figure()
pij = M.expected_adjacency(*sol['x'])
plt.plot(pij.sum(axis=0),A.sum(axis=0),'or')
#plt.plot(pij.sum(axis=0),pij.sum(axis=0),'-k')
plt.grid(True)
plt.show()