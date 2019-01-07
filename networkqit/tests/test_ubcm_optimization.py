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
from networkqit.graphtheory.models.MEModels import UBCM
import networkx as nx

A = nx.to_numpy_array(nx.karate_club_graph())
k = A.sum(axis=0)
n = len(A)
pairs = n*(n-1)/2
m = k.sum()

M = UBCM(N=len(A))

x0 = np.random.random([len(A),])
x0 = k/1000
x0 = np.ones_like(x0)
opt = nq.MLEOptimizer(A, x0=x0)

#sol = opt.run(M, xtol=1E-32, gtol=1E-32)
#print('Maximum likelihood solution=',-M.loglikelihood(*sol['x']))
#
#plt.figure()
#pij = M.expected_adjacency(*sol['x'])
#plt.plot(pij.sum(axis=0),A.sum(axis=0),'or')
#plt.plot(pij.sum(axis=0),pij.sum(axis=0),'-k')
#plt.grid(True)
#plt.show()


plt.figure()
sol = opt.runfsolve(model=UBCM(N=len(A)), xtol=1E-16)
print('Saddle point solution=', M.loglikelihood(*sol['x']))
pij = M.expected_adjacency(*sol['x'])
plt.plot(pij.sum(axis=0),A.sum(axis=0),'or')
#plt.plot(pij.sum(axis=0),pij.sum(axis=0),'-k')
plt.grid(True)
plt.show()
#
#sol = opt.run(M)
#
#print(M.loglikelihood(*sol['x']))