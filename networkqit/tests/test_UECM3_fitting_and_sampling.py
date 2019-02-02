#!/usr/bin/env python3

from pathlib import Path
home = str(Path.home())
import sys
import numpy as np
import matplotlib.pyplot as plt
import bct
import sys
sys.path.append(home + '/workspace/networkqit')
import networkqit as nq
from networkqit.graphtheory.models.MEModels import CWTECM, UBCM, UWCM,UECM3

def plot(G,pij,wij):
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
    im = plt.imshow(G)
    plt.colorbar(im,fraction=0.046, pad=0.04)
    plt.grid(False)
    plt.title('empirical matrix')

    plt.subplot(2,3,4)
    plt.plot((G>0).sum(axis=0),pij.sum(axis=0), 'b.')
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
    plt.grid(True)
    plt.ylabel('model')
    plt.xlabel('empirical')

    plt.tight_layout()
    plt.show()

if __name__=='__main__':

    filename = home + '/workspace/communityalg/data/Coactivation_matrix_weighted.adj'
    G = np.loadtxt(filename)[0:32, 0:32]
    G = np.round(G*100)
    print(np.unique(G))
    print(np.unique(G))
    W = G
    A = (G>0).astype(float)
    k = A.sum(axis=0)
    m = k.sum()
    s = W.sum(axis=0)
    Wtot = s.sum()
    n = len(W)
    pairs = n*(n-1)/2

    M = UECM3(N=len(W))
    x0 = (np.concatenate([k,s])+1E-5) * 1E-3 #+ (np.random.random([2*len(W),])*2-1)*1E-5

    # Optimize by L-BFGS-B
    opt = nq.MLEOptimizer(W, x0=x0, model=M)
    sol = opt.run(model=M, verbose=0, gtol=1E-3, method='MLE')
    print('Loglikelihood = ', M.loglikelihood(G,sol['x']))
    
    pij = M.expected_adjacency(sol['x'])
    wij = M.expected_weighted_adjacency(sol['x'])
    plot(W,pij,wij)

    Asample,Wsample = M.sample_adjacency(sol['x'],batch_size=1000)
    plt.plot(np.sum(A,axis=0),np.mean(np.sum(Asample,axis=1), axis=0),'.r')
    plt.plot(np.sum(A,axis=0),np.sum(A,axis=0),'-r')
    plt.title('Sampled degrees')
    plt.xlabel('Empirical')
    plt.ylabel('Model')
    plt.show()

    plt.plot(np.mean(np.sum(Asample*(Wsample),axis=1), axis=0),np.sum(W,axis=0),'.r')
    plt.plot(np.sum(W,axis=0),np.sum(W,axis=0),'-r')
    plt.title('Sampled strengths')
    plt.xlabel('Empirical')
    plt.ylabel('Model')
    plt.show()

    beta_range=np.logspace(-2,1,100)
    Lsample = np.zeros_like(Asample)
    for i in range(Asample.shape[0]):
        Lsample[i,:,:] = nq.graph_laplacian(Asample[i,:,:]*Wsample[i,:,:])
        plt.semilogx(beta_range,nq.batch_compute_vonneumann_entropy(Lsample[i,:,:],beta_range),color='r',alpha=0.2)
    plt.semilogx(beta_range,nq.batch_compute_vonneumann_entropy(nq.graph_laplacian(W),beta_range),color='b',linewidth=2)
    plt.show()

    # nq.MLEOptimizer(W, x0=sol['x'], model=M)
    # sol = opt.run(method='saddle_point', xtol=1E-12, gtol=1E-9)
    # print('Loglikelihood = ', M.loglikelihood(G,sol['x']))
    # pij = M.expected_adjacency(sol['x'])
    # wij = M.expected_weighted_adjacency(sol['x'])
    # plot(W,pij,wij)

    # sol = opt.run(method='saddle_point', basinhopping = True, basin_hopping_niter=10, xtol=1E-9, gtol=1E-9)
    # #print('Gradient at Least squares solution=\n',grad(sol['x']))
    # print('Loglikelihood = ', M.loglikelihood(G,sol['x']))
    # pij = M.expected_adjacency(sol['x'])
    # wij = M.expected_weighted_adjacency(sol['x'])
    # plot(W,pij,wij)
