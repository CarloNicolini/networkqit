#!/usr/bin/env python
"""
==========
Properties
==========

Visualization utilities for model optimization and entropy visualization
"""
#    Copyright (C) 2018 by
#    Carlo Nicolini <carlo.nicolini@iit.it>
#    All rights reserved.
#    BSD license.


import matplotlib.pyplot as plt
from drawnow import drawnow
import autograd.numpy as np
import seaborn as sns
from networkqit.infotheory.density import compute_vonneuman_density, compute_vonneumann_entropy
from networkqit.graphtheory import graph_laplacian as graph_laplacian

import matplotlib.gridspec as gridspec

def plot_mle(G,pij,wij=None):
    plt.figure(figsize=(12,8))
    plt.subplot(2,3,1)
    im = plt.imshow(pij)
    plt.colorbar(im,fraction=0.046, pad=0.04)
    plt.grid(False)
    plt.title('$p_{ij}$')

    if wij is not None:
        plt.subplot(2,3,2)
        im = plt.imshow(wij)
        plt.colorbar(im,fraction=0.046, pad=0.04)
        plt.grid(False)
        plt.title('$<w_{ij}>$')

    plt.subplot(2,3,3)
    im = plt.imshow(G)
    plt.colorbar(im,fraction=0.046, pad=0.04)
    plt.grid(False)
    plt.title('Weighted empirical')

    plt.subplot(2,3,4)
    plt.plot((G>0).sum(axis=0),pij.sum(axis=0), 'b.')
    plt.plot(np.linspace(0,pij.sum(axis=0).max()),np.linspace(0,pij.sum(axis=0).max()),'r-')
    plt.grid(True)
    plt.axis('equal')
    plt.title('Degrees reconstruction')
    plt.ylabel('model')
    plt.xlabel('empirical')

    if wij is not None:
        plt.subplot(2,3,5)
        plt.plot(G.sum(axis=0),wij.sum(axis=0), 'b.')
        plt.plot(np.linspace(0,wij.sum(axis=0).max()),np.linspace(0,wij.sum(axis=0).max()),'r-')
        plt.title('Strength reconstruction')
        plt.axis('equal')
        plt.grid(True)
        plt.ylabel('model')
        plt.xlabel('empirical')

    plt.subplot(2,3,6)
    plt.imshow(G>0)
    plt.title('Binary empirical')

    plt.tight_layout()
    plt.show()

def step_callback(A, beta, model, x, **kwargs):
    def drawfig1():
        P = model(x)
        #plt.suptitle(kwargs.get('model_name','Model optimization'))
        gs = gridspec.GridSpec(3, 3)
        ### Spectral entropy plot
        plt.subplot(gs[0, 0:2])
        plot_spectral_entropy(A, kwargs.get('beta_range',np.logspace(3,-3,50)),color='r')
        plot_spectral_entropy(model(x), kwargs.get('beta_range',np.logspace(3,-3,50)),color='b')
        #beta=beta,name=kwargs.get('name','Model'))
        ### Optimization variables plot
        plt.subplot(gs[:, 2])
        plt.bar(range(0,len(x)),x)
        for i, v in enumerate(x):
            plt.gca().text(i-0.25, v + 0.05 , "{:.4f}".format(v), color='grey')
        plt.xticks(range(0,len(x)), kwargs.get('labels',range(0,len(x))),rotation=30)
        plt.ylim([-np.abs(np.min(x))*1.2,np.max(x)*1.2])
        #plt.title('Optimization variables')

        # Original matrix plot
        plt.subplot(gs[1, 0])
        sns.heatmap(A)
        plt.xlabel('Node')
        plt.ylabel('Node')
        plt.title('Empirical matrix')
        # Model matrix plot
        plt.subplot(gs[1, 1])
        sns.heatmap(P)
        plt.xlabel('Node')
        plt.ylabel('Node')
        plt.title('Model matrix')
        # Strengths plot
        plt.subplot(gs[2, 0:2])
        plt.scatter([A.sum(axis=0)],[P.sum(axis=0)-P.diagonal()],c='k')
        plt.plot(np.linspace(np.min(A.sum(axis=0)),np.max(A.sum(axis=0)),2 ), np.linspace(np.min(A.sum(axis=0)),np.max(A.sum(axis=0)),2 ), 'r')
        #plt.axis([np.min(A.sum(axis=0)),np.max(A.sum(axis=0)),np.min(A.sum(axis=0)),np.max(A.sum(axis=0))])
        plt.title('Empirical vs model strength')
        plt.xlabel('Strength empirical')
        plt.ylabel('Strenght model')
        plt.tight_layout()
        #plt.show(block=True)
    drawnow(drawfig1)
    print("Beta = %.4f x=%s" % (beta,x))

def plot_spectral_entropy(A, beta_range, **kwargs):
    fobs = lambda beta : compute_vonneumann_entropy(L=graph_laplacian(A), beta=beta)
    Sobs = [fobs(beta)/np.log(len(A)) for beta in beta_range]
    plt.semilogx(1.0/beta_range, Sobs, kwargs.get('color','r'))
    plt.xlabel('$\\beta^{-1}$')
    plt.ylabel('$S/\\log(N)$')
    plt.legend([kwargs.get('name','Model')])
    plt.title(kwargs.get('name','Model'))
    plt.grid(True, color='w')

def print_status(beta, x):
    print('\r','beta=',beta,'x=',x,end="")
