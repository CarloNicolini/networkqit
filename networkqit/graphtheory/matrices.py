#!/usr/bin/env python
#    Copyright (C) 2018 by
#    Carlo Nicolini <carlo.nicolini@iit.it>
#    All rights reserved.
#    BSD license.

"""
Definition of some utility matrices.

The graph Laplacian is computed from an adjacency matrix as $L=D-A$.

The normalized graph Laplacian is computed as $\\mathcal{L}=I - D^{-1/2} A D^{-1/2}$.
"""

import numpy as np

def graph_laplacian(A):
    """
    Get the graph Laplacian from the adjacency matrix
    :math:`\\mathbf{L} = \\mathbf{D} - \\mathbf{A}`
    """
    D = np.zeros(A.shape)
    np.fill_diagonal(D,A.sum(axis=0))
    return D - A

def normalized_graph_laplacian(A):
    """
    Get the normalized graph laplacian 
    :math:`\\mathcal{L}=I - D^{-1/2} A D^{-1/2}`
    """
    invSqrtT = np.diag(1.0/np.sqrt(A.sum(axis=0)))
    return np.eye(A.shape[0]) - invSqrtT@A@invSqrtT

def modularity_matrix(A):
    """
    Returns the modularity matrix
    :math:`\\mathbf{B} = \\mathbf{A} - \\frac{\\mathbf{k} \\mathbf{k}^T}{2m}`
    """
    k = A.sum(axis=0)
    return A - np.outer(k,k)/A.sum()

def signed_laplacian(A):
    """
    Returns the signed Laplacian as defined in https://arxiv.org/pdf/1701.01394.pdf
    :math:`\\mathbf{\\bar{L}} = \\mathbf{\\bar{D}} - \\mathbf{A}
    where the diagonal matrix D is made of the absolute value of the row-sum of A.
    """
    D = np.diag(np.abs(A.sum(axis=0)))
    L = D - A
    return L

def planted_partition_graph(n, b, pin, pout):
    nb = int(n / b)
    A = (np.random.random((n, n)) < pout).astype(float)
    for i in range(0, b):
        T = np.triu((np.random.random((nb, nb)) < pin).astype(float))
        T = T + T.T
        A[i * nb:(i + 1) * nb, i * nb:(i + 1) * nb] = T

    np.fill_diagonal(A, 0)
    A = np.triu(A)
    A = A + A.T
    return A

def sbm(ers, nr):
    N = np.sum(nr) # total number of nodes
    b = len(ers) # number of blocks
    nrns = np.reshape(np.kron(nr, nr), [b, b])
    M = ers / nrns
    A = np.zeros([N, N])
    idx = np.cumsum([0] + nr) # it only works for lists!
    for i in range(0, b):
        ri = np.array(range(idx[i], idx[i + 1] + 1))
        for j in range(0, b):
            rj = np.array(range(idx[j], idx[j + 1] + 1))
            R = np.random.random([len(ri) - 1, len(rj) - 1])
            A[ri.min():ri.max(), rj.min():rj.max()] = R < ers[i, j]/nrns[i, j] # like a bernoulli rv with average ers[i,j]/nrns
    A = np.triu(A, 1)
    A += A.T
    return A

def dcsbm(ers, nr, ki):
    N = np.sum(nr) # total number of nodes
    L = np.sum(ki)
    b = len(ers) # number of blocks
    nrns = np.reshape(np.kron(nr, nr), [b, b])
    M = ers / nrns
    A = np.zeros([N, N])
    idx = np.cumsum([0] + nr) # it only works for lists!
    for i in range(0, b):
        ri = np.array(range(idx[i], idx[i + 1] + 1))
        kr = ki[ri.min():ri.max()]
        print(kr.sum())
        for j in range(0, b):
            rj = np.array(range(idx[j], idx[j + 1] + 1))
            R = np.random.random([len(ri) - 1, len(rj) - 1])
            ks = ki[rj.min():rj.max()]
            A[ri.min():ri.max(), rj.min():rj.max()] = kr.sum()*ks.sum() > ers[i,j]
    A = np.triu(A, 1)
    A += A.T
    return A

def sbm_p(sigma2rs, nr):
    b = len(ers) # number of blocks
    nrns = np.reshape(np.kron(nr, nr), [b, b])
    return hierarchical_random_graph2(sigma2rs * nrns, nr)

def wsbm(ers,nr,dist):
    # Weighted stochastic block model, where dist is a random variable sampling function taking one parameter
    N = np.sum(nr) # total number of nodes
    b = len(ers) # number of blocks
    nrns = np.reshape(np.kron(nr, nr), [b, b])
    #np.fill_diagonal(nrns,nrns.diagonal()/2)
    #print(nrns)
    M = ers / nrns
    A = np.zeros([N, N])
    idx = np.cumsum([0] + nr) # it only works for lists!
    for i in range(0, b):
        ri = np.array(range(idx[i], idx[i + 1] + 1))
        for j in range(0, b):
            rj = np.array(range(idx[j], idx[j + 1] + 1))
            R = np.random.random([len(ri) - 1, len(rj) - 1])
            V = dist(M[i,j],size=R.shape[0]*R.shape[1])
            A[ri.min():ri.max(), rj.min():rj.max()] = np.reshape(V,R.shape)
    A = np.triu(A, 1)
    A += A.T
    return A