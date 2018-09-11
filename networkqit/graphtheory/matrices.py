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

def hierarchical_random_graph(ers, nr):
    N = np.sum(nr) # total number of nodes
    b = len(ers) # number of blocks
    nrns = np.reshape(np.kron(nr, nr), [b, b])
    M = ers / nrns
    A = np.zeros([N, N])
    idx = np.cumsum([0] + nr)
    for i in range(0, b):
        ri = np.array(range(idx[i], idx[i + 1] + 1))
        for j in range(0, b):
            rj = np.array(range(idx[j], idx[j + 1] + 1))
            R = np.random.random([len(ri) - 1, len(rj) - 1])
            A[ri.min():ri.max(), rj.min():rj.max()] = (nrns[i, j] * R) < ers[i, j]
    A = np.triu(A, 1)
    A += A.T
    return A

def hierarchical_random_graph_p(sigma2rs, nr):
    b = len(ers) # number of blocks
    nrns = np.reshape(np.kron(nr, nr), [b, b])
    return hierarchical_random_graph2(sigma2rs * nrns, nr)