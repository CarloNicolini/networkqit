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
    :math:`L=D-A`
    """
    D = np.zeros(A.shape)
    np.fill_diagonal(D,A.sum(axis=0))
    return D - A

def normalized_graph_laplacian(A):
    """
    Get the normalized graph laplacian 
    :math:`\\mathcal{L}=I - D^{-1/2} A D^{-1/2}`
    """
    D = np.zeros(A.shape)
    np.fill_diagonal(D,np.sum(A,1))
    D = np.diag((1.0/np.sqrt(np.diag(D))))
    return np.eye(A.shape[0]) - D*A*D
    