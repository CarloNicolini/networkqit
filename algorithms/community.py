#!/usr/bin/env python
"""
==========
Properties
==========

Community detection related utility functions

"""
#    Copyright (C) 2018 by
#    Carlo Nicolini <carlo.nicolini@iit.it>
#    All rights reserved.
#    BSD license.

import numpy as np


def comm_mat(W, ci):
    """
    Returns multiple community related quantities.

    Input:
        W: a possibly weighted n x n adjacency matrix in the form of a numpy array
        ci: the nodal membership vector, a numpy array or list of integers

    Ouput:
        B: the CxC block matrix containing in its (r,s) element the number of links from community r to community s
        Bnorm: the CxC block matrix containing in its (r,s) element the density of links. It is the same as B but divided by the actual number of possible node pairs nr x ns for off-diagonal terms, (nr x (nr-1))/2 for diagonal eleents
    """
    u = np.unique(ci)
    C = np.zeros([len(u), len(ci)])
    for i in range(C.shape[0]):
        for j in range(0, len(ci)):
            C[i, j] = ci[j] == u[i]
    B = np.dot(np.dot(C, W), C.T)
    K = B.sum(axis=1)
    np.fill_diagonal(B, B.diagonal()/2)
    n = len(W)
    m = np.triu(W, 1).sum()
    p = n * (n - 1) / 2
    commsizes = C.sum(axis=1)
    commpairs = np.dot(np.diag(commsizes), np.diag(commsizes-1)) / 2
    commpairs2 = np.dot(np.dot(np.diag(commsizes), np.ones(
        [len(commsizes), len(commsizes)])), np.diag(commsizes))
    blockpairs = np.multiply(
        commpairs2, (1-np.eye(len(commsizes)))) + commpairs
    Bnorm = B / blockpairs
    return B, Bnorm


def reindex_membership(membership):
    """
    This function has the membership as input and output the membership
    where the communities number are ordered by the number of nodes in that community
    """
    ds = {}
    for u, v in enumerate(membership):
        if v not in ds.keys():
            ds[v] = []
        ds[v].append(u)

    S = dict(
        zip(range(0, len(ds)), sorted(ds.values(), key=len, reverse=True)))

    M = [-1]*len(membership)
    for u, vl in S.items():
        for v in vl:
            M[v] = u
    return M
