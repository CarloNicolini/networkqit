#!/usr/bin/env python
"""
Community detection related utility functions.
"""
#    Copyright (C) 2018 by
#    Carlo Nicolini <carlo.nicolini@iit.it>
#    All rights reserved.
#    BSD license.

import numpy as np


def comm_mat(adj, memb):
    """
    Returns multiple community related quantities.

    Input:
        adj: a possibly weighted n x n adjacency matrix in the form of a numpy array
        ci: the nodal membership vector, a numpy array or list of integers

    Ouput:
        B: the CxC block matrix containing in its (r,s) element the number of links from community r to community s
        Bnorm: the CxC block matrix containing in its (r,s) element the density of links. It is the same as B but divided by the actual number of possible node pairs nr x ns for off-diagonal terms, (nr x (nr-1))/2 for diagonal eleents
    """
    u = np.unique(memb)
    C = np.zeros([len(u), len(memb)])
    for i in range(C.shape[0]):
        for j in range(0, len(memb)):
            C[i, j] = memb[j] == u[i]
    B = np.dot(np.dot(C, adj), C.T)
    # K = B.sum(axis=1)
    np.fill_diagonal(B, B.diagonal()/2)
    # n = len(adj)
    # m = np.triu(W, 1).sum()
    # p = n * (n - 1) / 2
    commsizes = C.sum(axis=1)
    commpairs = np.dot(np.diag(commsizes), np.diag(commsizes-1)) / 2
    commpairs2 = np.dot(np.dot(np.diag(commsizes), np.ones([len(commsizes), len(commsizes)])), np.diag(commsizes))
    blockpairs = np.multiply(commpairs2, (1-np.eye(len(commsizes)))) + commpairs
    Bnorm = B / blockpairs
    return B, Bnorm


def reindex_membership(memb, key='community_size', compress_singletons=False, adj=None):
    """
    This function has the membership as input and output the membership
    where the communities number are ordered by the number of nodes in that community

    Input:
        memb (list): a list of nodes membership
        key (str): by default it sorts the community indices by decreasing number of nodes in each community.
                   other arguments are 'community_weight' (sum of all internal community weights)
                   or 'community_normalized_weight' (internal weight normalized by internal pairs).
                   Important
        compress_singletons (bool): if set to True assign all singleton communities into a single community.
        adj (np.array): the adjacency matrix where to read the
    """
    ds = {}
    for u, v in enumerate(memb):
        if v not in ds.keys():
            ds[v] = []
        ds[v].append(u)

    def community_size(x):
        return len(x)

    def community_weight(x):
        return adj[np.ix_(x, x)].sum()

    def community_normalized_weight(x):
        return adj[np.ix_(x, x)].sum() / (len(x) * len(x) - 1)

    mykey = None
    if key is not 'community_size' and adj is None:
            raise AssertionError('Must input adjacency matrix too')
    else:
        if key is 'community_size':
            mykey = community_size
        elif key is 'community_weight':
            mykey = community_weight
        elif key is 'community_normalized_weight':
            mykey = community_normalized_weight

    S = dict(zip(range(0, len(ds)), sorted(ds.values(), key=mykey, reverse=True)))

    memo_reindex = [-1] * len(memb)
    for u, vl in S.items():
        for v in vl:
            memo_reindex[v] = u
    if compress_singletons:
        memo_reindex = reassign_singletons(memo_reindex)
    return memo_reindex


def reassign_singletons(memb):
    """
    Set all the singleton communities into the same community.
    If membership has C communities, S of which are singletons, then
    this function sets the singleton communities with the label C + 1
    Input:

    """
    memb2 = np.array(reindex_membership(memb))
    max_memb = np.max(memb2) + 1
    memb3 = memb2.copy()

    for iu in np.unique(memb):
        ix = np.where(memb == iu)[0]
        if len(ix) > 1:
            max_memb = iu
    max_memb += 1
    for iu in np.unique(memb):
        ix = np.where(memb == iu)[0]
        if len(ix) == 1:
            memb3[ix] = max_memb
    memb3 = reindex_membership(memb3)
    return memb3