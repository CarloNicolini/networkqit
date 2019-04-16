#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# networkqit -- a python module for manipulations of spectral entropies framework
#
# Copyright (C) 2017-2018 Carlo Nicolini <carlo.nicolini@iit.it>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import autograd.numpy as np

__all__ = ['comm_mat',
            'comm_assortativity',
            'reindex_membership',
            'reassign_singletons'
            ]

def comm_mat(adj, memb):
    """
    Returns multiple community related quantities.

    Input
    -----
        adj : :class:`~np.array`: a possibly weighted n x n adjacency matrix in the form of a N x N numpy array
        memb: :class:`~np.array, list`: the nodal membership vector, a numpy array or list of integers.

    Ouput
    -----
        B: :class`~np.array`: the CxC block matrix containing in its (r,s) element the number of links from community
        `r` to community `s`.
        Bnorm: :class`~np.array`: the CxC block matrix containing in its (r,s) element the density of links.
        It is the same as B but divided by the actual number of possible node pairs nr x ns for off-diagonal terms,
        `(nr x (nr-1))/2` for diagonal elements.


    References
    ----------
    .. [holland-stochastic-1983] Paul W. Holland, Kathryn Blackmond Laskey,
       Samuel Leinhardt, "Stochastic blockmodels: First steps",
       Carnegie-Mellon University, Pittsburgh, PA 15213, U.S.A.,
       :doi:`10.1016/0378-8733(83)90021-7`.
    """
    u = np.unique(memb)
    C = np.zeros([len(u), len(memb)])
    for i in range(C.shape[0]):
        for j in range(0, len(memb)):
            C[i, j] = memb[j] == u[i]
    B = np.dot(np.dot(C, adj), C.T)
    K = B.sum(axis=1)
    #print(K**2)
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

def comm_assortativity(A, memb):
    """
    This function computes the modular group assortativity and the newman modularity as from Eq.18-19
    of the Peixoto paper "Nonparametric weighted stochastic block models"
    https://arxiv.org/pdf/1708.01432.pdf

    Args:
        A (np.array): the (weighted) adjacency matrix
        memb (list): the node block membership 
    Output:
        qr: the group modular assortativity
        Q: the Newman modularity


    References
    ----------
    .. [peixoto2017-weighted] Tiago Peixoto, "Non parametric weighted stochastic block model",
       https://arxiv.org/pdf/1708.01432.pdf
       :url:`https://arxiv.org/pdf/1708.01432.pdf`.
    """
    B = len(np.unique(memb))
    E = np.triu(A,1).sum()
    ers,ersnorm = comm_mat(A, memb)
    np.fill_diagonal(ers,2*np.diagonal(ers))
    err = np.diagonal(ers)
    er2 = np.sum(ers,axis=0)**2
    qr = B/(2.0*E)*(err-(er2/(2.0*E)))
    Q = np.mean(qr) # newman modularity
    return qr, Q


def reindex_membership(memb, key='community_size', compress_singletons=False, adj=None):
    """
    This function has the membership as input and output the membership
    where the communities number are ordered by the number of nodes in that community

    Args:
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
    Args:
        memb (np.array): the input membership vector
    Output:
        an array where all the nodes with a single community are merged 
        into one.
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