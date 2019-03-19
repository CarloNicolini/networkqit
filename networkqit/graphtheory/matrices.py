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

"""
Definition of some utility matrices.

The graph Laplacian is computed from an adjacency matrix as $L=D-A$.

The normalized graph Laplacian is computed as $\\mathcal{L}=I - D^{-1/2} A D^{-1/2}$.
"""

import autograd.numpy as np
from autograd.scipy.special import expit
EPS = np.finfo(float).eps

def graph_laplacian(A):
    """
    Get the graph Laplacian from the adjacency matrix
    :math:`\\mathbf{L} = \\mathbf{D} - \\mathbf{A}`
    If a batched adjacency matrix of shape [batch_size, N, N] is
    given, the batched laplacian is returned.
    """
    if len(A.shape)==3:
        N = A.shape[-1] # last dimension is number of nodes
        D = np.eye(N) * np.transpose(np.zeros([1, 1, N]) + np.einsum('ijk->ik', A), [1, 0, 2])
        return D - A
    else:
        return np.diag(A.sum(axis=0)) - A


def directed_graph_laplacian(A, walk_type='random_walk', alpha=0.95):
    r"""Return the directed Laplacian matrix of A.

    The graph directed Laplacian is the matrix

    .. math::

        L = I - (\Phi^{1/2} P \Phi^{-1/2} + \Phi^{-1/2} P^T \Phi^{1/2} ) / 2

    where `I` is the identity matrix, `P` is the transition matrix of the
    graph, and `\Phi` a matrix with the Perron vector of `P` in the diagonal and
    zeros elsewhere.

    Depending on the value of walk_type, `P` can be the transition matrix
    induced by a random walk, a lazy random walk, or a random walk with
    teleportation (PageRank).

    Parameters
    ----------
    A : numpy.array
       The adjacency matrix

    walk_type : string or None, optional (default=None)
       If None, `P` is selected depending on the properties of the
       graph. Otherwise is one of 'random', 'lazy', or 'pagerank'

    alpha : real
       (1 - alpha) is the teleportation probability used with pagerank

    Returns
    -------
    L : NumPy array
      Directed Laplacian of A.

    Notes
    -----
    Only implemented for DiGraphs

    References
    ----------
    .. [1] Fan Chung (2005).
       Laplacians and the Cheeger inequality for directed graphs.
       Annals of Combinatorics, 9(1), 2005
    """
    raise NotImplementedError('Still not implemented, take a look at networkx nx.directed_graph_laplacian')

def normalized_graph_laplacian(A):
    """
    Get the normalized graph laplacian 
    :math:`\\mathcal{L}=I - D^{-1/2} A D^{-1/2}`
    If a batched adjacency matrix of shape [batch_size, N, N] is
    given, the batched laplacian is returned.
    """
    if len(A.shape)==3:
        N = A.shape[-1]
        invSqrtD = np.eye(N) * np.transpose(np.zeros([1, 1, N]) + 1/np.sqrt(np.einsum('ijk->ik', A)), [1, 0, 2])
        return  np.eye(N) - invSqrtD @ A @ invSqrtD
    else:
        invSqrtT = np.diag(1.0 / np.sqrt(A.sum(axis=0)))
        return np.eye(A.shape[0]) - invSqrtT @ A @ invSqrtT


def modularity_matrix(A):
    """
    Returns the modularity matrix
    :math:`\\mathbf{B} = \\mathbf{A} - \\frac{\\mathbf{k} \\mathbf{k}^T}{2m}`
    """
    if len(A.shape)==3:
        N = A.shape[-1]
        b  = A.shape[0]
        k = np.einsum('ijk->ik', A)
        kikj = np.einsum('ij,ik->ijk', k, k)
        m = np.sum(np.sum(A,axis=1), axis=1, keepdims=True)
        B = A - (kikj/np.broadcast_to(np.expand_dims(m,axis=2),A.shape))    # batched kikj/2m
        return  B
    else:
        k = A.sum(axis=0)
        return A - np.outer(k, k) / k.sum()

def signed_laplacian(A):
    """
    Returns the signed Laplacian as defined in https://arxiv.org/pdf/1701.01394.pdf
    :math:`\\mathbf{\\bar{L}} = \\mathbf{\\bar{D}} - \\mathbf{A}
    where the diagonal matrix D is made of the absolute value of the row-sum of A.
    """
    if len(A.shape)==3:
        N = A.shape[-1] # last dimension is number of nodes
        D = np.eye(N) * np.transpose(np.zeros([1, 1, N]) + np.einsum('ijk->ik', A), [1, 0, 2])
        return np.abs(D) - A
    else:
        return np.diag(np.abs(A.sum(axis=0))) - A

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
    N = np.sum(nr)  # total number of nodes
    b = len(ers)  # number of blocks
    nrns = np.reshape(np.kron(nr, nr), [b, b])
    M = ers / nrns
    A = np.zeros([N, N])
    idx = np.cumsum([0] + nr)  # it only works for lists!
    for i in range(0, b):
        ri = np.array(range(idx[i], idx[i + 1] + 1))
        for j in range(0, b):
            rj = np.array(range(idx[j], idx[j + 1] + 1))
            R = np.random.random([len(ri) - 1, len(rj) - 1])
            A[ri.min():ri.max(), rj.min():rj.max()] = R < ers[i, j] / nrns[
                i, j]  # like a bernoulli rv with average ers[i,j]/nrns
    A = np.triu(A, 1)
    A += A.T
    return A


# not sure if this function is correct!!!
def dcsbm(ers, nr):
    # import warning
    # warning.warning('This function is not correct')
    N = np.sum(nr)  # total number of nodes

    b = len(ers)  # number of blocks
    nrns = np.reshape(np.kron(nr, nr), [b, b])
    M = ers / nrns
    A = np.zeros([N, N])
    idx = np.cumsum([0] + nr)  # it only works for lists!
    ki = ers.sum(axis=0)
    K = np.outer(ki, ki) / (np.sum(ki) ** 2)
    for r in range(0, b):
        idr = np.array(range(idx[r], idx[r + 1] + 1))
        # kr = ki[idr.min():idr.max()]
        for s in range(0, b):
            ids = np.array(range(idx[s], idx[s + 1] + 1))
            # ks = ki[ids.min():ids.max()]
            V = np.random.poisson(M[r, s], size=(len(idr) - 1) * (len(ids) - 1))
            A[idr.min():idr.max(), ids.min():ids.max()] = np.reshape(V, [(len(idr) - 1), (len(ids) - 1)])
    A = np.triu(A, 1)
    A += A.T
    return A


def sbm_p(sigma2rs, nr):
    b = len(ers)  # number of blocks
    nrns = np.reshape(np.kron(nr, nr), [b, b])
    return hierarchical_random_graph2(sigma2rs * nrns, nr)


def wsbm(ers, nr, dist):
    # Weighted stochastic block model, where dist is a random variable sampling function taking one parameter
    N = np.sum(nr)  # total number of nodes
    b = len(ers)  # number of blocks
    nrns = np.reshape(np.kron(nr, nr), [b, b])
    # np.fill_diagonal(nrns,nrns.diagonal()/2)
    # print(nrns)
    M = ers / nrns
    A = np.zeros([N, N])
    idx = np.cumsum([0] + nr)  # it only works for lists!
    for i in range(0, b):
        ri = np.array(range(idx[i], idx[i + 1] + 1))
        for j in range(0, b):
            rj = np.array(range(idx[j], idx[j + 1] + 1))
            R = np.random.random([len(ri) - 1, len(rj) - 1])
            V = dist(M[i, j], size=R.shape[0] * R.shape[1])
            A[ri.min():ri.max(), rj.min():rj.max()] = np.reshape(V, R.shape)
    A = np.triu(A, 1)
    A += A.T
    return A

def batched_symmetric_random(batch_size, N):
    rij = np.random.random([batch_size, N, N])
    rij = np.triu(rij, 1)
    rij += np.transpose(rij,[0,2,1]) # transpose last axis
    rij[rij<EPS]=EPS # to avoid zeros
    return rij

def batched_gumbel(batch_size, N, eps=1E-20):
  """Sample from Gumbel(0, 1)"""
  uij = batched_symmetric_random(batch_size,N)
  return -np.log(-np.log(uij))

def gumbel_softmax_sample(probits, temperature): 
    """
    Draw a sample from the Gumbel-Softmax distribution

    Example:
    p = 0.1
    K = 8
    probits = np.reshape(np.tile(np.array([p**(k)*(1-p) for k in range(
    K)]),[batch_size,]),[batch_size,K])

    Args:
    probits: [batch_size, n_class] unnormalized probabilities
    temperature: non-negative scalar
    """
    def softmax(x):
      return np.exp(x)/np.sum(np.exp(x))
    y = np.reshape(np.repeat(probits,[8,]),[3,8,8]) + batched_gumbel(probits.shape[0],probits.shape[1])
    return softmax( y / temperature)

def gumbel_softmax(probits, temperature, hard=False):
    """
    Sample from the Gumbel-Softmax distribution and optionally discretize.

    Args:
    probits: [batch_size, n_class] unnormalized probabilities
    temperature: non-negative scalar
    Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
        y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y

def multiexpit(x, slope=50):
    y = np.asarray([ expit(slope*(x-i)) for i in range(int(np.max(x))) ])
    return np.sum(y,axis=0)

def multiexpit2(x, slope=50):
    i = np.arange(int(min(x)//1),int(max(x)//1)+1)
    X, I = np.meshgrid(x,i)
    return np.sum(expit(slope*(X-I)),axis=0)+min(x)//1-1

def ilessjsum(Q):
    # This function is equivalent to np.triu(Q,1).sum() but 4 times faster
    return (Q.sum()- np.trace(Q))/2 