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
Method and functions based on the information theory of networks
"""
import autograd.numpy as np
from scipy.linalg import expm, logm, eigvalsh
from scipy.stats import entropy
from networkqit.graphtheory.matrices import graph_laplacian

from autograd.numpy.linalg import eigh
#from autograd.scipy.misc import logsumexp

"""
Definitions to work within the spectral entropies framework.
All these functions work for batched matrices, i.e. square matrices represented 
as numpy arrays where the first dimension is the batch size.
If the first dimension is missing then the functions work as on normal square matrices
automatically, otherwise a number of computational tricks are adopted.
"""

__all__ = ['density', 'entropy', 
           'entropy_beta_deriv',
           'find_beta_logc',
           'relative_entropy_one_component',
           'relative_entropy']

def density(L : np.array, beta_range : np.array):
    """
    Get the von neumann density matrix rho
    over a range of beta values, for single or batched laplacians

    :math:`\\frac{e^{-\\beta L}}{\\mathrm{Tr}[e^{-\\beta L}]}`

    Parameters
    ----------
    L: np.array
        A nxn square graph laplacian, or a batch [batch_size,n,n] of Laplacians

    beta: float, beta_range
        a single value of beta if float is passed, otherwise an array of different betas
        the numbers of values in beta_range in denoted as b

    Returns
    -------
    rho (np.array): the shape of the output rho array depends on the shape of the inputs
                    it is always [b,batch_size,n,n]. If both a square Laplacian and a float
                    beta are passed, then the output is simply [n,n].
                    Final dimensions are squeezed
    """
    n,m = L.shape[-1],L.shape[-2]
    if n != m:
        raise RuntimeError('Must input square or batched square laplacian [batch_size,n,n]')
    _beta_range = beta_range
    if isinstance(beta_range, float) or isinstance(beta_range, int):
        _beta_range = np.asarray([beta_range])
    lambd, Q = np.linalg.eigh(L) # eigenvalues and eigenvectors of (batched) Laplacian
    ndim = len(L.shape)

    if ndim==3:
        batch_size = L.shape[0]
        def batched_eigvals_to_batched_diag(beta,v):
            return np.einsum('lij,li->lij', (np.eye(n) + np.zeros([batch_size,n,n])), np.exp(-beta*v), optimize=True)
        rho = np.asarray([np.einsum('mij,mjk,mkl->mil', Q, batched_eigvals_to_batched_diag(beta, lambd) , np.transpose(Q,[0,2,1]), optimize=True)  for beta in _beta_range])
    elif ndim==2:
        rho = np.asarray([np.linalg.multi_dot([Q, np.diag(np.exp(-beta*lambd)) , Q.T]) for beta in _beta_range])[:,None,:,:]
    rho /= np.trace(rho,axis1=2,axis2=3)[:, :, None, None]
    return np.squeeze(rho)


def entropy(L : np.array, beta_range: np.array):
    """
    This function computes Von Neumann spectral entropy over a range of beta values
    for a (batched) network with Laplacian L
    :math:`S(\\rho) = -\\mathrm{Tr}[\\rho \\log \\rho]`

    Parameters
    ----------
    L: np.array
        The (batched) n x n graph laplacian. If batched, the input dimension is [batch_size,n,n]
    
    beta_range: (iterable) list or numpy.array
        The range of beta

    Returns
    -------
    np.array
        The unnormalized Von Neumann entropy over the beta values and over all batches.
        Final dimension is [b,batch_size]. If 2D input, final dimension is [b]
        where b is the number of elements in the array beta_range

    Raises
    ------
    None
    """
    ndim = len(L.shape)
    lambd, Q = np.linalg.eigh(L) # eigenvalues and eigenvectors of (batched) Laplacian
    if ndim==3:
        batch_size = L.shape[0]
        entropy = np.zeros([batch_size, len(beta_range)])
        lrho = np.exp(-np.multiply.fun.outer(beta_range,lambd))
        Z = np.sum(lrho, axis=2)
        entropy = np.log(Z) + beta_range[:,None]*np.sum(lambd*lrho,axis=2)/Z
    elif ndim==2:
        entropy = np.zeros_like(beta_range)
        for i,b in enumerate(beta_range):
            lrho = np.exp(-b * lambd)
            Z = lrho.sum()
            entropy[i] = np.log(np.abs(Z)) + b * (lambd * lrho).sum() / Z
    else:
        raise RuntimeError('Must provide a 2D or 3D array (as batched 2D arrays)')
    entropy[np.isnan(entropy)] = 0
    return entropy


def entropy_beta_deriv(L, beta_range):
    """
    Computes the derivative of Von Neumann entropy with
    respect to the beta parameter
    :math:`\\frac{\\partial S(\\rho)}{\\partial \\beta}`
    
    Note: this function uses autograd method to get the derivatives
    
    Parameters
    ----------
    L: np.array
        The graph laplacian
    beta_range: (iterable) list or numpy.array
        The range of beta

    Returns
    -------
    np.array
        The unnormalized Von Neumann entropy over the beta values

    Raises
    ------
    None
    """

    lambd = eigvalsh(L)
    def _entropy(b):
        lambd_rho = np.exp(-b * lambd)
        Z = lambd_rho.sum()
        return np.log(Z) + b * (lambd * lambd_rho).sum() / Z
    from autograd import grad
    dsdb = grad(lambda y: _entropy(y))
    return np.array([dsdb(x) for x in beta_range])


def find_beta_logc(L, c, a=1E-5, b=1E5):
    """
    Computes the exact beta such that the Von Neumann entropy is S=log(c) where c is a float.
    This method uses the bisection method to find the solution.

    Parameters
    ----------
    L: np.array
        The graph laplacian
    c: float
        a number, if integer, it is meant to represent the number of communities or blocks
    a: float (default 1E-5)
        lower bound on the beta search
    b: float (default 1E5)
        upper bound on the beta search

    Returns
    -------
    float
        the beta^* such that S(rho(beta^*)) = log(c)

    Raises
    ------
    None
    """

    from scipy.optimize import bisect
    lambd = eigvalsh(L)

    def s(b, l):
        lrho = np.exp(-b * l)
        Z = lrho.sum()
        return np.log(Z) + b * (l * lrho).sum() / Z
    return bisect(lambda x: s(x, lambd) - np.log(c), a, b)

def relative_entropy_one_component(Lobs : np.array, Lmodel : np.array, beta_range : np.array):
    """
    This function makes it possible to efficiently compute the average relative entropy E[S(rho || sigma)]
    over a batch of model laplacians, and given a range of beta.
    This function efficiently computes exploiting some numerical tricks

    Parameters
    ----------
    Lobs: np.array
        a square laplacian nxn matrix of the empirical network
    Lmodel: np.array
        a square (or batched) laplacian matrix of model network.
        if a [b,n,n] array is given, the first dimension is considered the batch size

    beta_range: np.array or float
        the range of beta values over which to compute the relative entropy
    
    Returns
    --------
    dkl: np.array
            the array of expected relative entropy between empirical and model networks,
            evalued at each beta

    Raises
    ------
    None
    """
    if len(Lobs.shape) != 2:
        raise RuntimeError('Must provide a square, non batched observed laplacian')
    
    if len(Lmodel.shape)==2:
        Lmodel = np.expand_dims(Lmodel,0) # add the zero dimension to have batched 

    Srho = entropy(L=Lobs, beta_range=beta_range)
    batch_size = Lmodel.shape[0]
    nbeta = len(beta_range)
    Em = np.zeros([nbeta,])
    beta_Fm = np.zeros([nbeta,])
    loglike = np.zeros([nbeta,])
    dkl = np.zeros([nbeta,])
    lambd_model = np.linalg.eigh(Lmodel)[0] # batched eigenvalues

    pade_expm = False
    if not pade_expm:
        lambd_obs, Q_obs = np.linalg.eigh(Lobs) # this trick makes it possible to compute rho once
    
    # keep only same eigenvalues set as observed
    same_components = False
    if same_components:
        idx_obs = np.argwhere(lambd_obs<1E-12)
        num_obs_connected_components = len(idx_obs)
        idx = np.argwhere( (lambd_model<1E-12).astype(int).sum(axis=1) == num_obs_connected_components ).flatten()
        Lmodel = Lmodel[idx,:]
        lambd_model = lambd_model[idx,:]
    
    # compute the quantities at every beta
    for i, beta in enumerate(beta_range):
        if pade_expm: # use the expm to compute von neumann density
            rho = density(L=Lobs, beta_range=[beta])
        else: # no need of batched operations here, one only observation!
            rho = np.linalg.multi_dot([Q_obs,np.diag(np.exp(-beta*lambd_obs)),Q_obs.T])
            rho /= np.trace(rho)
        Em[i] = np.mean(np.sum(np.sum(Lmodel * rho, axis=2), axis=1)) # average energy
        beta_Fm[i] =  - np.mean(logsumexp(-beta * lambd_model, axis=1)) # quenched log partition funtion
        loglike[i] = beta * Em[i] - beta_Fm[i]
        dkl[i] = loglike[i] - Srho[i]
    return dkl, loglike, Em, beta_Fm

def relative_entropy(Lobs : np.array, Lmodel : np.array, beta_range : np.array):
    """
    This function makes it possible to efficiently compute the average relative entropy E[S(rho || sigma)]
    over a batch of model Laplacians, and given a range of beta.
    This function efficiently computes S(rho || sigma) with some numerical tricks to avoid recalculation
    of the rho matrix over a range of beta values.

    Parameters
    ----------
    Lobs: np.array
        a square laplacian nxn matrix of the empirical network
    Lmodel: np.array
        a square (or batched) laplacian matrix of model network.
        if a [b,n,n] array is given, the first dimension is considered the batch size

    beta_range: np.array or float
        the range of beta values over which to compute the relative entropy
    
    Returns
    --------
    dkl: np.array
            the array of expected relative entropy between empirical and model networks,
            evalued at each beta

    Raises
    ------
    None
    """
    import bct
    Aobs = Lobs.copy()
    np.fill_diagonal(Aobs,0)
    Aobs = -Aobs
    idx, comps = bct.get_components(Aobs)
    ncomps = len(comps)
    batches = range(Lmodel.shape[0])
    avg_dkl = np.zeros_like(beta_range)

    if ncomps>1: # disconnected graph, need to average over this function
        for c in np.unique(idx):
            avg_dkl += relative_entropy_one_component(Lobs = Lobs[np.ix_(idx==c,idx==c)],
                                                      Lmodel=Lmodel[np.ix_(batches,idx==c,idx==c)],
                                                      beta_range=beta_range)[0]/beta_range
        return avg_dkl/ncomps
    else:
        return relative_entropy_one_component(Lobs, Lmodel, beta_range)


class SpectralDivergence(object):
    """
    This class defines the object SpectralDivergence that permits to avoid repetitive computations of the observed density rho.
    args:
        Lobs (numpy.array): the observed Laplacian matrix.
        Lmodel (numpy.array): the observed Laplacian matrix.
        beta (float): the beta hyperparameter.

    kwargs:
        rho (numpy.array): you can avoid computation of rho, if in some optimization method this is kept constant.
        fast_mode (bool): if fast_mode is set to True (default) the average model and observed energy are compuuted
                          as sum of elementwise products, otherwise trace of matrix product is used. 
                          Moreover computation of eigenvalues instead of tracing of matrix exponential is used.
    """

    def __init__(self, Lobs: np.array, Lmodel: np.array, beta: float, **kwargs):
        self.Lmodel = Lmodel
        self.Lobs = Lobs
        self.fast_mode = kwargs.get('fast_mode', True)
        if 'rho' in kwargs.keys():
            self.rho = kwargs['rho']
        else:
            self.rho = compute_vonneuman_density(Lobs, beta)

        # Average energy of the observation and model
        if self.fast_mode:
            # use the property of hadamard product
            # Trace of matrix product can be simplified by using sum of hadamard product
            # if matrices are symmetric
            # Tr(A B) =  Sum(A_ij B_ij)
            self.Em = (Lmodel * self.rho).sum()
            self.Eo = (Lobs * self.rho).sum()
        else:  # otherwise use correct version, slower for large graphs
            self.Em = np.trace(np.dot(Lmodel, self.rho))
            self.Eo = np.trace(np.dot(Lobs, self.rho))
        self.deltaE = self.Em - self.Eo
        # Computation of partition functions
        if self.fast_mode:  # prefer faster implementation based on eigenvalues
            lm = np.linalg.eigvalsh(Lmodel)
            lo = np.linalg.eigvalsh(Lobs)
            self.Zm = np.exp(-beta * lm).sum()
            self.Zo = np.exp(-beta * lo).sum()
        else:  # otherwise compute with matrix exponentials
            self.Zm = np.trace(expm(-beta * Lmodel))
            self.Zo = np.trace(expm(-beta * Lobs))

        # Computation of free energies from partition functions
        self.Fm = -np.log(self.Zm) / beta
        self.Fo = -np.log(self.Zo) / beta
        self.deltaF = self.Fm - self.Fo
        # Loglikelihood betweeen rho (obs) and sigma (model)
        self.loglike = beta * (-self.Fm + self.Em)

        # Entropy of observation (rho)
        self.entropy = beta * (-self.Fo + self.Eo)

        # use abs ONLY for numerical issues, as DKL must be positive
        # self.rel_entropy = np.trace(np.dot(compute_vonneuman_density(Lobs,beta),(logm(compute_vonneuman_density(Lobs,beta)) - logm(compute_vonneuman_density(Lmodel,beta)))))
        self.rel_entropy = np.abs(self.loglike - self.entropy)

        if kwargs.get('compute_js', False):
            l = np.linalg.eigvalsh(self.rho + self.sigma)
            self.jensen_shannon = entropy(
                l[l > 0]) - 0.5 * (entropy(eigvalsh(self.rho) + entropy(eigvalsh(self.sigma))))
        self.options = kwargs
