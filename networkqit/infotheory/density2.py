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
from autograd.scipy.misc import logsumexp


def density(L, beta_range):
    """
    Get the von neumann density matrix rho
    over a range of beta values

    :math:`\\frac{e^{-\\beta L}}{\\mathrm{Tr}[e^{-\\beta L}]}`

    Parameters
    ----------
    L: np.array
        A nxn square graph laplacian.

    beta_range: np.array
        the range of beta values
    """
    def compute_rho(L, beta):
        rho = expm(-beta * L)
        return rho / np.trace(rho)

    lambd, Q = np.linalg.eigh(L) # eigenvalues and eigenvectors of Laplacian

    rho = np.asarray([ np.linalg.multi_dot( [Q_obs, np.diag(np.exp(-beta*lambd_obs)), Q_obs.T] ) for beta in beta_range])
    ndim = len(L.shape)
    rho /= np.trace(X,axis1=ndim-2,axis2=ndim-1)
    return rho


def entropy(L, beta_range):
    """
    This function computes Von Neumann spectral entropy over a range of beta
    for a network with Laplacian L
    :math:`S(\\rho) = -\\mathrm{Tr}[\\rho \log \\rho]`

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
    entropy = []
    for b in beta_range:
        lrho = np.exp(-b * lambd)
        Z = lrho.sum()
        entropy.append(np.log(Z) + b * (lambd * lrho).sum() / Z)
    entropy = np.array(entropy)
    entropy[np.isnan(entropy)] = 0
    return np.array(entropy)


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
    def entropy(b):
        lambd_rho = np.exp(-b * lambd)
        Z = lambd_rho.sum()
        return np.log(Z) + b * (lambd * lambd_rho).sum() / Z
    from autograd import grad
    dsdb = grad(lambda y: entropy(y))
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


def batch_relative_entropy(Lobs : np.array, Lmodel : np.array, beta_range : np.array, pade_expm = False):
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
    pade_expm: bool
        whether to compute the matrix rho at each beta using the expm function (default false)
    
    Returns
    --------
    dkl_beta: np.array
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
    nbeta = len(beta_range)
    Emodel_beta = np.zeros([nbeta,])
    Fmodel_beta = np.zeros([nbeta,])
    loglike_beta = np.zeros([nbeta,])
    dkl_beta = np.zeros([nbeta,])
    lambd_model = eigh(Lmodel)[0] # batched eigenvalues
    if not pade_expm:
        lambd_obs, Q_obs = np.linalg.eigh(Lobs) # this trick makes it possible to compute rho once
    # and then play just with beta
    for i, beta in enumerate(beta_range):
        if pade_expm: # use the expm to compute von neumann density
            rho_beta = compute_vonneuman_density(L=Lobs, beta=beta)
        else:
            rho_beta = np.linalg.multi_dot([Q_obs,np.diag(np.exp(-beta*lambd_obs)),Q_obs.T])
            rho_beta /= np.trace(rho_beta)
        Emodel_beta[i] = np.mean(np.sum(np.sum(Lmodel * rho_beta, axis=2), axis=1))
        Fmodel_beta[i] =  - np.mean(logsumexp(-beta * lambd_model, axis=1))/beta
        loglike_beta[i] = beta * (Emodel_beta[i] - Fmodel_beta[i])
        dkl_beta[i] = loglike_beta[i] - Srho[i]
    return dkl_beta

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
            lm = eigvalsh(Lmodel)
            lo = eigvalsh(Lobs)
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
            l = eigvalsh(self.rho + self.sigma)
            self.jensen_shannon = entropy(
                l[l > 0]) - 0.5 * (entropy(eigvalsh(self.rho) + entropy(eigvalsh(self.sigma))))
        self.options = kwargs
