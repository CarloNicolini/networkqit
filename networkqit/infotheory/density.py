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


def compute_vonneuman_density(L, beta):
    """ Get the von neumann density matrix :math:`\\frac{e^{-\\beta L}}{\\mathrm{Tr}[e^{-\\beta L}]}` """
    rho = expm(-beta * L)
    return rho / np.trace(rho)


def compute_vonneumann_entropy(**kwargs):
    """
    Get the von neumann entropy of the density matrix :math:`S(\\rho) = -\\mathrm{Tr}[\\rho \log \\rho]`
    """

    def entropy(L, beta):
        lambd = eigvalsh(L)
        # l = l[l > 0] # introduces an error, entropy doesn't tend to 0 in beta=inf
        lrho = np.exp(-beta * lambd)
        Z = lrho.sum()
        return np.log(Z) + beta * (lambd * lrho).sum() / Z

    if 'density' in kwargs.keys():
        lambd = eigvalsh(kwargs['density'])
        return entropy(lambd[lambd > 0])

    elif 'A' in kwargs.keys() and 'beta' in kwargs.keys():
        A = kwargs['A']
        L = graph_laplacian(A)
        return entropy(L, kwargs['beta'])

    elif 'L' in kwargs.keys() and 'beta' in kwargs.keys():
        return entropy(kwargs['L'], kwargs['beta'])


def batch_compute_vonneumann_entropy(L, beta_range):
    """
    This function computes spectral entropy over a range of beta, given that L remains the same
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


def compute_vonneumann_entropy_beta_deriv(**kwargs):
    """
    Get the derivative of entropy with respect to inverse temperature :math:`\\frac{\\partial S(\\rho)}{\\partial \\beta}`
    """
    if 'A' in kwargs.keys() and 'beta' in kwargs.keys():
        A = kwargs['A']
        L = graph_laplacian(A)
        rho = compute_vonneuman_density(L, kwargs['beta'])
        logmrho = logm(rho)
        # Tr [ Lρ log ρ ] − Tr [ ρ log ρ ] Tr [ Lρ ]
        return np.trace(L @ rho @ logmrho) - np.trace(rho @ logmrho) * np.trace(L @ rho)

    elif 'L' in kwargs.keys() and 'beta' in kwargs.keys():
        L = kwargs['L']
        rho = compute_vonneuman_density(L, kwargs['beta'])
        return np.trace(L @ rho @ logm(rho)) - np.trace(rho @ logm(rho)) * np.trace(L @ rho)


def batch_compute_vonneumann_entropy_beta_deriv(L, beta_range):
    """
    Compute the Von Neumann entropy of a graph laplacian over the range of beta parameters

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

    from numdifftools import Derivative
    dsdb = Derivative(lambda y: entropy(y), n=1)
    return np.array([dsdb(x) for x in beta_range])


def find_beta_logc(L, c, a=1E-5, b=1E5):
    """
    Computes the exact beta such that the Von Neumann entropy is S=log(c) where c is a float.
    This method uses the bisection method to solve find the solution.

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


class VonNeumannDensity(object):
    def __init__(self, A, L, beta, **kwargs):
        self.L = L
        self.A = A
        self.beta = beta
        self.density = compute_vonneuman_density(self.L, beta)


class SpectralDivergence(object):
    """
    This class defines the object SpectralDivergence that permits to avoid repetitive computations of the observed density rho.
    args:
        Lobs (numpy.array): the observed Laplacian matrix.
        Lmodel (numpy.array): the observed Laplacian matrix.
        beta (float): the beta hyperparameter.

    kwargs:
        rho (numpy.array): you can avoid computation of rho, if in some optimization method this is kept constant.
        fast_mode (bool): if fast_mode is set to True (default) the average model and observed energy are coompuuted as sum of elementwise products, otherwise trace of matrix product is used. Moreover computation of eigenvalues instead of tracing of matrix exponential is used.
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
