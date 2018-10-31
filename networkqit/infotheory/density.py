#    Copyright (C) 2018 by
#    Carlo Nicolini <carlo.nicolini@iit.it>
#    All rights reserved.
#    BSD license.
"""
Method and functions based on the information theory of networks
"""
import numdifftools as nd
import numpy as np
from scipy.linalg import expm, logm, eigvalsh
from scipy.optimize import root
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

    def S(L, beta):
        l = eigvalsh(L)
        # l = l[l > 0] # introduces an error, entropy doesn't tend to 0 in beta=inf
        lrho = np.exp(-beta * l)
        Z = lrho.sum()
        return np.log(Z) + beta * (l * lrho).sum() / Z

    if 'density' in kwargs.keys():
        l = eigvalsh(kwargs['density'])
        return entropy(l[l > 0])

    elif 'A' in kwargs.keys() and 'beta' in kwargs.keys():
        A = kwargs['A']
        L = graph_laplacian(A)
        return S(L, kwargs['beta'])

    elif 'L' in kwargs.keys() and 'beta' in kwargs.keys():
        return S(kwargs['L'], kwargs['beta'])


def batch_compute_vonneumann_entropy(L, beta_range):
    """
    This function computes spectral entropy over a range of beta, given that L remains the same
    """
    l = eigvalsh(L)
    S = []
    for b in beta_range:
        lrho = np.exp(-b * l)
        Z = lrho.sum()
        S.append(np.log(Z) + b * (l * lrho).sum() / Z)
    S = np.array(S)
    S[np.isnan(S)] = 0
    return np.array(S)


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
    l = eigvalsh(L)

    def s(b):
        lrho = np.exp(-b * l)
        Z = lrho.sum()
        return np.log(Z) + b * (l * lrho).sum() / Z

    dsdb = nd.Derivative(lambda y: s(y), n=1)
    return np.array([dsdb(x) for x in beta_range])


def find_beta_logc(L, c):
    l = eigvalsh(L)

    def s(b, l):
        lrho = np.exp(-b * l)
        Z = lrho.sum()
        return np.log(Z) + b * (l * lrho).sum() / Z

    return root(lambda x: s(x, l) - np.log(c), x0=1).x


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
