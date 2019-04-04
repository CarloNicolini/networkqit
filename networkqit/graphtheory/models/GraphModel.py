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
Base class for the implementation of different graph models.
Here are specified many possible models of sparse and dense graphs with 
dependency on some input parameters.
"""

import autograd.numpy as np
from autograd import grad as agrad
from networkqit.graphtheory import graph_laplacian as graph_laplacian
from autograd.scipy.special import expit
from ..matrices import batched_symmetric_random, multiexpit
EPS = np.finfo(float).eps

class GraphModel:
    """
    GraphModel is the base class that defines all the operations inherited
    from models of expected adjacency matrix, expected Laplacian, gradients of
    expected laplacian and random graph models. It defines the behaviour of model
    operations such as addition or multiplication.
    """
    def __init__(self, **kwargs):
        """
        Optional arguments:
        kwargs:
            model_type: can be or topological or spatial and it is used 
                when inheriting from this class. In geometrical models, the 
                spatial distance matrix must be specified parameters: 
                    contains all the kwargs, used to memorize all the options.
            bounds: if necessary, provide a bound on the values of the model 
                    parameters.  Bounds must be provided in the form of list 
                    of tuples.
        """
        super().__init__()
        self.N = kwargs['N']
        self.args_mapping = None
        self.num_classes = len(GraphModel.__subclasses__())
        self.model_type = kwargs.get('model_type', None)
        self.parameters = kwargs
        self.bounds = None

    def __call__(self, theta):
        """
        This redefinition of the __call__ method allows to call any model with parameters $\theta$ as if it is a standard function.
        """
        return self.model(theta)

    def __add__(self, other):
        """
        Returns a GraphModel object that when called returns the entrywise sum of the expected adjacency matrices
        """
        return Add(self, other)

    def __mul__(self, other):
        """
        Returns a GraphModel object that when called returns the entrywise multiplication of the expected adjacency matrices
        """
        return Mul(self, other)

    def __truediv__(self, other):
        """
        Returns a GraphModel object that when called returns the entrywise division of the expected adjacency matrices
        """
        return Div(self, other)

    def __floordiv__(self, other):
        """
        Returns a GraphModel object that when called returns the entrywise division of the expected adjacency matrices
        """
        return Div(self, other)

    def expected_adjacency(self, theta):
        """
        In this base class the expected binary adjacency is not implemented and has to be implemented by every single inherited class.
        """
        raise NotImplementedError

    def expected_weighted_adjacency(self, theta):
        """
        In this base class the expected weighted adjacency is not implemented and has to be implemented by every single inherited class.
        """
        raise NotImplementedError        

    def expected_laplacian(self, theta):
        """
        Returns the expected laplacian from the parameters of the model provided as a variable number of inputs.

        args:
            theta are the model parameters
        """
        return graph_laplacian(self.expected_adjacency(theta))

    def model(self, theta):
        """
        Returns the expected adjacency matrix of a model with parameters specified in the numpy array x

        args:
            theta (numpy.array): the parameters of the model.
        """
        return self.expected_adjacency(*[theta[i] for i in range(0, len(self.args_mapping))])

    def expected_laplacian_grad(self, theta):
        """
        Compute the gradient of the expected laplacian with respect to the parameters,
        a NxNxk array, where k is the length of the parameters vector $\theta$.
        Automatic differentiation is provided by the Python autograd package.

        args:
            theta (numpy.array): parameters vector.
        """
        return agrad(lambda z : self.expected_laplacian(z))(theta)

    def loglikelihood(self, observed_adj, theta):
        """
        Calculates the loglikelihood of the model given the observed graph

        args:
            observed_adj (numpy.array): the adjacency matrix of the empirical graph
            theta (numpy:array): the parameters vector of the model
        """
        raise NotImplementedError

    def saddle_point(self, observed_adj, theta):
        """
        Returns the saddle point equations for the likelihood. For all models derived from maximum entropy framework
        this is equivalent to the gradients of the loglikelihood with respect to the model parameters, by the SL-theorem

        args:
            observed_adj (numpy.array): the adjacency matrix of the empirical graph
            theta (numpy:array): the parameters vector of the model
        """
        raise NotImplementedError

    def sample_adjacency(self, theta, batch_size=1, with_grads=False, slope=500):
        """
        Sample the adjacency matrix of the maximum entropy model with parameters theta.
        If a batch_size greater than 1 is given, a batch of random adjacency matrices is returned.
        If with_grads is True, the operation supports backpropagation.

        args:
            theta (numpy:array): the parameters vector of the model
            batch_size (int): the number of networks to sample
            with_grads (bool): if True it supports backpropagation over the parameters theta
            slope (float): the slope of the sigmoid curve that approximates the Heaviside step function.
        """
        raise NotImplementedError


class ErdosRenyi(GraphModel):
    """
    Erdos-Renyi expected model.
    When called it returns an adjacency matrix that is constant everywhere and zero on the diagonal.
    """
    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = ['c_er']
        self.model_type = 'topological'
        self.formula = '$c_{er}$'
        self.bounds = [(EPS, 1-EPS)]

    def expected_adjacency(self, theta):
        P = theta*(1 - np.eye(self.parameters['N']))
        return P
    
    def sample_adjacency(self, theta, batch_size=1, with_grads=False, slope=500):
        rij = batched_symmetric_random(batch_size, self.N)
        P = self.expected_adjacency(theta)
        if with_grads:
            A = expit(slope*(P-rij)) # broadcasting of P over rij
        else:
            A = (P > rij).astype(float)
        A = np.triu(A, 1)
        A += np.transpose(A,axes=[0,2,1])
        return A

    def loglikelihood(self, observed_adj, theta):
        A = (observed_adj>0).astype(float)
        Lstar = A.sum() / 2
        loglike = Lstar*np.log(theta) + (self.N*(self.N-1)/2 - Lstar)*(np.log(1-theta))
        return loglike

    def saddle_point(self, observed_adj, theta):
        A = (observed_adj>0).astype(float)
        Lstar = A.sum() / 2
        avgL = np.sum(theta * (self.N*(self.N-1)) / 2)
        return np.array([Lstar - avgL])

    def fit(self, G, x0=None, **opt_kwargs):
        from networkqit import MLEOptimizer
        A = (G>0).astype(float)
        if x0 is None:
            x0 = [0.1]
        opt = MLEOptimizer(A, x0=x0, model=self)
        sol = opt.run(**opt_kwargs)
        return sol

class IsingModel(GraphModel):
    """
    A model of N^2 independent Bernoulli variables
    """
    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        #self.args_mapping = ['c_er']
        self.model_type = 'topological'
        #self.formula = '$c_{er}$'
        self.bounds = [(0, None)] * self.N * self.N

    def expected_adjacency(self, theta):
        return np.reshape(theta,[self.N,self.N])
    
    def sample_adjacency(self, theta, batch_size=1, with_grads=False, slope=500):
        # sample symmetric random uniforms
        rij = batched_symmetric_random(batch_size, self.N)
        pij = np.reshape(theta,[self.N,self.N])
        #pij = np.reshape(np.tile(np.reshape(theta,[self.N,self.N]), [batch_size,1]),[batch_size, self.N, self.N])
        if with_grads:
            A = expit(slope*(pij-rij))
        else:
            A = (pij>rij).astype(float)
        #A = np.triu(A, 1)
        #A += np.transpose(A,axes=[0,2,1])
        return A

class Edr(GraphModel):
    """
    Exponential Distance Rule model.
    The pairwise spatial distance matrix must be specified as a kwargs argument.
    For example, `M = Edr(dij=dij)`
    """

    def __init__(self, dij, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = ['lambd_edr']
        self.model_type = 'spatial'
        self.formula = r'$e^{-\lambda_{edr} d_{ij}}$'
        self.bounds = [(0, None)]
        import copy
        self.dij = copy.deepcopy(dij)
        np.fill_diagonal(self.dij,1) # to avoid division warning
        self.invdij = 1/self.dij
        np.fill_diagonal(self.invdij,0) # set it back to 0

    def expected_adjacency(self, theta):
        P = theta[0] * np.exp(-theta[1] * self.dij)
        np.fill_diagonal(P, 0)
        return P

    def sample_adjacency(self, theta, batch_size=1, with_grads=False, slope=500):
        if with_grads:
            rij = batched_symmetric_random(batch_size, self.N)
            # to generate random weights, needs a second decorrelated random source
            W = -(theta[0]*(self.invdij))*np.log(1-rij) # oneline for exponential distribution
        else:
            W = np.random.exponential(-(self.dij)*theta[0], size=[batch_size,self.N,self.N])*theta[0]
            W = np.triu(W, 1)
            W += np.transpose(W, axes=[0,2,1])
        return W

# class EdrTruncated(GraphModel):
#     """
#     Truncate Exponential Distance Rule model
#     """
#     def __init__(self, **kwargs):
#         if kwargs is not None:
#             super().__init__(**kwargs)
#         self.args_mapping = ['c_edr_trunc', 'lambda_edr_trunc', 'b_edr_trunc']
#         self.model_type = 'spatial'
#         self.formula = r'$\frac{e^{-d_{ij}/\lambda}} {\lambda \left(1-e^{-b/\lambda} \right)}$'
#         self.bounds = [(0, None), (0, None), (0, None)]

#     def expected_adjacency(self, theta):
#         c = theta[0]
#         l = theta[1]
#         b = theta[2]
#         P = theta[0] * np.exp(-theta[1] * self.parameters['dij'])
#         P = c*np.exp(-self.parameters['dij']/l)/(l*(1-np.exp(-b/l)))
#         np.fill_diagonal(P, 0)
#         return P


# class PowerLawDistance(GraphModel):
#     """
#     Distance rule based on power law
#     """
#     def __init__(self, **kwargs):
#         if kwargs is not None:
#             super().__init__(**kwargs)
#         self.args_mapping = ['c_pldr', 'mu_pldr']
#         self.model_type = 'spatial'
#         self.formula = r'$c_{pldr} d_{ij}^{-\mu_{pldr}}$'
#         self.bounds = [(0, None), (0, None)]

#     def expected_adjacency(self, theta):
#         P = theta*(self.parameters['dij']**(-theta))
#         np.fill_diagonal(P, 0)
#         return P


# class EdrSum(GraphModel):
#     """
#     Sum of exponential distance rules
#     """
#     def __init__(self, **kwargs):
#         if kwargs is not None:
#             super().__init__(**kwargs)

#         num_exp = self.parameters['num_exponentials']
#         self.args_mapping = [
#             'c_edrsum_'+str(int(i/2)) if i % 2 == 0 else 'mu_edrsum_'+str(int(i/2)) for i in range(0, 2*num_exp)]
#         self.model_type = 'spatial'
#         self.formula = r'$\sum \limits_{l=0}^L c_l e^{-d_{ij} \mu_l}$'
#         self.bounds = [(0,None)] *2 * num_exp

#     def expected_adjacency(self, theta):
#         nargs = len(args)
#         if nargs % 2 is not 0:
#             raise 'Non compatible number of exponential sums'

#         num_exp = self.parameters['num_exponentials']
#         N = len(self.parameters['dij'])
#         P = np.zeros([N, N])
#         for i in range(0, int(nargs/2)):
#             P += args[i]*np.exp(-self.parameters['dij']*args[int(i+nargs/2)])
#         np.fill_diagonal(P, 0)
#         return P#/np.sum(args[0:int(nargs/2)])  # divide by sum of all constants


# class TopoIdentity(GraphModel):
#     def __init__(self, **kwargs):
#         if kwargs is not None:
#             super().__init__(**kwargs)
#         self.args_mapping = []
#         self.model_type = 'topological'
#         self.formula = r'$1$'
#         self.bounds = None

#     def expected_adjacency(self, *args):
#         return 1-np.eye(self.parameters['num_nodes'])


# class HiddenVariables(GraphModel):
#     def __init__(self, **kwargs):
#         if kwargs is not None:
#             super().__init__(**kwargs)
#         self.args_mapping = ['x_' + str(i) for i in range(0, kwargs['N'])]
#         self.model_type = 'topological'
#         self.formula = r'$1$'
#         self.bounds = [(0, None) for i in range(0, kwargs['N'])]
#         if self.parameters.get('powerlaw', False):
#             self.args_mapping += ['gamma']
#             self.bounds += [(0, None)]

#     def expected_adjacency(self, *args):
#         if self.parameters.get('powerlaw', True):
#             return np.outer([*args[0:-1]], [*args[0:-1]])**(-args[-1])
#         return np.outer([*args],[*args])

        
# class TopoDegreeProd(GraphModel):
#     """
#     Topological product of graph degrees (strengths) with powerlaw
#     """

#     def __init__(self, **kwargs):
#         if kwargs is not None:
#             super().__init__(**kwargs)
#         self.args_mapping = ['c_degprod', 'mu_degprod']
#         self.model_type = 'topological'
#         self.formula = r'$c_{degprod} (k_i k_j)^{-\mu_{degprod}}$'
#         self.bounds = [(0, None), (0, None)]

#     def expected_adjacency(self, *args):
#         k = self.parameters['k']
#         return args[0]*(np.outer(k, k)**(-args[1]))


# class TopoDegreeAvg(GraphModel):
#     """
#     Topological average of graph degrees (strength) with powerlaw
#     """

#     def __init__(self, **kwargs):
#         if kwargs is not None:
#             super().__init__(**kwargs)
#         self.args_mapping = ['c_degavg', 'mu_degavg']
#         self.model_type = 'topological'
#         self.formula = r'$0.5 c_{degavg} (k_i+ k_j)^{-\mu_{degavg}}$'
#         self.bounds = [(0, None), (0, None)]

#     def expected_adjacency(self, *args):
#         k = self.parameters['k']
#         T = np.outer(k, np.ones([1, len(k)]))
#         return args[0]*((T + T.T)/2)**(-args[1])


# class TopoDegreeDiff(GraphModel):
#     """
#     Topological absolute difference of graph degrees (strengths) with powerlaw
#     pij = c (ki - kj)^mu
#     """

#     def __init__(self, **kwargs):
#         if kwargs is not None:
#             super().__init__(**kwargs)
#         self.args_mapping = ['c_degdiff', 'mu_degdiff']
#         self.model_type = 'topological'
#         self.formula = r'$c_{degdiff} (|k_i - k_j|)^{-\mu_{degdiff}}$'
#         self.bounds = [(0, None), (0, None)]

#     def expected_adjacency(self, *args):
#         k = self.parameters['k']
#         T = np.outer(k, np.ones([1, len(k)]))
#         return args[0]*(np.abs(T-T.T))**(-args[1])

class TopoJaccard(GraphModel):
    """
    Topological models with probability link given by Jaccard coefficient
    For weighted graphs it automatically uses the Weighted Jaccard similarity.
    If 'normalized' is specified to False this model reproduces the
    "Economical Preferential Attachment model"
    Here we set the powerlaw exponent to be unbounded
    Reference:
    Vertes et al. Simple models of human brain functional networks. 
    PNAS (2012) https://doi.org/10.1073/pnas.1111738109
    """
    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = ['c_nei', 'mu_nei']
        self.model_type = 'topological'
        self.normalized = kwargs.get('normalized', True)
        if self.normalized:
            self.formula = r'$c_{jacc} (J_{ij})^{-\mu_{jacc}}$'
        else:
            self.formula = r'$c_{commnei} (\sum_l A_{il}A_{lj})^{-\mu_{commnei}}$'
        self.A = kwargs['A'] # save the adjacency matrix        
        self._generate_matching() # then generate the matching
        self.bounds = [(0, None), (0, None)]

    def _generate_matching(self):
        from scipy.spatial.distance import cdist
        # https://mathoverflow.net/questions/123339/weighted-jaccard-similarity
        # https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.spatial.distance.cdist.html
        if self.normalized:
            self.M = cdist(self.A, self.A, lambda u, v: (np.minimum(u,v).sum())/(np.maximum(u,v).sum()) )
        else:
            self.M = cdist(self.A, self.A, lambda u, v: np.minimum(u,v).sum() )
        
    def expected_adjacency(self, theta):
        return theta[0]*(self.M)**(-theta[1])

    def sample_adjacency(self, theta, batch_size=1, with_grads=False, slope=500):
        return None

class EconomicalClustering(GraphModel):
    """
    # We use the Matching Index model of
    # Generative models of the human connectome
    # Betzel, Richard F. et al, Neuroimage (2016)
    """
    def __init__(self, G, dij, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.G = G # save the adjacency matrix        
        self.model_type = 'topological'
        self.dij = dij
        self.normalized = kwargs.get('normalized', True)
        if self.normalized:
            self.formula = r'$c_{jacc} (J_{ij})^{-\mu_{jacc}}$'
        else:
            self.formula = r'$c_{commnei} (\sum_l A_{il}A_{lj})^{-\mu_{commnei}}$'
        self._generate_matching() # then generate the matching
        self.bounds = [(None, None), (None, None)]

    def _generate_matching(self):
        from scipy.spatial.distance import cdist
        # https://mathoverflow.net/questions/123339/weighted-jaccard-similarity
        # https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.spatial.distance.cdist.html
        if self.normalized:
            self.K = cdist(self.G, self.G, lambda u, v: (np.minimum(u,v).sum())/(np.maximum(u,v).sum()) )
        else:
            self.K = cdist(self.G, self.G, lambda u, v: np.minimum(u,v).sum() )
        
    def expected_adjacency(self, theta):
        p1 = (np.eye(self.N) + self.dij)**(theta[0])
        p1 *= (1-np.eye(self.N))
        p2 = (self.K**(theta[1]))
        return  p1 * p2


    def sample_adjacency(self, theta, batch_size=1, with_grads=False, slope=500):
        rij = batched_symmetric_random(batch_size, self.N)
        pij = self.expected_adjacency(theta)
        if with_grads:
            A = expit(slope*(pij-rij)) # sampling, approximates binomial with continuos
        else:
            A = (pij>rij).astype(float)
        A = np.triu(A, 1) # make it symmetric
        A += np.transpose(A, axes=[0, 2, 1])
        return A


# class ModelFactory():
    
#     @staticmethod
#     def factory(type, **kwargs):
#         raise RuntimeError('Must implement all the models manually')
#         if type == 'Edr':
#             return Edr(**kwargs)
#         assert 0, "Non supported model: " + type
#     #factory = staticmethod(factory)

#     def __init__(self, **kwargs):
#         super().__init__()
#         self.model_type = kwargs.get('model_type')
#         self.parameters = kwargs

#     def __iter__(self):
#         return self.__next__()

#     def __next__(self):
#         for m in GraphModel.__subclasses__():
#             if self.model_type is None:
#                 yield m
#                 #self.n = self.n+1
#             if m(**self.parameters).model_type == self.model_type:
#                 yield m
#                 #self.n = self.n+1
