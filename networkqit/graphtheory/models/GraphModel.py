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
from networkqit.graphtheory import graph_laplacian as graph_laplacian


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

    def __call__(self, x):
        """
        This redefinition of the __call__ method allows to call any model with parameters x as if it is a standard function.
        """
        return self.model(x)

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

    def expected_adjacency(self, *args):
        """
        In this base class the expected binary adjacency is not implemented and has to be implemented by every single inherited class.
        """
        raise NotImplementedError

    def expected_weighted_adjacency(self, *args):
        """
        In this base class the expected weighted adjacency is not implemented and has to be implemented by every single inherited class.
        """
        raise NotImplementedError        

    def _expected_laplacian_grad_autodiff(self, *args):
        """
        If the user does not provide an implementation of the method
        """
        import numdifftools as nd
        return nd.Gradient(lambda x : graph_laplacian(self(x)))(np.array([*args]))

    def expected_laplacian(self, *args):
        """
        Returns the expected laplacian from the parameters of the model provided as a variable number of inputs.

        args:
            args is a list of input variables of any length.
        """
        return graph_laplacian(self.expected_adjacency(*args))

    def model(self, x):
        """
        Returns the expected adjacency matrix of a model with parameters specified in the numpy array x

        args:
            x (numpy.array): the parameters of the model.
        """
        return self.expected_adjacency(*[x[i] for i in range(0, len(self.args_mapping))])

    def expected_laplacian_grad(self, x):
        """
        Compute the gradient of the expected laplacian with respect to the parameters,
        a NxNxk array, where k is the length of the parameters vector.
        It uses automatic differentiation if this method is not explicitly overridden by inherited classes
        hence it can be slow to evaluate, but accurate.
        Automatic differentiation is provided by the Python numdifftools package.

        args:
            x (numpy.array): parameters vector.
        """
        return self._expected_laplacian_grad_autodiff(*x)

    def loglikelihood(self, G, *args):
        # implement here where G is the adjacency matrix (Weighted or binary)
        raise NotImplementedError

    def saddle_point(self, G, *args):
        raise NotImplementedError

    def sample_adjacency(self, *args, **kwargs):
        raise NotImplementedError

    
class Operator(GraphModel):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right
        # unique elements, order preserving

        def unique_ordered(seq):
            seen = set()
            seen_add = seen.add
            return [x for x in seq if not (x in seen or seen_add(x))]
        self.args_mapping = unique_ordered(
            left.args_mapping + right.args_mapping)
        self.n = self.args_mapping
        self.idx_args_left = [i for i, e in enumerate(
            self.args_mapping) if e in set(left.args_mapping)]
        self.idx_args_right = [i for i, e in enumerate(
            self.args_mapping) if e in set(right.args_mapping)]
        self.bounds = left.bounds + right.bounds
        self.formula = left.formula[0:-1] + str(self) + right.formula[1:]
        # print(self.args_mapping,self.idx_args_left,self.idx_args_right,self.right.args_mapping)
    def loglikelihood(self, G, *args):
        pass
    
    def saddle_point(self, G, *args):
        pass
    
    def expected_adjacency(self,*args):
        pass
    
    def expected_laplacian(self,*args):
        pass
    
    
class Mul(Operator):
    def __call__(self, x):
        return self.left(x[self.idx_args_left]) * self.right(x[self.idx_args_right])

    def __str__(self):
        return '*'


class Add(Operator):
    def __call__(self, x):
        return self.left(x[self.idx_args_left]) + self.right(x[self.idx_args_right])

    def __str__(self):
        return '+'


class Div(Operator):
    def __call__(self, x):
        return self.left(x[self.idx_args_left]) / self.right(x[self.idx_args_right])

    def __str__(self):
        return '/'


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
        self.bounds = [(0, None)]

    def expected_adjacency(self, *args):
        P = args[0]*(1-  np.eye(self.parameters['N']))
        return P
    
    def expected_laplacian_grad(self, x):
        N = self.parameters['N']
        G = np.zeros([N, 1, N])
        G[:,0,:] = (N-1) * np.eye(N) - (1-np.eye(N))
        return G

    def sample_adjacency(self, *args, **kwargs):
        from autograd import numpy as anp
        def sigmoid(x):
            rij = anp.random.random([self.N, self.N])
            rij = anp.triu(rij,1)
            rij += rij.T
            slope = kwargs.get('slope', 500)
            P = x * (1.0 - anp.eye(self.N))
            return 1.0 / (1.0 + anp.exp(-slope*(P-rij)) )
        A = anp.triu(sigmoid(x), 1)
        A += A.T
        return A


class IsingModel(GraphModel):
    """
    A model of N^2 independent variables
    """

    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        #self.args_mapping = ['c_er']
        self.model_type = 'topological'
        #self.formula = '$c_{er}$'
        self.bounds = [(0, None)] * self.N

    def expected_adjacency(self, *args):
        return np.reshape(*args,[self.N,self.N])
    
    def expected_laplacian_grad(self, x):
        raise NotImplementedError

    def sample_adjacency(self, *args, **kwargs):
        batch_size = kwargs.get('batch_size', 1)
        # sample symmetric random uniforms
        rij = np.random.random([batch_size, self.N, self.N])
        rij = np.triu(rij,1) # batched triu!
        rij += np.transpose(rij,axes=[0,2,1]) # batched transposition
        slope = kwargs.get('slope', 200)
        P = np.reshape(np.tile(np.reshape(*args,[self.N,self.N]), [batch_size,1]),[batch_size, self.N, self.N])
        A = 1.0 / (1.0 + np.exp(-slope*(P-rij)) )
        A = np.triu(A, 1)
        A += np.transpose(A,axes=[0,2,1])
        return A

class Edr(GraphModel):
    """
    Exponential Distance Rule model.
    The pairwise spatial distance matrix must be specified as a kwargs argument.
    For example, `M = Edr(dij=dij)`
    """

    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = ['c_edr', 'mu_edr']
        self.model_type = 'spatial'
        self.formula = r'$c_{edr} e^{-\mu_{edr} d_{ij}}$'
        self.bounds = [(0, None), (0, None)]

    def expected_adjacency(self, *args):
        P = args[0] * np.exp(-args[1] * self.parameters['dij'])
        np.fill_diagonal(P, 0)
        return P
    # Manualy computation of the gradient function
    def expected_laplacian_grad(self, x):
        dij = self.parameters['dij']
        n = len(dij)
        dL = np.zeros((n,2,n))
        c = x[0]
        mu = x[1]
        dL[:,1,:] = c*dij*np.exp(-mu*dij)
        dL[:,1,:] += np.diag(-dL[:,1,:].sum(axis=0))
        dL[:,0,:] = (np.eye(n)-1)*np.exp(-mu*dij) + np.diag(np.exp(-mu*dij).sum(axis=0)-1)
        return dL

class EdrTruncated(GraphModel):
    """
    Truncate Exponential Distance Rule model
    """
    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = ['c_edr_trunc', 'lambda_edr_trunc', 'b_edr_trunc']
        self.model_type = 'spatial'
        self.formula = r'$\frac{e^{-d_{ij}/\lambda}} {\lambda \left(1-e^{-b/\lambda} \right)}$'
        self.bounds = [(0, None), (0, None), (0, None)]

    def expected_adjacency(self, *args):
        c = args[0]
        l = args[1]
        b = args[2]
        P = args[0] * np.exp(-args[1] * self.parameters['dij'])
        P = c*np.exp(-self.parameters['dij']/l)/(l*(1-np.exp(-b/l)))
        np.fill_diagonal(P, 0)
        return P


class PowerLawDistance(GraphModel):
    """
    Distance rule based on power law
    """
    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = ['c_pldr', 'mu_pldr']
        self.model_type = 'spatial'
        self.formula = r'$c_{pldr} d_{ij}^{-\mu_{pldr}}$'
        self.bounds = [(0, None), (0, None)]

    def expected_adjacency(self, *args):
        P = args[0]*(self.parameters['dij']**(-args[1]))
        np.fill_diagonal(P, 0)
        return P


class Weibull(GraphModel):
    """
    Weibull distribution
    Eq. 4.33 Barabasi - Network Science (advanced topis, power laws)
    """
    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = ['c_wei', 'mu_wei', 'gamma_wei']
        self.model_type = 'spatial'        
        self.formula = r'$c_{wei} d_{ij}^{\gamma_{wei}-1} e^{-(\mu_{wei} d_{ij})^\gamma_{wei}}$'
        self.bounds = [(0, None), (0, None), (0, None)]

    def expected_adjacency(self, *args):
        P = args[0]*np.exp(-(args[1]*self.parameters['dij'])**args[2])
        np.fill_diagonal(P, 0)
        return P


class EdrSum(GraphModel):
    """
    Sum of exponential distance rules
    """
    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)

        num_exp = self.parameters['num_exponentials']
        self.args_mapping = [
            'c_edrsum_'+str(int(i/2)) if i % 2 == 0 else 'mu_edrsum_'+str(int(i/2)) for i in range(0, 2*num_exp)]
        self.model_type = 'spatial'
        self.formula = r'$\sum \limits_{l=0}^L c_l e^{-d_{ij} \mu_l}$'
        self.bounds = [(0,None)] *2 * num_exp

    def expected_adjacency(self, *args):
        nargs = len(args)
        if nargs % 2 is not 0:
            raise 'Non compatible number of exponential sums'

        num_exp = self.parameters['num_exponentials']
        N = len(self.parameters['dij'])
        P = np.zeros([N, N])
        for i in range(0, int(nargs/2)):
            P += args[i]*np.exp(-self.parameters['dij']*args[int(i+nargs/2)])
        np.fill_diagonal(P, 0)
        return P#/np.sum(args[0:int(nargs/2)])  # divide by sum of all constants


class LevyFligth(GraphModel):
    """
    Levy law for mobile phone users (Gonzalez et al, 2009)
    Understanding individual human mobility patterns, Nature 453 (2009) 779–782
    # https://ac.els-cdn.com/S037015731000308X/1-s2.0-S037015731000308X-main.pdf
    """

    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = ['c_levy', 'delta_r_0_levy',
                             'gamma_levy', 'cutoff_levy']
        self.model_type = 'spatial'
        self.formula = r'$c_{levy} $'
        self.bounds = [(0, None), (0, None), (0, None), (0,None)]

    def expected_adjacency(self, *args):
        P = args[0]/((args[1]+self.parameters['dij'])**args[2]) * np.exp(-self.parameters['dij']/args[3])
        np.fill_diagonal(P, 0)
        return P


class TopoIdentity(GraphModel):
    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = []
        self.model_type = 'topological'
        self.formula = r'$1$'
        self.bounds = None

    def expected_adjacency(self, *args):
        return 1-np.eye(self.parameters['num_nodes'])


class HiddenVariables(GraphModel):
    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = ['x_' + str(i) for i in range(0, kwargs['N'])]
        self.model_type = 'topological'
        self.formula = r'$1$'
        self.bounds = [(0, None) for i in range(0, kwargs['N'])]
        if self.parameters.get('powerlaw', False):
            self.args_mapping += ['gamma']
            self.bounds += [(0, None)]

    def expected_adjacency(self, *args):
        if self.parameters.get('powerlaw', True):
            return np.outer([*args[0:-1]], [*args[0:-1]])**(-args[-1])
        return np.outer([*args],[*args])

        
class TopoDegreeProd(GraphModel):
    """
    Topological product of graph degrees (strengths) with powerlaw
    """

    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = ['c_degprod', 'mu_degprod']
        self.model_type = 'topological'
        self.formula = r'$c_{degprod} (k_i k_j)^{-\mu_{degprod}}$'
        self.bounds = [(0, None), (0, None)]

    def expected_adjacency(self, *args):
        k = self.parameters['k']
        return args[0]*(np.outer(k, k)**(-args[1]))


class TopoDegreeAvg(GraphModel):
    """
    Topological average of graph degrees (strength) with powerlaw
    """

    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = ['c_degavg', 'mu_degavg']
        self.model_type = 'topological'
        self.formula = r'$0.5 c_{degavg} (k_i+ k_j)^{-\mu_{degavg}}$'
        self.bounds = [(0, None), (0, None)]

    def expected_adjacency(self, *args):
        k = self.parameters['k']
        T = np.outer(k, np.ones([1, len(k)]))
        return args[0]*((T + T.T)/2)**(-args[1])


class TopoDegreeDiff(GraphModel):
    """
    Topological absolute difference of graph degrees (strengths) with powerlaw
    pij = c (ki - kj)^mu
    """

    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = ['c_degdiff', 'mu_degdiff']
        self.model_type = 'topological'
        self.formula = r'$c_{degdiff} (|k_i - k_j|)^{-\mu_{degdiff}}$'
        self.bounds = [(0, None), (0, None)]

    def expected_adjacency(self, *args):
        k = self.parameters['k']
        T = np.outer(k, np.ones([1, len(k)]))
        return args[0]*(np.abs(T-T.T))**(-args[1])

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
        self.generate_matching() # then generate the matching
        self.bounds = [(0, None), (0, None)]

    def generate_matching(self):
        from scipy.spatial.distance import cdist
        # https://mathoverflow.net/questions/123339/weighted-jaccard-similarity
        # https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.spatial.distance.cdist.html
        if self.normalized:
            self.M = cdist(self.A, self.A, lambda u, v: (np.minimum(u,v).sum())/(np.maximum(u,v).sum()) )
        else:
            self.M = cdist(self.A, self.A, lambda u, v: np.minimum(u,v).sum() )
        
    def expected_adjacency(self, *args):
        return args[0]*(self.M)**(-args[1])

#class S1(GraphModel):
#    def __init__(self, **kwargs):
#        if kwargs is not None:
#            super().__init__(**kwargs)
#        self.args_mapping = [
#            'x_' + str(i) for i in range(0, kwargs['N'])] + ['beta', 'mu']
#        self.model_type = 'topological'
#        self.formula = '$\frac{1}{1+\left(\frac{d_{ij}}{\mu k_i k_j}\right)^\beta}$'
#        self.bounds = [(0, None) for i in range(0, kwargs['N'])]
#
#    def expected_adjacency(self, *args):
#        beta = args[-2]
#        mu = args[-1]
#        O = args[-1]*np.outer(args[:-2])
#        return 1.0/(1.0 + O**args[-2])


class ModelFactory():
    
    @staticmethod
    def factory(type, **kwargs):
        raise RuntimeError('Must implement all the models manually')
        if type == 'Edr':
            return Edr(**kwargs)
        assert 0, "Non supported model: " + type
    #factory = staticmethod(factory)

    def __init__(self, **kwargs):
        super().__init__()
        self.model_type = kwargs.get('model_type')
        self.parameters = kwargs

    def __iter__(self):
        return self.__next__()

    def __next__(self):
        for m in GraphModel.__subclasses__():
            if self.model_type is None:
                yield m
                #self.n = self.n+1
            if m(**self.parameters).model_type == self.model_type:
                yield m
                #self.n = self.n+1