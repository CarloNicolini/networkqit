#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 14:48:46 2018

@author: carlo2
"""
from networkqit.graphtheory import graph_laplacian as graph_laplacian
import numpy as np
import math
import numdifftools as nd

class Model(object):
    def __init__(self, **kwargs):
        super().__init__()
        self.args_mapping = None
        self.num_classes = len(Model.__subclasses__())
        self.model_type = kwargs.get('model_type', None)
        self.parameters = kwargs
        self.bounds = None

    def __call__(self, x):
        return self.model(x)

    def __add__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Mul(self, other)

    def __truediv__(self, other):
        return Div(self, other)

    def __floordiv__(self, other):
        return Div(self, other)

    def adjacency(self, *args):
        raise NotImplementedError
        pass

    def laplacian_gradient(self, *args):
        return nd.Gradient(lambda x : graph_laplacian(self(x)))(np.array([*args]))
        #raise NotImplementedError
        #pass

    def laplacian(self, *args):
        return graph_laplacian(self.adjacency(*args))

    def model(self, x):
        return self.adjacency(*[x[i] for i in range(0, len(self.args_mapping))])

    def model_grad(self, x):
        return self.laplacian_gradient(*x)

    
class Operator(Model):
    def __init__(self, left, right):
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


class ErdosRenyi(Model):
    """
    Erdos-Renyi mean random field
    """

    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = ['c_er']
        self.model_type = 'topological'
        self.formula = '$c_{er}$'
        self.bounds = [(0, None)]

    def adjacency(self, *args):
        P = args[0]*(1-np.eye(self.parameters['N']))
        return P
    
    def model_grad(self, x):
        N = self.parameters['N']
        G = np.zeros([N,1,N])
        G[:,0,:] = (1-np.eye(N)) + (N-1)*np.eye(N)
        return G


class Edr(Model):
    """
    Exponential Distance Rule model
    """

    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = ['c_edr', 'mu_edr']
        self.model_type = 'spatial'
        self.formula = '$c_{edr} e^{-\mu_{edr} d_{ij}}$'
        self.bounds = [(0, None), (0, None)]

    def adjacency(self, *args):
        P = args[0] * np.exp(-args[1] * self.parameters['dij'])
        np.fill_diagonal(P, 0)
        return P
    # Manualy computation of the gradient function
    def model_grad(self, x):
        dij = self.parameters['dij']
        n = len(dij)
        dL = np.zeros((n,2,n))
        c = x[0]
        mu = x[1]
        dL[:,1,:] = c*dij*np.exp(-mu*dij)
        dL[:,1,:] += np.diag(-dL[:,1,:].sum(axis=0))
        dL[:,0,:] = (np.eye(n)-1)*np.exp(-mu*dij) + np.diag(np.exp(-mu*dij).sum(axis=0)-1)
        return dL

class EdrTruncated(Model):
    """
    Truncate Exponential Distance Rule model
    """
    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = ['c_edr_trunc', 'lambda_edr_trunc', 'b_edr_trunc']
        self.model_type = 'spatial'
        self.formula = '$\frac{e^{-d_{ij}/\lambda}} {\lambda \left(1-e^{-b/\lambda} \right)}$'
        self.bounds = [(0, None), (0, None), (0, None)]

    def adjacency(self, *args):
        c = args[0]
        l = args[1]
        P = args[0] * np.exp(-args[1] * self.parameters['dij'])
        P = c*np.exp(-dij/l)/(l*(1-np.exp(-b/l)))
        np.fill_diagonal(P, 0)
        return P


class PowerLawDistance(Model):
    """
    Distance rule based on power law
    """
    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = ['c_pldr', 'mu_pldr']
        self.model_type = 'spatial'
        self.formula = '$c_{pldr} d_{ij}^{-\mu_{pldr}}$'
        self.bounds = [(0, None), (0, None)]

    def adjacency(self, *args):
        P = args[0]*(self.parameters['dij']**(-args[1]))
        np.fill_diagonal(P, 0)
        return P


class Weibull(Model):
    """
    Weibull distribution
    Eq. 4.33 Barabasi - Network Science (advanced topis, power laws)
    """
    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = ['c_wei', 'mu_wei', 'gamma_wei']
        self.model_type = 'spatial'        
        self.formula = '$c_{wei} d_{ij}^{\gamma_{wei}-1} e^{-(\mu_{wei} d_{ij})^\gamma_{wei}}$'
        self.bounds = [(0, None), (0, None), (0, None)]

    def adjacency(self, *args):
        P = args[0]*np.exp(-(args[1]*self.parameters['dij'])**args[2])
        np.fill_diagonal(P, 0)
        return P


class EdrSum(Model):
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
        self.formula = '$\sum \limits_{l=0}^L c_l e^{-d_{ij} \mu_l}$'
        self.bounds = [(0,None)] *2 * num_exp

    def adjacency(self, *args):
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


class LevyFligth(Model):
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
        self.formula = '$c_{levy} $'
        self.bounds = [(0, None), (0, None), (0, None), (0,None)]

    def adjacency(self, *args):
        P = args[0]/((args[1]+self.parameters['dij'])**args[2]) * np.exp(-self.parameters['dij']/args[3])
        np.fill_diagonal(P, 0)
        return P


class TopoIdentity(Model):
    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = []
        self.model_type = 'topological'
        self.formula = '$1$'
        self.bounds = None

    def adjacency(self, *args):
        return (1-np.eye(self.parameters['num_nodes']))


class HiddenVariables(Model):
    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = ['x_' + str(i) for i in range(0, kwargs['N'])]
        self.model_type = 'topological'
        self.formula = '$1$'
        self.bounds = [(0, None) for i in range(0, kwargs['N'])]
        if self.parameters.get('powerlaw',False):
            self.args_mapping += ['gamma']
            self.bounds += [(0,None)]

    def adjacency(self, *args):
        if self.parameters.get('powerlaw',True):
            return np.outer([*args[0:-1]],[*args[0:-1]])**(-args[-1])
        else:
            return np.outer([*args],[*args])
        
class TopoDegreeProd(Model):
    """
    Topological product of graph degrees (strengths) with powerlaw
    """

    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = ['c_degprod', 'mu_degprod']
        self.model_type = 'topological'
        self.formula = '$c_{degprod} (k_i k_j)^{-\mu_{degprod}}$'
        self.bounds = [(0, None), (0, None)]

    def adjacency(self, *args):
        k = self.parameters['k']
        return args[0]*(np.outer(k, k)**(-args[1]))


class TopoDegreeAvg(Model):
    """
    Topological average of graph degrees (strength) with powerlaw
    """

    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = ['c_degavg', 'mu_degavg']
        self.model_type = 'topological'
        self.formula = '$0.5 c_{degavg} (k_i+ k_j)^{-\mu_{degavg}}$'
        self.bounds = [(0, None), (0, None)]

    def adjacency(self, *args):
        k = self.parameters['k']
        T = np.outer(k, np.ones([1, len(k)]))
        return args[0]*((T + T.T)/2)**(-args[1])


class TopoDegreeDiff(Model):
    """
    Topological absolute difference of graph degrees (strengths) with powerlaw
    pij = c (ki - kj)^mu
    """

    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = ['c_degdiff', 'mu_degdiff']
        self.model_type = 'topological'
        self.formula = '$c_{degdiff} (|k_i - k_j|)^{-\mu_{degdiff}}$'
        self.bounds = [(0, None), (0, None)]

    def adjacency(self, *args):
        k = self.parameters['k']
        T = np.outer(k, np.ones([1, len(k)]))
        return args[0]*(np.abs(T-T.T))**(-args[1])

class TopoJaccard(Model):
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
            self.formula = '$c_{jacc} (J_{ij})^{-\mu_{jacc}}$'
        else:
            self.formula = '$c_{commnei} (\sum_l A_{il}A_{lj})^{-\mu_{commnei}}$'
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
        
    def adjacency(self, *args):
        return args[0]*(self.M)**(-args[1])

class UBCM(Model):
    """
    Undirected binary configuration model
    pij = xi xj/(1+xi xj)
    """
    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = ['x_' + str(i) for i in range(0, kwargs['N'])]
        self.model_type = 'topological'
        self.formula = '$\frac{x_i x_j}{1+x_i x_j}$'
        self.bounds = [(0, None) for i in range(0, kwargs['N'])]

    def adjacency(self, *args):
        O = np.outer(args, args)
        return O / (1+O)

class UWCM(Model):
    """"
    Undirected weighted configuration model
    pij = yi yj/(1- yi yj)
    """
    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = ['y_' + str(i) for i in range(0, kwargs['N'])]
        self.model_type = 'topological'
        self.formula = '$\frac{y_i y_j}{1-y_i y_j}$'
        self.bounds = [(0, None) for i in range(0, kwargs['N'])]

    def adjacency(self, *args):
        O = np.outer(args, args)
        P = np.nan_to_num(O / (1.0 - O))
        P[P<=0] = np.finfo(float).eps
        return P

class UECM(Model):
    """
    Enhanced binary configuration model
    pij = (xixj * yiyj) / ((1 - yiyj + xixj * yiyj)*(1 - yiyj))
    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.args_mapping =   ['x_' + str(i) for i in range(0, kwargs['N'])] + ['y_' + str(i) for i in range(0, kwargs['N'])]
        self.formula = '$\frac{x_i x_j  y_i y_j)}{(1 - y_iy_j + x_i x_j y_i y_j)(1 - y_i y_j)}$'
        self.bounds = [(0, None) for i in range(0, kwargs['N'] ) ]*2
        self.N = kwargs['N']

    def adjacency(self, *args):
        x = args
        xixj = np.outer(x[0:self.N], x[0:self.N])
        yiyj = np.outer(x[self.N:], x[self.N:])
        #return (xixj * yiyj) / (1 - yiyj + xixj * yiyj)
        P = np.nan_to_num((xixj * yiyj) / ((1 - yiyj + xixj * yiyj)*(1 - yiyj)))
        P[P<0] = np.finfo(float).eps
        return P

    def adjacency_weighted(self, *args):
        x = args
        P = self.adjacency(*args)
        yiyj = np.outer(x[self.N:], x[self.N:])
        return P/(1-yiyj)

    
class SpatialCM(Model):
    """
    Implements the random graph model with spatial constraints from:
    Ruzzenenti, F., Picciolo, F., Basosi, R., & Garlaschelli, D. (2012). 
    Spatial effects in real networks: Measures, null models, and applications. 
    Physical Review E - Statistical, Nonlinear, and Soft Matter Physics, 86(6), 
    1–13. https://doi.org/10.1103/PhysRevE.86.066110
    """
    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = ['x_' + str(i) for i in range(0, kwargs['N'])] + ['gamma','z']
        self.formula = '$\frac{z x_i x_j e^{-\gamma d_{ij}}}{ 1 + z x_i x_j e^{-\gamma d_{ij}}  }$'
        self.bounds = [(0,None) for i in range(0,kwargs['N'])] + [(0,None),(0,None)]
        self.dij = kwargs['dij']
        self.expdij = np.exp(-kwargs['dij'])
        self.is_weighted = kwargs.get('is_weighted',False)
    
    def adjacency(self, *args):        
        O = args[-1]*np.outer(args[0:-2],args[0:-2]) * self.expdij**(args[-2])
        if self.is_weighted:
            return O / (1 + O)
        else:
            return O / (1-O)

class S1(Model):
    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = [
            'x_' + str(i) for i in range(0, kwargs['N'])] + ['beta', 'mu']
        self.model_type = 'topological'
        self.formula = '$\frac{1}{1+\left(\frac{d_{ij}}{\mu k_i k_j}\right)^\beta}$'
        self.bounds = [(0, None) for i in range(0, kwargs['N'])]

    def adjacency(self, *args):
        beta = args[-2]
        mu = args[-1]
        O = args[-1]*np.outer(args[:-2])
        return 1.0/(1.0 + O**args[-2])


class ModelFactory(object):
    def factory(type, **kwargs):
        if type == 'Edr':
            return Edr(**kwargs)
        if type == 'EdrStretched':
            return EdrStretched(**kwargs)
        assert 0, "Non supported model: " + type
    factory = staticmethod(factory)

    def __init__(self, **kwargs):
        super().__init__()
        self.model_type = kwargs.get('model_type')
        self.parameters = kwargs

    def __iter__(self):
        return self.__next__()

    def __next__(self):
        for m in Model.__subclasses__():
            if self.model_type is None:
                yield m
                #self.n = self.n+1
            if m(**self.parameters).model_type == self.model_type:
                yield m
                #self.n = self.n+1
