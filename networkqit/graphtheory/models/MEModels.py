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
Some Maximum entropy graph models, inherit from ExpectedModel.
See Tiziano Squartini thesis for more details
"""

from networkqit.graphtheory import graph_laplacian as graph_laplacian
import numpy as np
import numdifftools as nd
from .ExpectedGraphModel import ExpectedModel

class UBCM(ExpectedModel):
    """
    Undirected binary configuration model
    
    Hamiltonian of the problem:
    H(A) = sum_{i<j} (theta_i) k_i^*
    
    Expected link probability
    pij = xi xj/(1+xi xj)
    
    where x_i = exp(-theta_i)
    """
    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.N = kwargs['N']
        self.args_mapping = ['x_' + str(i) for i in range(0, self.N)]
        self.model_type = 'topological'
        self.formula = '$\frac{x_i x_j}{1+x_i x_j}$'
        self.bounds = [(0, None) for i in range(0, self.N)]

    def expected_adjacency(self, *args):
        xixj = np.outer(args, args)
        return xixj / (1+xixj)

    def expected_laplacian_grad(self, x):
        N = self.N 
        g = np.zeros([N,N,N])
        for l in range(0,N):
            degL = 0
            v = set(range(0,N))-set([l])
            for k in v:
                degL += x[k]/((1+x[l]*x[k])**2)
            for i in range(0,N):
                g[i,i,l] = x[i]/((1+x[i]*x[l])**2)
            g[l,l,l] = degL
            for i in v:
                g[l,i,l] = -x[i]/((1+x[i]*x[l])**2)
                g[:,l,l] = g[l,:,l]
        return g

    def likelihood(self, G, *args):
        # See the definition here:
        # Garlaschelli, D., & Loffredo, M. I. (2008).
        # Maximum likelihood: Extracting unbiased information from complex networks.
        # PRE 78(1), 1–4. https://doi.org/10.1103/PhysRevE.78.015101
        # G here is the graph binary adjacency matrix
        pij = self.expected_adjacency(*args)
        #one_min_pij = 1.0 - pij
        #one_min_pij[one_min_pij <= 0] = np.finfo(float).eps
        #loglike = (W * (np.log(pij) - np.log(one_min_pij)) + np.log(one_min_pij)).sum()
        #loglike = np.triu(W * (np.log(pij) - np.log(one_min_pij)) + np.log(one_min_pij), 1).sum()
        loglike = G*np.log(pij) + (1-G)*np.log(1-pij)
        loglike[np.logical_or(np.isnan(loglike), np.isinf(loglike))] = 0
        return np.triu(loglike,1).sum()

class UWCM(ExpectedModel):
    """"
    Undirected weighted configuration model
    Constraints: strength sequence
    Hamiltonian of the problem
    H(W) = sum_{i<j} theta_i s_i^*
    
    Expected number of links
    wij = yi yj/(1- yi yj)

    where y_i = exp(-theta_i)
    """
    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = ['y_' + str(i) for i in range(0, kwargs['N'])]
        self.model_type = 'topological'
        self.formula = '$\frac{y_i y_j}{1-y_i y_j}$'
        self.bounds = [(0, None) for i in range(0, kwargs['N'])]

    def expected_adjacency(self, *args):
        O = np.outer(args, args)
        P = np.nan_to_num(O / (1.0 - O))
        P[P<=0] = np.finfo(float).eps
        return P
    
    def likelihood(self, G, *args):
        pij = np.outer(args,args) # see Squartini thesis page 116
        loglike = G*np.log(pij) + (np.log(1-pij))
        loglike[np.logical_or(np.isnan(loglike), np.isinf(loglike))] = 0
        return np.triu(loglike,1).sum()


class UBWRG(ExpectedModel):
    """
    Undirected binary weighted random graph model
    Constraints: total number of links L, total weight W
    Hamiltonian:
        
    H(G) = sum_{i<j} \alpha Heaviside(w_{ij}) + (beta_i + beta_j) w_{ij}
         = \alpha L(A) + beta Wtot(W)
    
    Substitutions:
        x = exp(-alpha)
        y = exp(-beta)
    
    Number of paramters: 2
    
    Link probability <aij>
    <aij> = pij = x*y/(1 - y + x*y)
    
    Expected link weights <wij>
    <wij> = (x*y) / ( (1-y + x*y)*(1-y) )
    
    Reference: Squartini thesis page 124
    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.args_mapping =   ['x_' + str(i) for i in range(0, kwargs['N'])] + ['y_' + str(i) for i in range(0, kwargs['N'])]
        #self.formula = '$\frac{x  y_i y_j)}{(1 - y_iy_j + x y_i y_j)(1 - y_i y_j)}$' TODO
        self.N = kwargs['N']
        self.bounds = [(0, None)]*(self.N+1)
        
    def expected_adjacency(self, *args):
        x,y = args[0], args[1]
        pij = np.nan_to_num((x * y) / ((1 - y + x * y)))
        pij[pij<0] = np.finfo(float).eps
        return pij

    def adjacency_weighted(self, *args):
        y = args[(self.N):]
        pij = self.expected_adjacency(*args)
        yiyj = np.outer(y,y)
        return pij / (1-yiyj)

    def likelihood(self, G, *args):
        # see Squartini thesis page 126
        x,y = args[0:self.N], args[(self.N):]
        k = (G>0).sum(axis=0)
        s = G.sum(axis=0)
        xij = np.outer(x,x)
        yij = np.outer(y,y)
        loglike = (x*k).sum() + (y*s).sum() + np.triu(np.log( (1-yij)/(1-yij + xij*yij) ) ,1).sum()
        return loglike

class UECM2(ExpectedModel):
    """
    Undirected Enhanced binary configuration model II
    Constraints: total number of links L, strength sequence s_i
    Hamiltonian:
        
    H(G) = sum_{i<j} \alpha Heaviside(w_{ij}) + (beta_i + beta_j) w_{ij}
         = \alpha L(A) + beta_i s_i^*
    
    Substitutions:
        x = exp(-alpha)
        y_i = exp(-beta_i)
    
    Number of paramters: N + 1
    
    Link probability <aij>
    <aij> = pij = x*yiyj/(1-yiyj+x*yiyj)
    
    Expected link weights <wij>
    <wij> = (x*yiyj) / ( (1-yiy+x*yiyj)*(1-yiyj) )
    
    Reference: Squartini thesis page 124
    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.args_mapping =   ['x_' + str(i) for i in range(0, kwargs['N'])] + ['y_' + str(i) for i in range(0, kwargs['N'])]
        #self.formula = '$\frac{x  y_i y_j)}{(1 - y_iy_j + x y_i y_j)(1 - y_i y_j)}$' TODO
        self.N = kwargs['N']
        self.bounds = [(0, None)]*(self.N+1)
        
    def expected_adjacency(self, *args):
        x,y = args[0:self.N], args[(self.N):]
        xij = np.outer(x,x)
        yij = np.outer(y,y)
        pij = np.nan_to_num((xij * yij) / ((1 - yij + xij * yij)))
        pij[pij<0] = np.finfo(float).eps
        return pij

    def adjacency_weighted(self, *args):
        y = args[(self.N):]
        pij = self.expected_adjacency(*args)
        yiyj = np.outer(y,y)
        return pij / (1-yiyj)

    def likelihood(self, G, *args):
        # see Squartini thesis page 126
        x,y = args[0:self.N], args[(self.N):]
        k = (G>0).sum(axis=0)
        s = G.sum(axis=0)
        xij = np.outer(x,x)
        yij = np.outer(y,y)
        loglike = (x*k).sum() + (y*s).sum() + np.triu(np.log( (1-yij)/(1-yij + xij*yij) ) ,1).sum()
        return loglike

class UECM3(ExpectedModel):
    """
    Undirected Enhanced binary configuration model III
    Constraints: degree sequence, strength sequence
    Hamiltonian:
        
    H(G) = sum_{i<j} (alpha_i+alpha_j) Heaviside(w_{ij}) + (beta_i + beta_j) w_{ij}
         = sum_i alpha_i k_i^* + beta_i s_i^*
    
    Substitutions:
        x = exp(-alpha)
        y_i = exp(-beta_i)
        
    Number of parameters: 2N
    
    Link probability <aij>
    <aij> = pij = (xixj * yiyj) / ((1 - yiyj + xixj * yiyj))
    
    Expected link weights <wij>
    <wij> = (xixj*yiyj) / ( (1-yiy+xixj*yiyj)*(1-yiyj) )
    
    Reference: Squartini thesis page 126
    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.args_mapping =   ['x_' + str(i) for i in range(0, kwargs['N'])] + ['y_' + str(i) for i in range(0, kwargs['N'])]
        #self.formula = '$\frac{x_i x_j  y_i y_j)}{(1 - y_iy_j + x_i x_j y_i y_j)(1 - y_i y_j)}$' TODO
        self.N = kwargs['N']
        self.bounds = [(0,None)]*(2*self.N)
        

    def expected_adjacency(self, *args):
        x,y = args[0:self.N], args[(self.N):]
        xij = np.outer(x,x)
        yij = np.outer(y,y)
        pij = np.nan_to_num((xij * yij) / ((1 - yij + xij * yij)))
        pij[pij<0] = np.finfo(float).eps
        return pij

    def adjacency_weighted(self, *args):
        y = args[(self.N):]
        pij = self.expected_adjacency(*args)
        yiyj = np.outer(y,y)
        return pij / (1.0-yiyj)

    def likelihood(self, G, *args):
        # see Squartini thesis page 126
        x,y = args[0:self.N], args[(self.N):]
        k = (G>0).sum(axis=0)
        s = G.sum(axis=0)
        xij = np.outer(x,x)
        yij = np.outer(y,y)
        loglike = (x*k).sum() + (y*s).sum() + np.triu(np.log( (1-yij)/(1-yij + xij*yij) ) ,1).sum()
        return loglike

class cWECMt1(ExpectedModel):
    """
    Enhanced Weighted Continuous Configuration Model with threshold only on link presence
    Hamiltonian of the problem
    H(W) = sum_{i<j} alpha_i Heaviside(w_{ij}-t) + beta_i w_{ij}

    Expected link probability
    pij = xij*((yij)**t)/(1.0+xij*yijt - yijt)

    Expected link weight
    (t*(xij-1.0)*yijt)/((1.0 + xij*yijt - yijt )) - 1.0/(np.log(np.abs(yij+eps)))
    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.args_mapping =   ['x_' + str(i) for i in range(0, kwargs['N'])] + ['y_' + str(i) for i in range(0, kwargs['N'])]
        self.formula = '$\frac{x_i x_j (y_i y_j)^t}{1+ x_i x_j(y_i y_j)^t - (y_i y_j^t)}'
        self.bounds = [(0, None) for i in range(0, kwargs['N'] ) ]*2
        self.bounds = [0,np.inf]
        self.N = kwargs['N']
        self.threshold = kwargs['threshold']

    def expected_adjacency(self, *args):
        x,y = args[0:self.N], args[(self.N):]

        t = self.threshold
        xij = np.outer(x,x)
        yij = np.outer(y,y)
        yijt = np.real(yij**t)
        pij = np.nan_to_num(xij*((yij)**t)/(1.0+xij*yijt - yijt))
        pij[pij<0] = np.finfo(float).eps
        return pij
    
    def adjacency_weighted(self, *args):
        x,y = args[0:self.N], args[(self.N):]
        t = self.threshold
        xij = np.outer(x,x)
        yij = np.outer(y,y)
        yijt = np.real(yij**t)
        eps = 1E-16
        wij = np.nan_to_num((t*(xij-1.0)*yijt)/((1.0 + xij*yijt - yijt )) - 1.0/(np.log(np.abs(yij+eps))))
        wij[wij<0] = eps
        return wij


class cWECMt2(ExpectedModel):
    """
    Enhanced Weighted Continuous Configuration Model with threshold only on link presence and weight
    Hamiltonian of the problem
    H(W) = sum_{i<j} (alpha_i+alpha_j) Heaviside(w_{ij}-t) + (beta_i+beta_j) w_{ij} Heaviside(w_{ij}-t)

    Expected link probability
    pij = xij*(yij^t) / (t*log(yij) + xij*(yij)^t)

    Expected link weight
    wij = (1+yij^t)(xij (yij)^t) / ( log(yij)*(t*log(yij) +xij*(yij)^t ) )
    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.args_mapping =   ['x_' + str(i) for i in range(0, kwargs['N'])] + ['y_' + str(i) for i in range(0, kwargs['N'])]
        self.formula = '$\frac{x_i x_j (y_i y_j)^t}{t \log(yi y_j) + x_i x_j (y_i y_j)^t}$'
        self.bounds = [(0, None) for i in range(0, kwargs['N'] ) ]*2
        self.N = kwargs['N']
        self.threshold = kwargs['threshold']

    def expected_adjacency(self, *args):
        x,y = args[0:self.N], args[(self.N):]
        t = self.threshold
        xij = np.outer(x,x)
        yij = np.outer(y,y)
        yijt = np.real(yij**t)
        eps = 1E-16
        pij = np.nan_to_num(xij*(yij**t) / (t*np.log(yij+eps) + xij*(yij)**t))
        pij[pij<0] = np.finfo(float).eps
        return pij
    
    def adjacency_weighted(self, *args):
        x,y = args[0:self.N], args[(self.N):]
        t = self.threshold
        xij = np.outer(x,x)
        yij = np.outer(y,y)
        yijt = np.real(yij**t)
        eps = 1E-16
        num = (1+yijt)*(xij)*(yijt)
        den = np.log(np.abs(yij+eps)) * ( np.log(np.abs(yij+eps)) + xij*yijt ) 
        wij = np.nan_to_num( num / den )
        wij[wij<0] = eps
        return wij



class SpatialCM(ExpectedModel):
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
    
    def expected_adjacency(self, *args):        
        O = args[-1]*np.outer(args[0:-2],args[0:-2]) * self.expdij**(args[-2])
        if self.is_weighted:
            return O / (1 + O)
        return O / (1-O)
