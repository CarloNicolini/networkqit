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
Some Maximum entropy graph models, inherit from GraphModel.
See Tiziano Squartini thesis for more details
Also reference:
Garlaschelli, D., & Loffredo, M. I. (2008).
Maximum likelihood: Extracting unbiased information from complex networks.
PRE 78(1), 1–4. https://doi.org/10.1103/PhysRevE.78.015101

G here is the graph adjacency matrix, A is the binary adjacency, W is the weighted
"""

import autograd.numpy as np
from .GraphModel import GraphModel

EPS = np.finfo(float).eps


class UBCM(GraphModel):
    """
    1. Model name: Undirected Binary Configuration Model

    2. Constraints: degree sequence k_i^*
    
    3. Hamiltonian:
        H(A) = sum_{i<j} (alpha_i) k_i^*
    
    4. Lagrange multipliers substitutions:
        x_i = exp(-alpha_i)
    
    5. Number of parameters:
        N
    
    6. Link probability <aij>
        <aij> = pij = xixj/(1 + xixj)

    7. Expected link weight <wij>
        <wij> = <aij>

    8. LogLikelihood logP:
        sum_{i<j} a_{ij}*log(pij) + (1-aij)*log(1-pij)
    
    9. Reference:
        Squartini thesis page 110
    """
    
    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = ['x_' + str(i) for i in range(0, self.N)]
        self.model_type = 'topological'
        self.formula = '$p_{ij} = \frac{x_i x_j}{1+x_i x_j}$'
        self.bounds = [(EPS, None) for i in range(0, self.N)]

    def expected_adjacency(self, *args):
        xixj = np.outer(args, args)
        return xixj / (1 + xixj)

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

    def loglikelihood(self, G, *args):
        pij = self.expected_adjacency(*args)
        loglike = G * np.log(pij + EPS) + (1 - G)*np.log(1 - pij - EPS)
        loglike[np.logical_or(np.isnan(loglike), np.isinf(loglike))] = 0
        return np.triu(loglike, 1).sum()
    
    def saddle_point(self, G, *args):
        k = (G>0).sum(axis=0)
        pij = self.expected_adjacency(*args)
        avgk = pij.sum(axis=0)
        return k-avgk

    def sample_adjacency(self, *args, **kwargs):
        batch_size = kwargs.get('batch_size', 1)
        rij = np.random.random([batch_size, self.N, self.N])
        rij = np.triu(rij, 1)
        rij += np.transpose(rij,[0,2,1]) # transpose last axis
        slope = kwargs.get('slope', 200.0)
        batch_args = np.tile(*args,[batch_size, 1]) # replicate
        xixj = np.einsum('ij,ik->ijk', batch_args, batch_args)
        P = xixj / (1.0 + xixj)
        A = 1.0 / (1.0 + np.exp(-slope*(P-rij))) # sampling, approximates binomial with continuos
        A = np.triu(A, 1) # make it symmetric
        A += np.transpose(A, axes=[0, 2, 1])
        return A

class UWCM(GraphModel):
    """"
    1. Model name: Undirected Weighted Configuration Model

    2. Constraints: strength sequence s_i^*
    
    3. Hamiltonian:
        H(A) = sum_{i<j} (beta_i) s_i^*
    
    4. Lagrange multipliers substitutions:
        y_i = exp(-beta_i)
        pij = y_i y_j
    
    5. Number of parameters:
        N
    
    6. Link probability <aij>:
        <aij> = pij = yiyj

    7. Expected link weight <wij>:
        <wij> = yiyj/(1-yiyj)

    8. LogLikelihood logP:
        logP = sum_{i<j} w_{ij}*log(pij) + log(1-pij)
    
    9. Reference:
        Squartini thesis page 110
    """
    
    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = ['y_' + str(i) for i in range(0, kwargs['N'])]
        self.model_type = 'topological'
        #self.formula = '$<w_{ij}> = \frac{y_i y_j}{1-y_i y_j}$'
        self.bounds = [(EPS, None) for i in range(0, self.N)]

    def expected_adjacency(self, *args):
        self.pij = np.outer(args, args)
        return self.pij
    
    def expected_weighted_adjacency(self, *args):
        pij = self.expected_adjacency(args)
        self.wij = pij / (1 - pij)
        return self.wij
    
    def loglikelihood(self, G, *args):
        pij = self.expected_adjacency(args)
        loglike = G*np.log(pij) + (np.log(1-pij))
        loglike[np.logical_or(np.isnan(loglike), np.isinf(loglike))] = 0
        return np.triu(loglike,1).sum()

    def saddle_point(self, G, *args):
        s = G.sum(axis=0)
        wij = self.expected_weighted_adjacency(args)
        return s - wij.sum(axis=0)

    def sample_adjacency(self, *args, **kwargs):
        batch_size = kwargs.get('batch_size', 1)
        rij = np.random.random([batch_size, self.N, self.N])
        rij = np.triu(rij, 1)
        rij += np.transpose(rij, [0, 2, 1])  # transpose last axis
        slope = kwargs.get('slope', 200.0)
        batch_args = np.tile(*args, [batch_size, 1])  # replicate
        yiyj = np.einsum('ij,ik->ijk', batch_args, batch_args)
        P = yiyj / (1.0 - yiyj)
        # TODO IMPLEMENT GEOMETRIC SAMPLING
        A = 1.0 / (1.0 + np.exp(-slope * (P - rij)))  # sampling, approximates binomial with continuos
        A = np.triu(A, 1)  # make it symmetric
        A += np.transpose(A, axes=[0, 2, 1])
        return A


class UBWRG(GraphModel):
    """
    1. Model name: Undirected Binary Weighted Random Graph model
    
    2. Constraints: total number of links L, total weight W
    3. Hamiltonian:
        H(G) = sum_{i<j} \alpha Heaviside(w_{ij}) + beta w_{ij}
             = \alpha L(A) + beta Wtot(W)
    
    4. Substitutions:
        x = exp(-alpha)
        y = exp(-beta)
    
    5. Number of paramters:
        2
    
    6. Link probability <aij>
        <aij> = pij = x*y/(1 - y + x*y)
    
    7. Expected link weights <wij>
        <wij> = (x*y) / ( (1-y + x*y)*(1-y) )
    
    8. Loglikelihood logP:
        logP = L(A)log(x) +  Wtot(W)log(y) + N(N-1)/2 log( (1-y)/(1-y + xy) )
        
    Reference: Squartini thesis page 122
    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.args_mapping =   ['x','y']
        # TODO self.formula = '$\frac{x  y_i y_j)}{(1 - y_iy_j + x y_i y_j)(1 - y_i y_j)}$'
        self.N = kwargs['N']
        self.bounds = [(EPS, None)]*(self.N+1)
        
    def expected_adjacency(self, *args):
        x,y = args[0], args[1]
        pij = np.nan_to_num((x * y) / ((1 - y + x * y)))
        pij[pij<0] = EPS
        return pij

    def expected_weighted_adjacency(self, *args):
        y = args[1]
        pij = self.expected_adjacency(*args)
        return pij / (1-y)

    def loglikelihood(self, G, *args):
        x,y = args[0], args[1]
        L = (np.triu(G,1)>0).sum()
        Wtot = (np.triu(G,1)).sum()
        loglike = L*np.log(x) + Wtot*np.log(y) + self.N*(self.N-1)/2 * np.log( (1-y)/(1-y+x*y))
        return loglike
    
    def saddle_point(self, G, *args):
        L = np.triu(G>0,1).sum()
        Wtot = np.triu(G,1).sum()
        p = self.expected_adjacency(*args)
        w = self.expected_weighted_adjacency(*args)
        pairs = self.N*(self.N-1)/2
        return np.hstack([L-p*pairs,Wtot-w*pairs])


class UECM3(GraphModel):
    """
    1. Model name: Undirected Enhanced Configuration model III

    2. Constraints: strenght sequence s_i^*, degree sequence k_i^*

    3. Hamiltonian:
        H(G) = sum_i \alpha_i k_i^* + \beta_i s_i^* 
             = sum_{i<j} (\alpha_i+\alpha_i) Heaviside(w_{ij}) + (beta_i + beta_j) w_{ij}

    4. Lagrange multipliers substitutions:
        x_i = exp(-alpha_i)
        y_i = exp(-beta_i)

    5. Number of parameters:
        2N

    6. Link probability <aij>:
        <aij> = (xixj yiyj) / (1 - yiyj + xixj*yiyj)

    7. Link probability <wij>:
        <wij> = (xixj yiyj) / ((1 - yiyj + xixj*yiyj)(1-yiyj))

    8. LogLikelihood logP:
        logP = sum_i k_i(A) log(x_i) + sum_i s_i(W) log(y_i) + sum_{i<j} log( (1-yiyj) / (1-yiyj+xixj*yiyj) )

    9. Reference:
        Squartini thesis page 126
    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.args_mapping =   ['x_' + str(i) for i in range(0, kwargs['N'])] + ['y_' + str(i) for i in range(0, kwargs['N'])]
        #self.formula = '$\frac{x_i x_j  y_i y_j)}{(1 - y_iy_j + x_i x_j y_i y_j)(1 - y_i y_j)}$' TODO
        self.N = kwargs['N']
        self.bounds = [(EPS ,None)] * (2*self.N)
        

    def expected_adjacency(self, *args):
        x,y = args[0:self.N], args[(self.N):]
        xixj = np.outer(x,x)
        yiyj = np.outer(y,y)
        pij = np.nan_to_num((xixj * yiyj) / ((1 - yiyj + xixj * yiyj)))
        pij[pij<0] = EPS
        return pij

    def expected_weighted_adjacency(self, *args):
        y = args[(self.N):]
        pij = self.expected_adjacency(*args)
        yiyj = np.outer(y,y)
        return pij / (1.0-yiyj)

    def loglikelihood(self, G, *args):
        x,y = args[0:self.N], args[(self.N):]
        k = (G>0).sum(axis=0)
        s = G.sum(axis=0)
        xixj = np.outer(x,x)
        yiyj = np.outer(y,y)
        loglike = (x*k).sum() + (y*s).sum() + np.triu(np.log( (1-yiyj)/(1-yiyj + xixj*yiyj) ) ,1).sum()
        return loglike



#################### Continuous models #######################

class CWTECM(GraphModel):
    """
    1. Model name: Continuous Weighted Thresholded Enhanced Configuration Model

    2. Constraints: degree sequence k_i^*, strenght sequence s_i and threshold hyperparameter t
    
    3. Hamiltonian:
        H(W) = sum_{i<j} (alpha_i+alpha_j) Heaviside(w_{ij}-t) + (beta_i+beta_j) w_{ij} Heaviside(w_{ij}-t)
    
    4. Lagrange multipliers substitutions:
        x_i = exp(-alpha_i)
        y_i = exp(-beta_i)
    
    5. Number of parameters:
        2N
    
    6. Link probability <aij>
        

    7. Expected link weight <wij>
        

    8. LogLikelihood logP:
        sum_{i<j} a_{ij}*log(pij) + (1-aij)*log(1-pij)

    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.args_mapping =   ['x_' + str(i) for i in range(0, kwargs['N'])] + ['y_' + str(i) for i in range(0, kwargs['N'])]
        #self.formula = '$\frac{x_i x_j (y_i y_j)^t}{t \log(yi y_j) + x_i x_j (y_i y_j)^t}$'
        self.bounds = [(EPS, None) for i in range(0, 2*self.N ) ]
        self.N = kwargs['N']
        self.threshold = kwargs['threshold']

    def expected_adjacency(self, *args):
        x,y = args[0][0:self.N], args[0][(self.N):]
        t = self.threshold
        xixj = np.abs(np.outer(x,x)) # these variables are always > 0
        yiyj = np.abs(np.outer(y,y)) # these variables are always > 0
        yiyjt = yiyj**t
        pij = (xixj*yiyjt )/ (xixj*yiyjt - t*np.log(yiyj)) # <aij>
        return np.nan_to_num(pij)
    
    def expected_weighted_adjacency(self, *args):
        x,y = args[0][0:self.N], args[0][(self.N):]
        t = self.threshold
        xixj = np.abs(np.outer(x,x)) # these variables are always > 0
        yiyj = np.abs(np.outer(y,y)) # these variables are always > 0
        yiyjt = yiyj**t
        wij = ((t*np.log(yiyj)-1)/(t*np.log(yiyj)))
        wij *= ((xixj*yiyjt )/ (xixj*yiyjt - t*np.log(yiyj))) # <aij>
        return np.nan_to_num(wij)
    
    def saddle_point(self, G, *args):
        k = (G>0).sum(axis=0)
        pij = self.expected_adjacency(*args)
        avgk = pij.sum(axis=0) #- pij.diagonal()
        w = G.sum(axis=0)
        wij = self.expected_weighted_adjacency(*args)
        avgw = wij.sum(axis=0) #- wij.diagonal()
        return np.hstack([k-avgk,w-avgw])

    def loglikelihood(self, G, *args):
        x,y = args[0][0:self.N], args[0][(self.N):]
        t = self.threshold
        xixj = np.abs(np.outer(x,x)) # these variables are always > 0
        yiyj = np.abs(np.outer(y,y)) # these variables are always > 0
        yiyjt = yiyj**t
        A = (G>0).astype(float)
        loglike = np.log(-np.log(yiyj)) + A*np.log(xixj) + G*np.log(yiyj) - np.log(xixj*(yiyj**t) - t*np.log(yiyj))
        loglike = np.nan_to_num(loglike)
        return np.triu(loglike,1).sum()

class SpatialCM(GraphModel):
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
        #self.formula = '$\frac{z x_i x_j e^{-\gamma d_{ij}}}{ 1 + z x_i x_j e^{-\gamma d_{ij}}  }$'
        self.bounds = [(EPS,None) for i in range(0,kwargs['N'])] + [(EPS,None),(EPS,None)]
        self.dij = kwargs['dij']
        self.expdij = np.exp(-kwargs['dij'])
        self.is_weighted = kwargs.get('is_weighted',False)
    
    def expected_adjacency(self, *args):        
        O = args[-1]*np.outer(args[0:-2],args[0:-2]) * self.expdij**(args[-2])
        if self.is_weighted:
            return O / (1 + O)
        return O / (1-O)
