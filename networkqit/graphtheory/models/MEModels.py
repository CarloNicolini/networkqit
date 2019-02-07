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
from .GraphModel import expit, batched_symmetric_random, multiexpit

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
        xixj = np.outer(*args, *args)
        return xixj / (1.0 + xixj)

    def loglikelihood(self, observed_adj, *args):
        pij = self.expected_adjacency(*args)
        loglike = np.sum(np.triu(observed_adj * np.log(pij) + (1.0 - observed_adj) * np.log(1.0 - pij),1))
        return loglike
    
    def saddle_point(self, G, *args):
        k = (G>0).sum(axis=0)
        pij = self.expected_adjacency(*args)
        avgk = pij.sum(axis=0) - np.diag(pij)
        return k - avgk

    def sample_adjacency(self, *args, **kwargs):
        batch_size = kwargs.get('batch_size', 1)
        xixj = np.outer(*args, *args)
        pij = xixj / (1.0 + xixj)
        rij = batched_symmetric_random(batch_size, self.N)
        if kwargs.get('with_grads',False):
            slope = kwargs.get('slope', 50.0)
            A = expit(slope*(pij-rij)) # sampling, approximates binomial with continuos
        else:
            A = (pij>rij).astype(float)
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
    
    9. Parameters bounds
        0 < yij < 1

    Reference:
        Squartini thesis page 110
    """
    
    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = ['y_' + str(i) for i in range(0, kwargs['N'])]
        self.model_type = 'topological'
        self.formula = '$<w_{ij}> = \frac{y_i y_j}{1-y_i y_j}$'
        self.bounds = [(EPS, 1.0-EPS) for i in range(0, self.N)]

    def expected_adjacency(self, *args):
        pij = np.outer(*args, *args)
        return pij
    
    def expected_weighted_adjacency(self, *args):
        pij = self.expected_adjacency(*args)
        wij = pij / (1.0 - pij)
        return wij
    
    def loglikelihood(self, observed_adj, *args):
        pij = self.expected_adjacency(*args)
        loglike = observed_adj * np.log(pij) + np.log(1.0 - pij)
        return np.sum(np.triu(loglike,1))

    def saddle_point(self, G, *args):
        s = G.sum(axis=0)
        wij = self.expected_weighted_adjacency(*args)
        return s - (wij.sum(axis=0) - np.diag(wij))

    def sample_adjacency(self, *args, **kwargs):
        """
        Sample the adjacency matrix of the UWCM
        """
        batch_size = kwargs.get('batch_size', 1)
        pij = self.expected_adjacency(*args)
        rij = batched_symmetric_random(batch_size, self.N)
        if kwargs.get('with_grads',False):
            slope = kwargs.get('slope', 50.0)
            qij = np.log(rij+EPS)/np.log(np.abs(1.0 - pij))
            W = multiexpit(qij)
        else:
            W = np.random.geometric(1-pij,size=[batch_size,self.N,self.N])
        W = np.triu(W, 1) # make it symmetric
        W += np.transpose(W, axes=[0, 2, 1])
        return W


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

    9. Parameters bounds
        x > 0
        0 < y < 1
        
    Reference: Squartini thesis page 122
    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.args_mapping =   ['x','y']
        # TODO self.formula = '$\frac{x  y_i y_j)}{(1 - y_iy_j + x y_i y_j)(1 - y_i y_j)}$'
        self.N = kwargs['N']
        self.bounds = [(EPS, None), (EPS, 1-EPS)] # bounds of the variables
        
    def expected_adjacency(self, *args):
        x,y = args[0], args[1]
        pij = (x * y) / ((1 - y + x * y))
        pij[pij<0] = EPS
        return pij

    def expected_weighted_adjacency(self, *args):
        y = args[1]
        pij = self.expected_adjacency(*args)
        return pij / (1 - y)

    def loglikelihood(self, observed_adj, *args):
        x,y = args[0], args[1]
        L = (np.triu(observed_adj, 1) > 0).sum()
        Wtot = (np.triu(observed_adj, 1)).sum()
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

    9. Parameters bounds
        very complicated
        1. xi > 0
        2. yi > 0
        3. yi < 1
        4. 0 < xi yi < 1

    Reference:
        Squartini thesis page 126
    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.args_mapping =   ['x_' + str(i) for i in range(0, kwargs['N'])] + ['y_' + str(i) for i in range(0, kwargs['N'])]
        #self.formula = '$\frac{x_i x_j  y_i y_j)}{(1 - y_iy_j + x_i x_j y_i y_j)(1 - y_i y_j)}$' TODO
        self.N = kwargs['N']
        self.bounds = [(EPS ,None)] * self.N + [(EPS, 1-EPS)] * self.N

    def expected_adjacency(self, *args):
        x,y = args[0][0:self.N], args[0][(self.N):]
        xixj = np.outer(x,x)
        yiyj = np.outer(y,y)
        pij = (xixj * yiyj) / ((1 - yiyj + xixj * yiyj))
        return pij

    def expected_weighted_adjacency(self, *args):
        x,y = args[0][0:self.N], args[0][(self.N):]
        pij = self.expected_adjacency(*args)
        yiyj = np.outer(y,y)
        return pij / (1.0-yiyj)

    def loglikelihood(self, observed_adj, *args):
        x,y = args[0][0:self.N], args[0][(self.N):]
        k = (observed_adj > 0).sum(axis=0)
        s = observed_adj.sum(axis=0)
        xixj = np.outer(x,x)
        yiyj = np.outer(y,y)
        loglike = (k*np.log(x)).sum() + (s*np.log(y)).sum() + np.triu(np.log( (1-yiyj)/(1-yiyj + xixj*yiyj) ) ,1).sum()
        return loglike

    def saddle_point(self, G, *args):
        k = (G>self.threshold).sum(axis=0)
        pij = self.expected_adjacency(*args)
        avgk = pij.sum(axis=0) - np.diag(pij)
        w = G.sum(axis=0)
        wij = self.expected_weighted_adjacency(*args)
        avgw = wij.sum(axis=0) - np.diag(wij)
        return np.hstack([k-avgk,w-avgw])


    def sample_adjacency(self, *args, **kwargs):
        """
        Sample the adjacency matrix of the UECM
        """
        batch_size = kwargs.get('batch_size', 1)
        slope = kwargs.get('slope', 50.0)
        rij = batched_symmetric_random(batch_size, self.N)
        x,y = args[0][0:self.N], args[0][(self.N):]
        xixj = np.outer(x,x)
        yiyj = np.outer(y,y)
        pij = xixj*yiyj/(1-yiyj + xixj*yiyj)
        if kwargs.get('with_grads',False):
            # broadcasting pij and rij
            A = expit(slope*(pij-rij)) # sampling, approximates binomial with continuos
            A = np.triu(A, 1) # make it symmetric
            A += np.transpose(A, axes=[0, 2, 1])
            # then must extract from a geometric distribution with probability P
            # https://math.stackexchange.com/questions/580901/r-generate-sample-that-follows-a-geometric-distribution
            q = np.log(rij+EPS)/np.log(np.abs(1.0-pij))
            W = multiexpit(slope*(q-1.0)) # continuous approximation to floor(x)
        else:
            # Questa è la soluzione corretta
            A = np.triu(pij>rij,1)
            A += np.transpose(A, axes=[0, 2, 1])
            W = np.triu(np.random.geometric(1-yiyj,size=[batch_size,self.N,self.N]),1)
            W += np.transpose(W, axes=[0, 2, 1])
        return W*A

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
    
    6. Link probability pij=<aij>
        (xixj*yiyjt )/ (xixj*yiyjt - t*log(yiyj))
        
    7. Expected link weight <wij>
        wij = (t*log(yiyj)-1)/(t*log(yiyj))* <aij>

    8. LogLikelihood logP:
        sum_{i<j}  xij^{Aij} yij^{Aij wij} - log(t - (xij yij^t/ log (yij)) )

    9. Constraints:
        1. xi > 0
        2. 0 < yi < 1
        3  0 < xi yi < 1

    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.args_mapping =   ['x_' + str(i) for i in range(0, kwargs['N'])] + ['y_' + str(i) for i in range(0, kwargs['N'])]
        #self.formula = '$\frac{x_i x_j (y_i y_j)^t}{t \log(yi y_j) + x_i x_j (y_i y_j)^t}$'
        self.N = kwargs['N']
        self.threshold = kwargs['threshold']
        self.bounds = [(EPS, None)] * self.N + [(EPS, 1-EPS)] * self.N

    def expected_adjacency(self, *args):
        x,y = args[0][0:self.N], args[0][(self.N):]
        t = self.threshold
        xixj = np.abs(np.outer(x,x)) # these variables are always > 0
        yiyj = np.abs(np.outer(y,y)) # these variables are always > 0
        yiyjt = yiyj**t
        pij = (xixj*yiyjt) / (xixj*yiyjt - t*np.log(yiyj)) # <aij>
        return pij
    
    def expected_weighted_adjacency(self, *args):
        x,y = args[0][0:self.N], args[0][(self.N):]
        t = self.threshold
        xixj = np.outer(x,x) # these variables are always > 0
        yiyj = np.outer(y,y) # these variables are always > 0
        yiyjt = yiyj**t
        wij = self.expected_adjacency(*args) * ((t*np.log(yiyj)-1.0) / (np.log(yiyj)))
        return wij
    
    def saddle_point(self, G, *args):
        k = (G>0).sum(axis=0)
        pij = self.expected_adjacency(*args)
        avgk = pij.sum(axis=0) - np.diag(pij)
        w = G.sum(axis=0)
        wij = self.expected_weighted_adjacency(*args)
        avgw = wij.sum(axis=0) - np.diag(wij)
        return np.hstack([k-avgk,w-avgw])
    
    def loglikelihood(self, wij, *args):
        t = self.threshold
        x,y = args[0][0:self.N], args[0][(self.N):]
        xixj = np.outer(x,x)
        yiyj = np.outer(y,y)
        if not hasattr(self, '_k'):
            self._k =  (wij>t).astype(float).sum(axis=0)
            self._s = wij.sum(axis=0)
        loglike = (self._s*np.log(y) + self._k*np.log(x)).sum() + np.triu(np.log(-np.log(yiyj)/(xixj*(yiyj**t) -t*(np.log(yiyj)) ) ),1).sum()
        return loglike

    def sample_adjacency(self, *args, **kwargs):
        """
        Sample the adjacency matrix of the CWTECM
        """
        batch_size = kwargs.get('batch_size', 1)
        slope = kwargs.get('slope', 500000.0)
        rij = batched_symmetric_random(batch_size, self.N)
        pij = self.expected_adjacency(*args)
        wij = self.expected_weighted_adjacency(*args)
        requires_grads = kwargs.get('with_grads',False)
        if requires_grads:
            A = expit(slope*(pij-rij)) # it needs a gigantic slope to reduce error
            rij = batched_symmetric_random(batch_size, self.N)
            rij[rij<EPS]=EPS
            A = np.triu(A, 1)
            A += np.transpose(A, axes=[0, 2, 1])
            W = np.triu(-np.log(rij)/(pij/wij), 1)
            W +=  np.transpose(W, axes=[0, 2, 1])
        else:
            A = (pij>rij).astype(float)
            A = np.triu(A, 1) # make it symmetric
            A += np.transpose(A, axes=[0, 2, 1])
            W = np.triu(np.random.exponential(wij/pij,size=[batch_size,self.N,self.N]),1)
            W +=  np.transpose(W, axes=[0, 2, 1])
        return A*W

