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

G here is the graph adjacency matrix, A is the binary adjacency, W is the weighted adjacency
"""

import autograd.numpy as np
from .GraphModel import GraphModel
from .GraphModel import expit, multiexpit, batched_symmetric_random
from networkqit.algorithms import MLEOptimizer
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

    def expected_adjacency(self, theta):
        xixj = np.outer(theta, theta)
        return xixj / (1.0 + xixj)

    def loglikelihood(self, observed_adj, theta):
        pij = self.expected_adjacency(theta)
        loglike = np.sum(np.triu(observed_adj * np.log(pij) + (1.0 - observed_adj) * np.log(1.0 - pij),1))
        return loglike
    
    def saddle_point(self, observed_adj, theta):
        k = (observed_adj>0).sum(axis=0)
        pij = self.expected_adjacency(theta)
        avgk = pij.sum(axis=0) - np.diag(pij)
        return k - avgk

    def sample_adjacency(self, theta, batch_size=1, with_grads=False, slope=500):
        xixj = np.outer(theta, theta)
        pij = xixj / (1.0 + xixj)
        rij = batched_symmetric_random(batch_size, self.N)
        if with_grads:
            A = expit(slope*(pij-rij)) # sampling, approximates binomial with continuos
        else:
            A = (pij>rij).astype(float)
        A = np.triu(A, 1) # make it symmetric
        A += np.transpose(A, axes=[0, 2, 1])
        return A

    def fit(self, G, x0=None, **opt_kwargs):
        from networkqit import MLEOptimizer
        A = (G>0).astype(float)
        k = A.sum(axis=0)
        if x0 is None:
            x0 = k / k.max()
        opt = MLEOptimizer(G, x0=x0, model=self)
        sol = opt.run(**opt_kwargs)
        return sol


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

    def expected_adjacency(self, theta):
        pij = np.outer(theta, theta)
        return pij
    
    def expected_weighted_adjacency(self, theta):
        pij = self.expected_adjacency(theta)
        wij = pij / (1.0 - pij)
        return wij
    
    def loglikelihood(self, observed_adj, theta):
        pij = self.expected_adjacency(theta)
        loglike = observed_adj * np.log(pij) + np.log(1.0 - pij)
        return np.sum(np.triu(loglike,1))

    def saddle_point(self, observed_adj, theta):
        s = observed_adj.sum(axis=0)
        wij = self.expected_weighted_adjacency(theta)
        avgs = wij.sum(axis=0) - np.diag(wij)
        return s - avgs

    def sample_adjacency(self, theta, batch_size=1, with_grads=False, slope=500):
        """
        Sample the adjacency matrix of the UWCM
        """
        pij = self.expected_adjacency(theta)
        rij = batched_symmetric_random(batch_size, self.N)
        if with_grads:
            qij = np.log(rij+EPS)/np.log(np.abs(1.0 - pij))
            W = multiexpit(qij)
        else: # differently from matlab, numpy generates random geometric in [1,infinity]
            W = np.random.geometric(1-pij,size=[batch_size,self.N,self.N]) - 1
        W = np.triu(W, 1) # make it symmetric
        W += np.transpose(W, axes=[0, 2, 1])
        return W

    def fit(self, G, x0=None, **opt_kwargs):
        from networkqit import MLEOptimizer
        A = (G>0).astype(float)
        k = A.sum(axis=0)
        if x0 is None:
            x0 = k / k.max()
        opt = MLEOptimizer(G, x0=x0, model=self)
        sol = opt.run(**opt_kwargs)
        return sol

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
        self.args_mapping = ['x_' + str(i) for i in range(0, kwargs['N'])] + ['y_' + str(i) for i in range(0, kwargs['N'])]
        #self.formula = '$\frac{x_i x_j  y_i y_j)}{(1 - y_iy_j + x_i x_j y_i y_j)(1 - y_i y_j)}$' TODO
        self.N = kwargs['N']
        self.bounds = [(EPS ,None)] * self.N + [(EPS, 1-EPS)] * self.N

    def expected_adjacency(self, theta):
        x,y = theta[0:self.N], theta[(self.N):]
        xixj = np.outer(x,x)
        yiyj = np.outer(y,y)
        pij = (xixj * yiyj) / ((1 - yiyj + xixj * yiyj))
        return pij

    def expected_weighted_adjacency(self, theta):
        x,y = theta[0:self.N], theta[(self.N):]
        pij = self.expected_adjacency(theta)
        yiyj = np.outer(y,y)
        return pij / (1.0 - yiyj)

    def loglikelihood(self, observed_adj, theta):
        x,y = theta[0:self.N], theta[(self.N):]
        k = (observed_adj > 0).sum(axis=0)
        s = observed_adj.sum(axis=0)
        xixj = np.outer(x,x)
        yiyj = np.outer(y,y)
        loglike = (k*np.log(x)).sum() + (s*np.log(y)).sum() + np.triu(np.log( (1-yiyj)/(1-yiyj + xixj*yiyj) ) ,1).sum()
        return loglike

    def saddle_point(self, observed_adj, theta): # equations (9,10) of 10.1088/1367-2630/16/4/043022
        k = (observed_adj>0).sum(axis=0)
        pij = self.expected_adjacency(theta) 
        avgk = pij.sum(axis=0) - np.diag(pij)
        w = observed_adj.sum(axis=0)
        wij = self.expected_weighted_adjacency(theta)
        avgw = wij.sum(axis=0) - np.diag(wij)
        return np.hstack([k - avgk, w - avgw])

    def sample_adjacency(self, theta, batch_size=1, with_grads=False, slope=500):
        rij = batched_symmetric_random(batch_size, self.N)
        x,y = theta[0:self.N], theta[(self.N):]
        xixj = np.outer(x,x)
        yiyj = np.outer(y,y)
        pij = xixj*yiyj/(1-yiyj + xixj*yiyj)
        if with_grads:
            # broadcasting pij and rij
            A = expit(slope*(pij-rij)) # sampling, approximates binomial with continuos
            # then must extract from a geometric distribution with probability P
            # https://math.stackexchange.com/questions/580901/r-generate-sample-that-follows-a-geometric-distribution
            q = np.log(rij+EPS)/np.log(np.abs(1.0-pij))
            W = multiexpit(slope*(q-1.0)) # continuous approximation to floor(x)
        else:
            A = pij>rij
            W = np.random.geometric(1-yiyj,size=[batch_size,self.N,self.N])
        W = np.triu(A*W,1)
        W += np.transpose(W, axes=[0, 2, 1])
        return W

    def fit(self, G, x0=None, **opt_kwargs):
        A = (G>0).astype(float)
        k = A.sum(axis=0)
        s = G.sum(axis=0)
        if x0 is None:
            x0 = np.concatenate([k,s])
            x0 = np.clip(x0/x0.max(), np.finfo(float).eps, 1-np.finfo(float).eps )
        opt = MLEOptimizer(G, x0=x0, model=self)
        sol = opt.run(**opt_kwargs)
        return sol


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
        (xixj*(yiyj**t) )/ (xixj*(yiyj**t) - t*log(yiyj))
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
        self.N = kwargs['N']
        self.threshold = kwargs['threshold']
        self.bounds = [(EPS, None)] * self.N + [(EPS, 1-EPS)] * self.N

    def expected_adjacency(self, theta):
        x,y = theta[0:self.N], theta[(self.N):]
        t = self.threshold
        xixj = np.outer(x,x)
        yiyj = np.outer(y,y)
        yiyjt = yiyj**t
        pij = (xixj*yiyjt) / (xixj*yiyjt - t*np.log(yiyj)) # <aij>
        return pij
    
    def expected_weighted_adjacency(self, theta):
        x,y = theta[0:self.N], theta[(self.N):]
        t = self.threshold
        xixj = np.outer(x,x)
        yiyj = np.outer(y,y)
        yiyjt = yiyj**t
        wij = self.expected_adjacency(theta) * ((t*np.log(yiyj)-1.0) / (np.log(yiyj)))
        return wij
    
    def saddle_point(self, observed_adj, theta):
        k = (observed_adj>0).sum(axis=0)
        pij = self.expected_adjacency(theta)
        avgk = pij.sum(axis=0) - np.diag(pij)
        w = observed_adj.sum(axis=0)
        wij = self.expected_weighted_adjacency(theta)
        avgw = wij.sum(axis=0) - np.diag(wij)
        return np.hstack([k - avgk, w - avgw])

    # def saddle_point_jac(self, observed_adj, theta):
    #     x,y = theta[0:self.N], theta[(self.N):]
    #     t = self.threshold
    #     xixj = np.outer(x,x)
    #     yiyj = np.outer(y,y)
    #     dpijdxixj = t*(xixj*(yiyj**t))/(yiyj**t*(-xixj)+2*t*np.log(yiyj)) * np.log(yiyj)
    #     dpijdyiyj = t*xixj*(yiyj**t)*(t*np.log(yiyj) -1)/(xixj*(yiyj**t) - 2*t*np.log(yiyj))
    
    def loglikelihood(self, observed_adj, theta):
        x,y = theta[0:self.N], theta[(self.N):]
        t = self.threshold
        xixj = np.outer(x,x)
        yiyj = np.outer(y,y)
        if not hasattr(self, '_k'):
            self._k =  (observed_adj>0).astype(float).sum(axis=0)
            self._s = observed_adj.sum(axis=0)
        loglike = (self._s*np.log(y) + self._k*np.log(x)).sum() + np.triu(np.log(-np.log(yiyj)/(xixj*(yiyj**t) -t*(np.log(yiyj)) ) ),1).sum()
        return loglike

    def sample_adjacency(self, theta, batch_size=1, with_grads=False, slope=500):
        rij = batched_symmetric_random(batch_size, self.N)
        pij = self.expected_adjacency(theta)
        wij = self.expected_weighted_adjacency(theta)
        if with_grads:
            A = expit(slope*(pij-rij)) # it needs a gigantic slope to reduce error
            # to generate random weights, needs a second decorrelated random source
            rij = batched_symmetric_random(batch_size, self.N)
            W = -wij*np.log(rij)/pij
        else:
            A = (pij>rij).astype(float)
            W = np.random.exponential(wij/pij,size=[batch_size,self.N,self.N])
        W = np.triu(A*W,1)
        W +=  np.transpose(W, axes=[0, 2, 1])
        return W

    def fit(self, G, x0=None, **opt_kwargs):
        from networkqit import MLEOptimizer
        A = (G>self.threshold).astype(float)
        k = A.sum(axis=0)
        s = G.sum(axis=0)
        if x0 is None:
            x0 = np.concatenate([k,s])
            x0 = np.clip(x0/x0.max(), np.finfo(float).eps, 1-np.finfo(float).eps )
        opt = MLEOptimizer(G, x0=x0, model=self)
        sol = opt.run(**opt_kwargs)
        return sol
