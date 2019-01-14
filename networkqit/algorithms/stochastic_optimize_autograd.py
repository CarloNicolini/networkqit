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

Define the base and inherited classes for model optimization, both in the continuous approximation
and for the stochastic optimization.

The `ModelOptimizer` class defines the base class where all the other optimization classes must inherit.
The most important class to optimize an expected adjacency model is `ExpectedModelOptimizer`. In this class the gradients are defined as:

.. math::
    
        \\frac{\\partial S(\\rho \\| \\sigma(\\mathbb{E}_{\\theta}[L]))}{\\partial \\theta_k} = \\beta \\textrm{Tr}\\biggl \\lbrack \\left(\\rho - \\sigma(\\mathbb{E}_{\\theta}[L])\\right)\\frac{\\partial \\mathbb{E}_{\\theta}[L]}{\\partial \\theta_k} \\biggr \\rbrack

In the `StochasticOptimizer` class we instead address the issue to implement stochastic gradient descent methods.
In these methods the gradients are defined as:

.. math::
   
   \\frac{\\partial \\mathbb{E}_{\\theta}[S(\\rho \\| \\sigma)]}{\\partial \\theta_k} = \\beta \\textrm{Tr}\\biggl \\lbrack \\rho \\frac{\\partial  \\mathbb{E}_{\\theta}[L]}{\\partial \\theta_k}\\biggr \\rbrack + \\frac{\\partial}{\\partial \\theta_k}\\mathbb{E}_{\\theta}\\biggl \\lbrack \\log \\left( \\textrm{Tr}\\left \\lbrack e^{-\\beta L(\\theta)} \\right \\rbrack \\right) \\biggr \\rbrack

The stochastic optimizer is the **correct** optimizer, as it makes no approximation on the Laplacian eigenvalues.
It is more suitable for small graphs and intermediate $\\beta$, where the differences between the random matrix spectrum and its expected counterpart are non-neglibile.
For large and dense enough graphs however the `ExpectedModelOptimizer` works well and yields deterministic results, as the optimization landscape is smooth.

In order to minimize the expected relative entropy then, we need both the expected Laplacian formula, which is simple to get, and a way to estimate the second summand in the gradient, that involves averaging over different realizations of the log trace of  $e^{-\\beta L(\\theta)}$.
The better the approximation to the expected logtrace $\\mathbb{E}_{\\theta}[\\log \\textrm{Tr}[\\exp{(-\\beta L)}]]$ is, the better is the estimate of the gradients.


Finally, the `MLEOptimizer` maximizes the standard likelihood of a model and it is not related to the spectral entropies framework introduced in the paper on which **networkqit** is based.

"""
#    Copyright (C) 2018 by
#    Carlo Nicolini <carlo.nicolini@iit.it>
#    All rights reserved.
#    BSD license.

#from autograd import grad, elementwise_grad, jacobian

from networkqit.graphtheory import *
from networkqit.graphtheory import graph_laplacian as graph_laplacian
from networkqit.infotheory.density import VonNeumannDensity, SpectralDivergence, compute_vonneuman_density

from .optimize import ModelOptimizer

import autograd.numpy as np
import autograd
from autograd.numpy.linalg import eigh
from autograd.scipy.misc import logsumexp
from autograd.misc.optimizers import adam

################################################
## Stochastic optimzation from random samples ##
################################################
class StochasticOptimizer(ModelOptimizer):
    """
    This class is at the base of possible implementation of methods based
    on stochastic gradient descent.
    The idea behind this class is to help the user in designing a nice stochastic gradient descent method,
    such as ADAM, AdaGrad or older methods, like the Munro-Robbins stochastic gradients optimizer.
    Working out the expression for the gradients of the relative entropy, one remains with the following:

    :math: `\nabla_{\theta}S(\rho \| \sigma) = \beta \textrm\biggl \lbrack \rho \nabla_{\theta}\mathbb{E}_{\theta}[L]} \biggr \rbrack`
        
    :math: `\frac{\partial S(\rho \| \sigma)}{\partial \theta_k} = \beta \textrm{Tr}\lbrack \rho \frac{\partial}{\partial \theta_k} \rbrack + \frac{\partial}{\partial \theta_k}\mathbb{E}_{\theta}\log \textrm{Tr} e^{-\beta L(\theta)}\lbrack \rbrack`
    
    This class requires either Tensorflow or Pytorch to support backpropagation in the eigenvalues routines.
    Alternatively you can use github.com/HIPS/autograd method for full CPU support.
    """

    def __init__(self, A, x0, beta_range, **kwargs):
        super().__init()
        pass

    def setup(self, model, step_callback=None):
        """
        Setup the optimizer. Must specify the model.

        args:
            adj_fun: a function in the form f(x) that once called returns the adjacency matrix of a random graph. Not to be confused with the expected adjacency matrix.
            expected_laplacian_grad: a function in the form f(x) that once called returns the expected gradients of the laplacian of the random graph.
            step_callback: a callback function to control the current status of optimization.
        """
        self.expected_adj_fun = model.expected_adjacency
        self.sample_adjacency_fun = model.sample_adjacency
        self.expected_laplacian_grad_fun = model.expected_laplacian_grad
        self.step_callback = step_callback
        #self.bounds = expected_adj_fun.bounds

    # Need to rewrite this graph laplacian to use autograd implementation of numpy
    def graph_laplacian(self, A):
        return np.diag(np.sum(A, axis=0)) - A

    def gradient(self, x, rho, beta, num_samples=1):
        # Compute the relative entropy between rho and sigma, using tensors with autograd support
        def rel_entropy(z):
            #Lmodel = [self.graph_laplacian( self.sample_adjacency_fun(z) ) for i in range(num_samples)]
            Lmodel = self.graph_laplacian( self.sample_adjacency_fun(z) )
            Emodel = np.sum(np.multiply(Lmodel,rho)) / num_samples
            Eobs = np.sum(np.multiply(self.L, rho)) / num_samples
            lambd_model = eigh(Lmodel)[0]
            lambd_obs = eigh(self.L)[0]
            Fmodel = - logsumexp(-beta*lambd_model) / beta
            Fobs = - logsumexp(-beta*lambd_obs) / beta
            loglike = beta*(Emodel-Fmodel)
            entropy = beta*(Eobs-Fobs)
            dkl = loglike - entropy
            return dkl

        fgradx = autograd.grad(rel_entropy) # gradient of relative entropy as a function
        return fgradx(x)

    def run(self, model, **kwargs):
        """
        Start the optimization process
        """
        pass


class StochasticGradientDescent(StochasticOptimizer):
    """
    Implements the ADAM stochastic gradient descent.
    """

    def __init__(self, A, x0, beta_range, **kwargs):
        self.A = A
        self.L = graph_laplacian(self.A)
        self.x0 = x0
        self.beta_range = beta_range

    def run(self, **kwargs):
        x = self.x0
        num_samples = kwargs.get('num_samples', 1)
        clip_gradients = kwargs.get('clip_gradients', None)
        max_iters = kwargs.get('max_iters', 1000)
        eta = kwargs.get('eta', 1E-3)
        gtol = kwargs.get('gtol', 1E-5)
        xtol = kwargs.get('xtol', 1E-3)
        # Populate the solution list as function of beta
        # the list sol contains all optimization points
        sol = []
        # Iterate over all beta provided by the user

        for beta in self.beta_range:
            x = self.x0
            rho = VonNeumannDensity(A=None, L=self.L, beta=beta).density
            converged = False
            opt_message = ''
            t = 0
            while not converged:
                t += 1
                grad_t =  self.gradient(x, rho, beta)
                # Convergence status
                if np.linalg.norm(grad_t) < gtol:
                    converged, opt_message = True, 'gradient tolerance exceeded'
                if t > max_iters:
                    converged, opt_message = True, 'max_iters_exceed'
                # if x[0]<0:
                #    converged, opt_message = True, 'bounds_exceeded'
                x_old = x.copy()
                x -= eta * grad_t
                print('\rbeta=', beta, '|grad|=', np.linalg.norm(grad_t), 'x=', np.linalg.norm(x), ' m=', self.expected_adj_fun(x).sum() / 2, end='')
                if self.step_callback is not None:
                    self.step_callback(beta, x)

            sol.append({'x': x})
        self.sol = sol
        return sol

