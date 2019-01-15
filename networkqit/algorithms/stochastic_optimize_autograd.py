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
        self.bounds = model.bounds

    # Need to rewrite this graph laplacian to use autograd implementation of numpy
    def graph_laplacian(self, A):
        return np.diag(np.sum(A, axis=0)) - A

    def gradient(self, x, rho, beta, batch_size=2):
        # Compute the relative entropy between rho and sigma, using tensors with autograd support
        # TODO: Fix because now it has no multiple samples support
        def rel_entropy(z):
            Lmodel = self.graph_laplacian(self.sample_adjacency_fun(z))
            print(Lmodel.dtype)
            Emodel = np.sum(np.multiply(Lmodel, rho)) / batch_size
            Eobs = np.sum(np.multiply(self.L, rho)) / batch_size
            lambd_model = eigh(Lmodel)[0]
            lambd_obs = eigh(self.L)[0]
            Fmodel = - logsumexp(-beta*lambd_model) / beta
            Fobs = - logsumexp(-beta*lambd_obs) / beta
            loglike = beta*(Emodel-Fmodel)
            entropy = beta*(Eobs-Fobs)
            dkl = loglike - entropy
            return dkl

        def rel_entropy2(z):
            Lmodel = np.array([ self.graph_laplacian( self.sample_adjacency_fun(z) )  for i in range(batch_size)])
            Emodel = np.mean(np.sum(np.sum(np.multiply(Lmodel,rho),axis=2),axis=1))
            Eobs = np.sum(np.multiply(self.L, rho))
            lambd_obs = eigh(self.L)[0]
            lambd_model = np.array([eigh(Lmodel[i])[0] for i in range(batch_size)])
            Fmodel = - np.mean(logsumexp(-beta*lambd_model, axis=1) / beta)
            Fobs = - logsumexp(-beta*lambd_obs) / beta
            loglike = beta*(Emodel-Fmodel)
            entropy = beta*(Eobs-Fobs)
            dkl = loglike - entropy
            return dkl

        def rel_entropy_batched(z):
            N = len(z) # number of free variables
            # advanced broadcasting here!
            Amodel = self.sample_adjacency_fun(z, batch_size=batch_size) # a batched number of adjacency matrices [batch_size,N,N]
            Dmodel = np.eye(N) * np.transpose(np.zeros([1,1,N]) + np.einsum('ijk->ik',Amodel),[1,0,2])
            Lmodel =  Dmodel - self.sample_adjacency_fun(z) # returns a batch_size x N x N tensor
            # do average over batches of the sum of product of matrix elements (done with * operator)
            Emodel = np.mean(np.sum(np.sum(Lmodel*rho,axis=2),axis=1))
            Eobs = np.sum(self.L*rho) # = Tr[rho Lobs]
            lambd_obs = eigh(self.L)[0] # eigh is a batched operation
            lambd_model = eigh(Lmodel)[0] # eigh is a batched operation, 
            Fmodel = - np.mean(logsumexp(-beta*lambd_model, axis=1) / beta)
            Fobs = - logsumexp(-beta*lambd_obs) / beta
            loglike = beta*(Emodel-Fmodel)
            entropy = beta*(Eobs-Fobs)
            dkl = loglike - entropy
            return dkl

        f_and_df = autograd.value_and_grad(rel_entropy_batched) # gradient of relative entropy as a function
        f_and_df_vals = f_and_df(x)
        return f_and_df_vals[0], f_and_df_vals[1]

    def run(self, model, **kwargs):
        """
        Start the optimization process
        """
        raise NotImplementedError


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
        #batch_size = kwargs.get('batch_size', 1)
        clip_gradients = kwargs.get('clip_gradients', None)
        max_iters = kwargs.get('max_iters', 1000)
        eta = kwargs.get('eta', 1E-3)
        gtol = kwargs.get('gtol', 1E-6)
        xtol = kwargs.get('xtol', 1E-3)
        # Populate the solution list as function of beta
        # the list sol contains all optimization points
        sol = []
        # Iterate over all beta provided by the user
        batch_size = kwargs.get('batch_size', 1)
        for beta in self.beta_range:
            x = self.x0
            rho = VonNeumannDensity(A=None, L=self.L, beta=beta).density
            converged = False
            opt_message = ''
            t = 0
            while not converged:
                t += 1
                dkl, grad_t =  self.gradient(x, rho, beta, batch_size=batch_size)
                # Convergence status
                if np.linalg.norm(grad_t) < gtol:
                    converged, opt_message = True, 'gradient tolerance exceeded'
                if t > max_iters:
                    converged, opt_message = True, 'max_iters_exceed'
                if np.any(x[0]<0):
                    raise RuntimeError('bounds_exceeded')
                x_old = x.copy()
                x -= eta * grad_t
                #print('x=', x, '|grad|=', np.linalg.norm(grad_t), ' m=', self.expected_adj_fun(x).sum() / 2, 'beta=', beta)
                print(' m=', self.expected_adj_fun(x).sum() / 2)
                if self.step_callback is not None:
                    self.step_callback(beta, x)
            print(opt_message)
            sol.append({'x': x})
        self.sol = sol
        return sol

class Adam(StochasticOptimizer):
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
        #batch_size = kwargs.get('batch_size', 1)
        clip_gradients = kwargs.get('clip_gradients', None)
        max_iters = kwargs.get('max_iters', 1000)
        eta = kwargs.get('eta', 1E-3)
        gtol = kwargs.get('gtol', 1E-6)
        xtol = kwargs.get('xtol', 1E-3)

        alpha = kwargs.get('alpha', 1E-3)
        gtol = kwargs.get('gtol', 1E-3)
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1E-8

        from drawnow import drawnow, figure
        import matplotlib.pyplot as plt
        # if global namespace, import plt.figure before drawnow.figure
        figure(figsize=(8, 4))

        # Populate the solution list as function of beta
        # the list sol contains all optimization points
        sol = []
        # Iterate over all beta provided by the user
        mt, vt = np.zeros(self.x0.shape), np.zeros(self.x0.shape)
        all_x = []
        all_dkl = []
        batch_size = kwargs.get('batch_size', 1)
        for beta in self.beta_range:
            x = self.x0
            rho = VonNeumannDensity(A=None, L=self.L, beta=beta).density
            converged = False
            opt_message = ''
            t = 0
            while not converged:
                t += 1
                dkl, grad_t =  self.gradient(x, rho, beta, batch_size=batch_size)
                # Convergence status
                if np.linalg.norm(grad_t) < gtol:
                    converged, opt_message = True, 'gradient tolerance exceeded'
                if t > max_iters:
                    converged, opt_message = True, 'max_iters_exceed'
                if np.any(x[0]<0):
                    raise RuntimeError('bounds_exceeded')
                mt = beta1 * mt + (1.0 - beta1) * grad_t
                vt = beta2 * vt + (1.0 - beta2) * grad_t * grad_t
                mttilde = mt / (1.0 - (beta1 ** t))  # compute bias corrected first moment estimate
                vttilde = vt / (1.0 - (beta2 ** t))  # compute bias-corrected second raw moment estimate
                x_old = x.copy()
                x -= alpha * mttilde / np.sqrt(vttilde + epsilon)
                if t % 1000 == 0:
                    print('iter=',t, '|grad|=', np.linalg.norm(grad_t))#, ' m=', self.expected_adj_fun(x).sum() / 2, 'beta=', beta)
                #print(' m=', self.expected_adj_fun(x).sum() / 2)
                if self.step_callback is not None:
                    self.step_callback(beta, x)
                all_x.append(x[0])
                all_dkl.append(dkl)
                #sol[-1]['rel_entropy'] = dkl
                if t % 1000 == 0:
                    def draw_fig():
                        sol.append({'x': x.copy()})
                        # Here creates the output data structure as a dictionary of the optimization parameters and variables
                        # spect_div = SpectralDivergence(Lobs=self.L, Lmodel=graph_laplacian(self.sample_adjacency_fun(sol[-1]['x'])), beta=beta)
                        # sol[-1]['DeltaL'] = (np.trace(self.L) - np.trace(graph_laplacian(self.sample_adjacency_fun(sol[-1]['x'])))) / 2
                        # sol[-1]['T'] = 1.0 / beta
                        # sol[-1]['beta'] = beta
                        # sol[-1]['loglike'] = spect_div.loglike
                        # sol[-1]['rel_entropy'] = spect_div.rel_entropy
                        # sol[-1]['entropy'] = spect_div.entropy
                        plt.subplot(1,3,1)
                        plt.imshow(self.A)
                        plt.subplot(1,3,2)
                        plt.imshow(self.sample_adjacency_fun(x))
                        plt.subplot(1,3,3)
                        plt.plot(all_dkl)
                        plt.tight_layout()
                    drawnow(draw_fig)
        self.sol = sol
        return sol

