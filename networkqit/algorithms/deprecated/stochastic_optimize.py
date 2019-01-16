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
    
    This class requires either Tensorflow or Pytorch to support backpropagation in the eigenvalues routines
    """

    def __init__(self, A, x0, beta_range, **kwargs):
        pass

    def setup(self, expected_adj_fun, adj_sampling_fun, expected_laplacian_grad=None, step_callback=None):
        """
        Setup the optimizer. Must specify the model function, as a function that returns the expected adjacency matrix,
        Differently from the expected model optimizer here we need not only the expected Laplacian, but also the expectation
        of the log(Tr(exp(-betaL))). This can only be compu

        args:
            adj_fun: a function in the form f(x) that once called returns the adjacency matrix of a random graph. Not to be confused with the expected adjacency matrix.
            expected_laplacian_grad: a function in the form f(x) that once called returns the expected gradients of the laplacian of the random graph.
            step_callback: a callback function to control the current status of optimization.
        """
        self.modelfun = expected_adj_fun
        self.samplingfun = adj_sampling_fun
        self.modelfun_grad = expected_laplacian_grad
        self.step_callback = step_callback
        self.bounds = expected_adj_fun.bounds

    def gradient(self, x, rho, beta, num_samples=1):
        # Compute the first part of the gradient, the one depending linearly on the expected laplacian, easy to get
        grad = np.array(
            [beta * np.sum(np.multiply(rho, self.modelfun_grad(x)[:, i, :])) for i in range(0, len(self.x0))])

        # Now compute the second part, dependent on the gradient of the expected log partition function
        def quenched_log_partition_gradient(x, rho, beta, num_samples=1):  # quenched gradient estimation
            logZ = lambda y: logsumexp(-beta * eigvalsh(graph_laplacian(self.samplingfun(y))))
            meanlogZ = lambda w: np.mean([logZ(w) for i in range(0, num_samples)])
            return nd.Gradient(meanlogZ)(x)

        def annealed_log_partition_gradient(x, rho, beta, num_samples=1):  # annealed gradient estimation
            logZ = lambda y: logsumexp(-beta * eigvalsh(graph_laplacian(self.samplingfun(y))))
            meanlogZ = lambda w: np.mean([logZ(w) for i in range(0, num_samples)])
            return nd.Gradient(meanlogZ)(x)

        def quenched(x, rho, beta, num_samples=1):
            g = np.zeros_like(x)
            for r in range(0, num_samples):
                lxplus = eigvalsh(graph_laplacian(self.samplingfun(x + 0.01)))
                lx = eigvalsh(graph_laplacian(self.samplingfun(x)))
                g += (logsumexp(lxplus) - logsumexp(lx)) * 100
            return g / num_samples

        gradlogtrace = annealed_log_partition_gradient(x, rho, beta, num_samples)
        return grad + gradlogtrace

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
            x = np.random.random(self.x0.shape)
            rho = VonNeumannDensity(A=None, L=self.L, beta=beta).density
            converged = False
            opt_message = ''
            t = 0
            while not converged:
                t += 1
                grad_t = None
                if clip_gradients is None:
                    grad_t = self.gradient(x, rho, beta)
                else:
                    grad_t = np.clip(self.gradient(x, rho, beta), clip_gradients[0],
                                     clip_gradients[1])  # clip the gradients
                # Convergence status
                if np.linalg.norm(grad_t) < gtol:
                    converged, opt_message = True, 'gradient tolerance exceeded'
                if t > max_iters:
                    converged, opt_message = True, 'max_iters_exceed'
                # if x[0]<0:
                #    converged, opt_message = True, 'bounds_exceeded'
                x_old = x.copy()
                x -= eta * grad_t
                print('\rbeta=', beta, '|grad|=', np.linalg.norm(grad_t), 'x=', np.linalg.norm(x), ' m=',
                      self.modelfun(x).sum() / 2, end='')
                if self.step_callback is not None:
                    self.step_callback(beta, x)

            sol.append({'x': x})
            # Here creates the output data structure as a dictionary of the optimization parameters and variables
            sol[-1]['opt_message'] = opt_message
            spect_div = SpectralDivergence(Lobs=self.L, Lmodel=graph_laplacian(self.samplingfun(sol[-1]['x'])),
                                           beta=beta)
            sol[-1]['DeltaL'] = (np.trace(self.L) - np.trace(graph_laplacian(self.samplingfun(sol[-1]['x'])))) / 2
            if kwargs.get('compute_sigma', False):
                Lmodel = graph_laplacian(self.modelfun(sol[-1]['x']))
                rho = VonNeumannDensity(A=None, L=self.L, beta=beta).density
                sigma = VonNeumannDensity(A=None, L=Lmodel, beta=beta).density
                sol[-1]['<DeltaL>'] = np.trace(np.dot(rho, Lmodel)) - np.trace(np.dot(sigma, Lmodel))
            sol[-1]['T'] = 1 / beta
            sol[-1]['beta'] = beta
            sol[-1]['loglike'] = spect_div.loglike
            sol[-1]['rel_entropy'] = spect_div.rel_entropy
            sol[-1]['entropy'] = spect_div.entropy
            sol[-1]['AIC'] = 2 * len(self.modelfun.args_mapping) - 2 * sol[-1]['loglike']
            # for i in range(0, len(self.modelfun.args_mapping)):
            #    sol[-1][self.modelfun.args_mapping[i]] = sol[-1]['x'][i]
        self.sol = sol
        return sol


############################################################
############################################################
############################################################
class AutoGradOptimize(StochasticOptimizer):
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
            x = np.random.random(self.x0.shape)
            rho = VonNeumannDensity(A=None, L=self.L, beta=beta).density
            converged = False
            opt_message = ''
            t = 0
            from autograd import grad as autograd_grad
            while not converged:
                t += 1

                def cost(z):
                    return SpectralDivergence(Lobs=self.L, Lmodel=graph_laplacian(self.samplingfun(z)),
                                              beta=beta).rel_entropy

                grad_t = autograd_grad(cost)
                grad_t = grad_t(x)

                # if clip_gradients is None:
                #    grad_t = self.gradient(x,rho,beta)
                # else:
                #    grad_t = np.clip(self.gradient(x,rho,beta),clip_gradients[0],clip_gradients[1]) # clip the gradients
                # Convergence status

                # if np.linalg.norm(grad_t) < gtol:
                #    converged, opt_message = True, 'gradient tolerance exceeded'
                if t > max_iters:
                    converged, opt_message = True, 'max_iters_exceed'
                # if x[0]<0:
                #    converged, opt_message = True, 'bounds_exceeded'
                x_old = x.copy()
                x -= eta * grad_t
                print('\rbeta=', beta, '|grad|=', np.linalg.norm(grad_t), 'x=', np.linalg.norm(x), ' m=',
                      self.modelfun(x).sum() / 2, end='')
                if self.step_callback is not None:
                    self.step_callback(beta, x)
            print(opt_message, t)

            sol.append({'x': x})
            # Here creates the output data structure as a dictionary of the optimization parameters and variables
            sol[-1]['opt_message'] = opt_message
            spect_div = SpectralDivergence(Lobs=self.L, Lmodel=graph_laplacian(self.samplingfun(sol[-1]['x'])),
                                           beta=beta)
            sol[-1]['DeltaL'] = (np.trace(self.L) - np.trace(graph_laplacian(self.samplingfun(sol[-1]['x'])))) / 2
            if kwargs.get('compute_sigma', False):
                Lmodel = graph_laplacian(self.modelfun(sol[-1]['x']))
                rho = VonNeumannDensity(A=None, L=self.L, beta=beta).density
                sigma = VonNeumannDensity(A=None, L=Lmodel, beta=beta).density
                sol[-1]['<DeltaL>'] = np.trace(np.dot(rho, Lmodel)) - np.trace(np.dot(sigma, Lmodel))
            sol[-1]['T'] = 1 / beta
            sol[-1]['beta'] = beta
            sol[-1]['loglike'] = spect_div.loglike
            sol[-1]['rel_entropy'] = spect_div.rel_entropy
            sol[-1]['entropy'] = spect_div.entropy
            sol[-1]['AIC'] = 2 * len(self.modelfun.args_mapping) - 2 * sol[-1]['loglike']
            # for i in range(0, len(self.modelfun.args_mapping)):
            #    sol[-1][self.modelfun.args_mapping[i]] = sol[-1]['x'][i]
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
        num_samples = kwargs.get('num_samples', 1)
        clip_gradients = kwargs.get('clip_gradients', None)
        max_iters = kwargs.get('max_iters', 1000)
        alpha = kwargs.get('alpha', 1E-3)
        gtol = kwargs.get('gtol', 1E-3)
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1E-8

        # Populate the solution list as function of beta
        # the list sol contains all optimization points
        sol = []
        # Iterate over all beta provided by the user
        mt, vt = np.zeros(self.x0.shape), np.zeros(self.x0.shape)
        all_x = []
        for beta in self.beta_range:
            rho = VonNeumannDensity(A=None, L=self.L, beta=beta).density
            converged = False
            t = 0
            opt_message = ''
            while not converged:
                t += 1
                grad_t = None
                if clip_gradients is None:
                    grad_t = 1E3 * self.gradient(x, rho, 1E-3, num_samples)
                else:
                    grad_t = 1E3 * np.clip(self.gradient(x, rho, 1E-3), clip_gradients[0],
                                           clip_gradients[1])  # clip the gradients

                if np.linalg.norm(grad_t) < gtol:
                    converged, opt_message = True, 'gradient tolerance exceeded'
                    break
                if t > max_iters:
                    converged, opt_message = True, 'max_iters_exceed'
                    break
                if x[0] < 0:
                    converged, opt_message = True, 'bounds_exceeded'
                    break
                x_old = x.copy()

                mt = beta1 * mt + (1.0 - beta1) * grad_t
                vt = beta2 * vt + (1.0 - beta2) * grad_t * grad_t
                mttilde = mt / (1.0 - (beta1 ** t))  # compute bias corrected first moment estimate
                vttilde = vt / (1.0 - (beta2 ** t))  # compute bias-corrected second raw moment estimate
                x_old = x.copy()
                x -= alpha * mttilde / np.sqrt(vttilde + epsilon)
                print('\rbeta=', beta, 'grad=', grad_t[0], 'x=', x, end='')
                if self.step_callback is not None:
                    self.step_callback(beta, x)
                all_x.append(x[0])
            sol.append({'x': x.copy()})
            # Here creates the output data structure as a dictionary of the optimization parameters and variables
            spect_div = SpectralDivergence(Lobs=self.L, Lmodel=graph_laplacian(self.samplingfun(sol[-1]['x'])),
                                           beta=beta)
            sol[-1]['DeltaL'] = (np.trace(self.L) - np.trace(graph_laplacian(self.samplingfun(sol[-1]['x'])))) / 2
            sol[-1]['T'] = 1 / beta
            sol[-1]['beta'] = beta
            sol[-1]['loglike'] = spect_div.loglike
            sol[-1]['rel_entropy'] = spect_div.rel_entropy
            sol[-1]['entropy'] = spect_div.entropy
            sol[-1]['AIC'] = 2 * len(self.modelfun.args_mapping) - 2 * sol[-1]['loglike']
            for i in range(0, len(self.modelfun.args_mapping)):
                sol[-1][self.modelfun.args_mapping[i]] = sol[-1]['x'][i]

            if kwargs.get('compute_sigma', False):
                Lmodel = graph_laplacian(self.modelfun(sol[-1]['x']))
                rho = VonNeumannDensity(A=None, L=self.L, beta=beta).density
                sigma = VonNeumannDensity(A=None, L=Lmodel, beta=beta).density
                sol[-1]['<DeltaL>'] = np.trace(np.dot(rho, Lmodel)) - np.trace(np.dot(sigma, Lmodel))

        self.sol = sol
        return sol

