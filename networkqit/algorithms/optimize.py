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

import autograd
import autograd.numpy as np
from scipy.linalg import eigvalsh # a different routine, with no autograd support
from autograd.numpy.linalg import eigh
from autograd.scipy.misc import logsumexp
from scipy.optimize import minimize, least_squares, fsolve
from networkqit.graphtheory import *

from networkqit.infotheory.density import *


class ModelOptimizer:
    def __init__(self, **kwargs):
        """
        This represents the base abstract class from which to inherit all the possible model optimization classes
        """
        super().__init__()

    def setup(self, **kwargs):
        """
        The setup method must be implemented by each single inherited class.
        Here all the details and additional optimization variables are inserted as kwargs.
        """
        pass

    def gradient(self, **kwargs):
        """
        The gradient method must be implemented by each single inherited class
        Here all the details and additional optimization variables are inserted as kwargs.
        The gradient method muust return a NxNxk array where N is the dimension of the network
        to optimize and k is the number of free parameters of the model.
        """
        pass

    def run(self, model, **kwargs):
        """
        The  method to call to start optimization
        """
        pass


class MLEOptimizer(ModelOptimizer):
    """
    This class, inheriting from the model optimizer class solves the problem of 
    maximum likelihood parameters estimation in the classical settings.
    """

    def __init__(self, G, x0, **kwargs):
        """
        Init the optimizer.
        
        Args:
            G (numpy.array) :is the empirical network to study. A N x N adjacency matrix as numpy.array. Can be weighted or unweighted.
            x0 (numpy.array): is the k-element array of initial parameters estimates. Typically set as random.    
        """
        self.G = G
        self.x0 = x0
        self.sol = None
        super().__init__(**kwargs)

    def run(self, model, **kwargs):
        """
        Maximimize the likelihood of the model given the observed network G. 
        """
        # Use the Trust-Region reflective algorithm to maximize loglikelihood
        opts = {'ftol': kwargs.get('ftol', 1E-10),
                'gtol': kwargs.get('gtol', 1E-10),
                'eps':  kwargs.get('eps', 1E-10),
                }

        # Optimize using L-BFGS-B which typically returns good results
        self.sol = minimize(fun=lambda z: -model.loglikelihood(self.G, z),
                            x0=np.squeeze(self.x0),
                            method='L-BFGS-B',
                            bounds=model.bounds,
                            options=opts)

        if self.sol['status'] != 0:
            raise Exception('Method did not converge to maximum likelihood')

        return self.sol

    def runfsolve(self, model, **kwargs):
        """
        Alternative method to estimate model parameters based on 
        the solution of saddle-point equations (non-linear system of equations).
        This method finds the parameters such that the gradients are zero.
        Here we adopt two different methods. Models leading to convex likelihood
        are solved by a direct call to the least-square optimizer, where we 
        minimize the square error cost function:

        $sum_i (c_i^* - <c_i>)^2$
        
        where c_i is the empirical value of the specific constraint from MaxEnt 
        and <c_i> is its ensemble average as defined from the model.

        Empirically we found that the 'dogbox' method works best in most cases 
        with a nice tradeoff between speed and precision.

        Args:
            model: a model from MEModels with "saddle_point" method implemented
        
        kwargs:
            basin_hopping: (bool) whether to run global optimization with local
                            optimization by least_squares
            basin_hopping_niter: (int) number of basin hopping repetitions
            xtol:       (float) set the termination tolerance on parameters
                        (default 1E-10)
            gtol:       (float) set the termination tolerance on gradients
                        (default 1E-10)
            max_nfev:   (int) set the maximum number of evaluations of the cost.
                        Default value number of parameters * 1E6
            verbose:    (int) Verbosity level. No output=0, 2=iterations output.
        
        Outputs:
            sol: (np.array) parameters at optimal likelihood
        """
        # Use the Dogbox method to optimize likelihood
        def basin_opt(*basin_opt_args, **basin_opt_kwargs):
            max_nfev = len(self.x0) * 100000
            opt_result = least_squares(fun=basin_opt_args[0],
                                       x0=np.squeeze(basin_opt_args[1]),
                                       bounds=[np.finfo(float).eps, np.inf],
                                       method='dogbox',
                                       xtol=kwargs.get('xtol', 1E-10),
                                       gtol=kwargs.get('xtol', 1E-10),
                                       max_nfev=kwargs.get('max_nfev', max_nfev),
                                       verbose=kwargs.get('verbose',0))
            # use this form with linear loss as in the basinhopping
            # func argument to be consistent
            opt_result['fun'] = 0.5 * np.sum(opt_result['fun']**2)
            return opt_result

        # If the model is convex, we simply run least_squares
        if not kwargs.get('basinhopping', False):
            self.sol = basin_opt(lambda z: model.saddle_point(self.G, z), self.x0)
        else:  # otherwise combine local and global optimization with basinhopping
            from .basinhoppingmod import basinhopping, BHBounds, BHRandStepBounded
            xmin = np.zeros_like(self.x0) + np.finfo(float).eps
            xmax = xmin + np.inf  # broadcast infinity
            bounds = BHBounds(xmin=xmin)
            bounded_step = BHRandStepBounded(xmin, xmax, stepsize=0.5)
            self.sol = basinhopping(func=lambda z: 0.5*(model.saddle_point(self.G, z)**2).sum(),
                                    x0=np.squeeze(self.x0),
                                    T=kwargs.get('T', 1),
                                    minimize_routine = basin_opt,
                                    minimizer_kwargs={'saddle_point_equations': lambda z: model.saddle_point(self.G, z)},
                                    accept_test=bounds,
                                    take_step=bounded_step,
                                    niter=kwargs.get('basin_hopping_niter', 5),
                                    disp=bool(kwargs.get('verbose')))
        return self.sol


class ExpectedModelOptimizer(ModelOptimizer):
    """
    Continuos optimization method of spectral entropy in the continuous approximation S(rho, sigma(E[L])))
    """

    def __init__(self, A, x0, beta_range, **kwargs):
        """
        Initialization method, must provide the observed network in form of adjacency matrix,
        the initial optimization parameters and the range over which to span $\beta$.

        args:
            A (numpy.array): The observed adjacency matrix
            x0 (numpy.array): The initial value of the optimization parameters (also called θ_0)
            beta_range (numpy.array, list): The values for which to run optimization
        """
        self.A = A
        self.L = graph_laplacian(A)
        self.beta_range = beta_range
        self.x0 = x0
        self.expected_adj_fun = None
        self.expected_lapl_fun_grad = None
        self.step_callback = None
        self.bounds = None
        super().__init__(**kwargs)

    def setup(self, model, step_callback=None):
        """
        Setup the optimizer. Must specify the model function, as a function that returns the expected adjacency matrix,
        Optionally one can also provide the modelfun_grad, a function that returns the gradient of the expected Laplacian.

        args:
            model: a model from GraphModel
        """
        self.expected_adj_fun = model.expected_adjacency
        self.expected_lapl_fun_grad = model.expected_laplacian
        self.step_callback = step_callback
        self.bounds = model.bounds

    def gradient(self, x, rho, beta):
        """
        This method computes the gradient as 
        
        :math:`\\frac{s(\\rho \| \\sigma)}{\\partial \\theta_k} = \\beta \textrm{Tr}\left \lbrack \left( \rho - \sigma(\theta)\right) \frac{\mathbb{E}\mathbf{L}(\theta)}{\partial \theta_k} \right \rbrack`
        
        args:
            x (numpy.array): the current parameters
            rho (numpy.array): the observed density matrix
            beta (float): the beta, a positive real.

        Returns:
            the gradient as a three index numpy array. The last index is the one pertaining to the k-th component of x
        """
        sigma = compute_vonneuman_density(graph_laplacian(self.expected_adj_fun(x)), beta)
        # Here instead of tracing over the matrix product, we just sum over the entrywise product of the two matrices
        # (rho-sigma) and the ∂E_θ[L]/.
        return np.array([np.sum(np.multiply(rho - sigma, self.self.expected_lapl_fun_grad(x)[:, i, :].T)) for i in range(0, len(self.x0))])

    def run(self, **kwargs):
        """
        Starts the optimization. Default options are:

        method: 'BFGS'
        if the optimization problem is bounded, instead use one of the scipy constrained optimizer,
        which are 'L-BFGS-B', 'TNC', 'SLSQP' or 'least_squares'
        
        **kwargs:
            gtol: gradient tolerance to be used in optimization (default 1E-12)
            maxfun: maximum number of function evaluations
            maxiter: maximum number of iterations of gradient descent
            xtol: tolerance in the change of variables theta
            
        """
        self.method = kwargs.get('method', 'BFGS')
        if self.bounds is not None and self.method not in ['L-BFGS-B', 'TNC', 'SLSQP', 'least_squares']:
            raise RuntimeWarning(
                'This model has bounds. BFGS cannot handle constraints nor bounds, switch to either L-BFGS-B, TNC or SLSQP')

        # Populate the solution list as function of beta
        # the list sol contains all optimization points
        sol = []
        # Iterate over all beta provided by the user
        for beta in self.beta_range:
            # define the relative entropy function, dependent on current beta
            self.rel_entropy_fun = lambda x: SpectralDivergence(Lobs=self.L, Lmodel=graph_laplacian(self.expected_adj_fun(x)),
                                                                beta=beta).rel_entropy

            # Define initially fgrad as None, relying on numerical computation of it
            fgrad = None

            # Define observed density rho as none initially, if necessary it is computed
            rho = None

            # If user provides gradients of the model, use them, redefyining fgrad to pass to solver
            if self.self.expected_lapl_fun_grad is not None:
                # compute rho for each beta, just once
                rho = VonNeumannDensity(A=None, L=self.L, beta=beta).density
                # define the gradient of the Dkl, given rho at current beta
                fgrad = lambda x: self.gradient(x, rho, beta)

            # Append the solutions
            # Here we adopt two different strategies, either minimizing the residual of gradients of Dkl
            # using least_squares or minimizing the Dkl itself.
            # The least_squares approach requires the gradients of the model, if they are not implemented 
            # in the model, the algorithmic differentiation is used, which is slower
            if self.method is 'least_squares':
                if rho is None:  # avoid recomputation of rho
                    rho = VonNeumannDensity(A=None, L=self.L, beta=beta).density
                sol.append(least_squares(lambda x: self.gradient(x, rho, beta), x0=self.x0,
                                         bounds=kwargs.get('bounds', (0, np.inf)),
                                         loss=kwargs.get('loss', 'soft_l1'),
                                         # robust choice for the loss function of the residuals
                                         xtol=kwargs.get('xtol', 1E-9),
                                         gtol=kwargs.get('gtol', 1E-12)))
            else:  # otherwise directly minimize the relative entropy function (change the default arguments)
                sol.append(minimize(fun=self.rel_entropy_fun,
                                    x0=self.x0,
                                    jac=fgrad,
                                    method=self.method,
                                    options={'disp': kwargs.get('disp', False),
                                             'gtol': kwargs.get('gtol', 1E-12),
                                             'maxiter': kwargs.get('maxiter', 5E4),
                                             'maxfun': kwargs.get('maxfun', 5E4)},
                                    bounds=self.bounds))

            # important to reinitialize from the last solution, solution is restarted at every step otherwise
            if kwargs.get('reinitialize', True):
                self.x0 = sol[-1].x
            # Call the step_callback function to print or display current solution
            if self.step_callback is not None:
                self.step_callback(beta, sol[-1].x)

            # Here creates the output data structure as a dictionary of the optimization parameters and variables
            spect_div = SpectralDivergence(Lobs=self.L, Lmodel=graph_laplacian(self.expected_adj_fun(sol[-1].x)), beta=beta)
            sol[-1]['DeltaL'] = (np.trace(self.L) - np.trace(graph_laplacian(self.expected_adj_fun(sol[-1].x)))) / 2
            if kwargs.get('compute_sigma', False):
                Lmodel = graph_laplacian(self.expected_adj_fun(sol[-1].x))
                rho = VonNeumannDensity(A=None, L=self.L, beta=beta).density
                sigma = VonNeumannDensity(A=None, L=Lmodel, beta=beta).density
                sol[-1]['<DeltaL>'] = np.trace(np.dot(rho, Lmodel)) - np.trace(np.dot(sigma, Lmodel))
                # from scipy.integrate import quad
                # from numpy import vectorize
                # Q1 = vectorize(quad)(lambda x : expm(-x*beta*self.L)@(self.L-Lmodel)@expm(x*beta*self.L),0,1)
                # sol[-1]['<DeltaL>_1'] = beta*( np.trace(rho@Q1@Lmodel) - np.trace(rho@Q1)*np.trace(rho@Lmodel) )
            sol[-1]['T'] = 1 / beta
            sol[-1]['beta'] = beta
            sol[-1]['loglike'] = spect_div.loglike
            sol[-1]['rel_entropy'] = spect_div.rel_entropy
            sol[-1]['entropy'] = spect_div.entropy
            sol[-1]['AIC'] = 2 * len(self.expected_adj_fun.args_mapping) - 2 * sol[-1]['loglike']
            for i in range(0, len(self.expected_adj_fun.args_mapping)):
                sol[-1][self.expected_adj_fun.args_mapping[i]] = sol[-1].x[i]
        self.sol = sol
        return sol

    def summary(self, to_dataframe=False):
        """
        A convenience function to summarize all the optimization process, with results of optimization.

        args:
            to_dataframe (bool): if True, returns a pandas dataframe, otherwise returns a list of dictionaries
        """
        if to_dataframe:
            import pandas as pd
            return pd.DataFrame(self.sol).set_index('T')  # it's 1/beta
        else:
            s = "{:>20} " * (len(self.expected_adj_fun.args_mapping) + 1)
            print('=' * 20 * (len(self.expected_adj_fun.args_mapping) + 1))
            print('Summary:')
            print('Model:\t' + str(self.expected_adj_fun.formula))
            print('=' * 20 * (len(self.expected_adj_fun.args_mapping) + 1))
            print('Optimization method: ' + self.method)
            print('Variables bounds: ')
            for i, b in enumerate(self.bounds):
                left = '-∞' if self.bounds[i][0] is None else str(
                    self.bounds[i][0])
                right = '+∞' if self.bounds[i][1] is None else str(
                    self.bounds[i][1])
                print("{: >1} {:} {: >10} {:} {: >1}".format(
                    left, '<=', self.expected_adj_fun.args_mapping[i], '<=', right))
            print('=' * 20 * (len(self.expected_adj_fun.args_mapping) + 1))
            print('Results:')
            print('=' * 20 * (len(self.expected_adj_fun.args_mapping) + 1))
            print('Von Neumann Log Likelihood:\t' +
                  str(self.sol[-1]['loglike']))
            print('Von Neumann Entropy:\t\t' + str(self.sol[-1]['entropy']))
            print('Von Neumann Relative entropy:\t' +
                  str(self.sol[-1]['rel_entropy']))
            print('AIC:\t\t\t\t' + str(self.sol[-1]['AIC']))
            print('=' * 20 * (len(self.expected_adj_fun.args_mapping) + 1))
            print('Estimate:')
            print('=' * 20 * (len(self.expected_adj_fun.args_mapping) + 1))
            print(s.format('beta', *self.expected_adj_fun.args_mapping))
            for i in range(0, len(self.sol)):
                row = [str(x) for x in self.sol[i].x]
                print(s.format(self.sol[i]['beta'], *row))


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
        self.A = A
        self.L = graph_laplacian(self.A)
        self.x0 = x0
        self.beta_range = beta_range

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

    def gradient(self, x, rho, beta, batch_size=1):
        # Compute the relative entropy between rho and sigma, using tensors with autograd support
        # TODO: Fix because now it has no multiple samples support
        Eobs = np.sum(self.L*rho) # = Tr[rho Lobs]
        lambd_obs = eigh(self.L)[0] # eigh is a batched operation
        Fobs = - logsumexp(-beta*lambd_obs) / beta
        entropy = beta*(Eobs-Fobs)

        def rel_entropy(z):
            N = len(self.A)
            # advanced broadcasting here!
            # Sample 'batch_size' adjacency matrices shape=[batch_size,N,N]
            Amodel = self.sample_adjacency_fun(z, batch_size=batch_size)
            Dmodel = np.eye(N) * np.transpose(np.zeros([1,1,N]) + np.einsum('ijk->ik', Amodel),[1,0,2])
            Lmodel =  Dmodel - Amodel # returns a batch_size x N x N tensor
            # do average over batches of the sum of product of matrix elements (done with * operator)
            Emodel = np.mean(np.sum(np.sum(Lmodel*rho,axis=2), axis=1))
            lambd_model = eigh(Lmodel)[0] # eigh is a batched operation, 
            Fmodel = - np.mean(logsumexp(-beta*lambd_model, axis=1) / beta)
            loglike = beta * (Emodel - Fmodel)
            dkl = loglike - entropy
            return dkl

        # value and gradient of relative entropy as a function
        dkl_and_dkldx = autograd.value_and_grad(rel_entropy)
        return dkl_and_dkldx(x)

        f_and_df = autograd.value_and_grad(rel_entropy_batched) # gradient of relative entropy as a function
        return f_and_df(x)

    def run(self, model, **kwargs):
        """
        Start the optimization process
        """
        raise NotImplementedError

class Adam(StochasticOptimizer):
    """
    Implements the ADAM stochastic gradient descent.
    Adam: A Method for Stochastic Optimization
    Diederik P. Kingma, Jimmy Ba

    https://arxiv.org/abs/1412.6980
    """

    def run(self, **kwargs):
        x = self.x0
        batch_size = kwargs.get('batch_size', 1)
        max_iters = kwargs.get('max_iters', 1000)
        # ADAM parameters
        eta = kwargs.get('eta', 1E-3)
        gtol = kwargs.get('gtol', 1E-4)
        xtol = kwargs.get('xtol', 1E-3)
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1E-8
        # visualization options
        refresh_frames = 10
        from drawnow import drawnow, figure
        import matplotlib.pyplot as plt
        # if global namespace, import plt.figure before drawnow.figure
        figure(figsize=(8, 4))
        # Populate the solution list as function of beta
        # the list sol contains all optimization points
        sol = []
        # Iterate over all beta provided by the user
        mt, vt = np.zeros(self.x0.shape), np.zeros(self.x0.shape)
        all_dkl = []
        # TODO implement model boundaries in Adam
        # bounds = np.array(np.ravel(self.model.bounds),dtype=float)
        for beta in self.beta_range:
            # if rho is provided, user rho is used, otherwise is computed at every beta
            rho = kwargs.get('rho', compute_vonneuman_density(L=self.L, beta=beta))
            # initialize at x0
            x = self.x0
            converged = False
            opt_message = ''
            t = 0 # t is the iteration number
            while not converged:
                t += 1
                # get the relative entropy value and its gradient w.r.t. variables
                dkl, grad_t =  self.gradient(x, rho, beta, batch_size=batch_size)
                # Convergence status
                if np.linalg.norm(grad_t) < gtol:
                    converged, opt_message = True, 'gradient tolerance exceeded'
                if t > max_iters:
                    converged, opt_message = True, 'maximum iterations exceed'
                # TODO implement check boundaries in Adam
                #if np.any(np.ravel(self.model.bounds)):
                #    raise RuntimeError('variable bounds exceeded')
                mt = beta1 * mt + (1.0 - beta1) * grad_t
                vt = beta2 * vt + (1.0 - beta2) * grad_t * grad_t
                mttilde = mt / (1.0 - (beta1 ** t))  # compute bias corrected first moment estimate
                vttilde = vt / (1.0 - (beta2 ** t))  # compute bias-corrected second raw moment estimate
                x_old = x.copy()
                x -= eta * mttilde / np.sqrt(vttilde + epsilon)
                
                all_dkl.append(dkl)
                if t % refresh_frames == 0:
                    def draw_fig():
                        sol.append({'x': x.copy()})
                        A0 = np.mean(self.sample_adjacency_fun(x, batch_size=batch_size),axis=0)
                        plt.subplot(2,2,1)
                        plt.imshow(self.A)
                        plt.subplot(2,2,2)
                        plt.imshow(A0)
                        plt.subplot(2,2,3)
                        plt.plot(all_dkl)
                        plt.subplot(2,2,4)
                        plt.semilogx(self.beta_range,batch_compute_vonneumann_entropy(self.L, self.beta_range),'.-')
                        plt.semilogx(self.beta_range,batch_compute_vonneumann_entropy(graph_laplacian(A0), self.beta_range),'.-')
                        #plt.legend(loc='best')
                        plt.tight_layout()
                    drawnow(draw_fig)
        self.sol = sol
        return sol
