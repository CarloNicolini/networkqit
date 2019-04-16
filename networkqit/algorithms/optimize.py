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
The most important class to optimize an expected adjacency model is `ContinuousOptimizer`.
In this class the gradients are defined as:

.. math::
    
        \\frac{\\partial S(\\boldsymbol \\rho \\| \\boldsymbol \\sigma(\\mathbb{E}_{\\boldsymbol \\theta}[\\mathbf{L}]))}{\\partial \\boldsymbol \\theta_k} = \\beta \\textrm{Tr}\\biggl \\lbrack \\left(\\boldsymbol \\rho - \\boldsymbol \\sigma(\\mathbb{E}_{\\boldsymbol \\theta}[L])\\right)\\frac{\\partial \\mathbb{E}_{\\boldsymbol \\theta}[\mathbf{L}]}{\\partial \\theta_k} \\biggr \\rbrack

In the `StochasticOptimizer` class we instead address the issue to implement stochastic gradient descent methods.
In these methods the gradients are defined as:

.. math::
   
   \\frac{\\partial \\mathbb{E}_{\\boldsymbol \\theta}[S(\\boldsymbol \\rho \\| \\boldsymbol \\sigma)]}{\\partial \\theta_k} = \\beta \\textrm{Tr}\\biggl \\lbrack \\boldsymbol \\rho \\frac{\\partial  \\mathbb{E}_{\\boldsymbol \\theta}[L]}{\\partial \\boldsymbol \\theta_k}\\biggr \\rbrack + \\frac{\\partial}{\\partial \\theta_k}\\mathbb{E}_{\\boldsymbol \\theta}\\biggl \\lbrack \\log \\left( \\textrm{Tr}\\left \\lbrack e^{-\\beta L(\\boldsymbol \\theta)} \\right \\rbrack \\right) \\biggr \\rbrack

The stochastic optimizer is the **correct** optimizer, as it makes no approximation on the Laplacian eigenvalues.
It is more suitable for small graphs and intermediate $\\beta$, where the differences between the random matrix s
pectrum and its expected counterpart are non-neglibile.
For large and dense enough graphs however the `ContinuousOptimizer` works well and yields deterministic results,
as the optimization landscape is smooth.

In order to minimize the expected relative entropy then, we need both the expected Laplacian formula, which is simple
to get, and a way to estimate the second summand in the gradient, that involves averaging over different realizations
of the log trace of  $e^{-\\beta L(\\boldsymbol \\theta)}$.
A good the approximation to the expected logtrace $\\mathbb{E}_{\\boldsymbol \\theta}[\\log \\textrm{Tr}[\\exp{(-\\beta L)}]]$ is,
makes a better is the estimate of the gradients.

Finally, the `MLEOptimizer` maximizes the standard likelihood of a model and it is not related to the spectral entropies
framework introduced in the paper on which **networkqit** is based.

"""
#    Copyright (C) 2018 by
#    Carlo Nicolini <carlo.nicolini@iit.it>
#    All rights reserved.
#    BSD license.

import autograd
import autograd.numpy as np
from autograd.numpy.linalg import eigh
from autograd.scipy.misc import logsumexp
from networkqit.graphtheory.matrices import softmax
from scipy.optimize import minimize, least_squares, OptimizeResult
from networkqit.graphtheory import * # imports GraphModel
from networkqit.infotheory.density import *

__all__ = ['MLEOptimizer',
           'ContinuousOptimizer',
           'StochasticOptimizer',
           'Adam']

class MLEOptimizer:
    """
    This class, inheriting from the model optimizer class solves the problem of 
    maximum likelihood parameters estimation in the classical settings.
    """

    def __init__(self, G: np.array, x0: np.array, model: GraphModel, **kwargs):
        """
        Initialize the optimizer with the observed graph, an initial guess and the 
        model to optimize.

        Args:
            G (numpy.array) :is the empirical network to study. A N x N adjacency matrix as numpy.array.
            x0 (numpy.array): is the k-element array of initial parameters estimates. Typically set as random.
            model (nq.GraphModel): the graph model to optimize the likelihood of.
        """
        self.G = G
        self.x0 = x0
        self.model = model
        self.sol = None
        super().__init__(**kwargs)

    def run(self, method, ftol=1E-10, gtol=1E-5, xtol=1E-8, maxiter=1E4, **kwargs):
        """
        Maximimize the likelihood of the model given the observed network G.

        Args:
            method (str): optimization method to use. Can be either 'MLE' or 'saddle_point'.
                          'MLE' uses L-BFGS-B method to optimize the likelihood of the model
                          'saddle_point' finds the saddle point solution solving a system of
                          nonlinear equation. This method finds the parameters such that the
                          gradients are zero. Here we adopt two different methods.
                          Models leading to convex likelihood are solved by a direct call to
                          the least-square optimizer, where we minimize the square error cost:
                          $sum_i (c_i^* - <c_i>)^2$
                          where c_i is the empirical value of the specific constraint from MaxEnt
                          and <c_i> is its ensemble average as defined from the model.
                          Empirically we found that the 'dogbox' method works best in most cases
                          with a nice tradeoff between speed and precision.

            xtol:       (float) set the termination tolerance on parameters
                        (default 1E-10)

            gtol:       (float) set the termination tolerance on gradients
                        (default 1E-10)

            maxiter:    (int) set the maximum number of iteration. Default 10.000 iterations

        Kwargs:
            maxfun:     (int) set the maximum number of evaluations of the cost.
                        Default value is very large ten thousand (1E4)

            verbose:    (int) Verbosity level. No output=0, 2=iterations output.

            basin_hopping: (bool) whether to run global optimization with local
                optimization by least_squares (only for saddle point method)

            basin_hopping_niter: (int) number of basin hopping repetitions

        Outputs:
            sol: (scipy.optimize.OptimizeResult) parameters at optimal likelihood
        """

        opts = {'ftol': ftol,
                'gtol': gtol, # as default of LBFGSB
                'xtol': xtol,
                'maxiter': maxiter,
                'eps': kwargs.get('eps', 1E-8), # as default of LBFGSB
                'maxfun': kwargs.get('maxfun', 1E10),
                'verbose' : kwargs.get('verbose', 0),
                'disp':  bool(kwargs.get('verbose', 0)),
                'iprint': kwargs.get('verbose', 0),
                'maxls' : 100
                }

        if method is 'MLE':
            # If the model has non-linear constraints, must use Sequential Linear Square Programming SLSQP
            from autograd import jacobian # using automatic jacobian from autograd
            J = jacobian(lambda z : -self.model.loglikelihood(self.G,z))
            if hasattr(self.model, 'constraints'):
                #H=hessian(lambda z : -self.model.loglikelihood(self.G,z))
                # remove some options to avoid warnings
                opts.pop('gtol'), opts.pop('verbose'), opts.pop('xtol'),opts.pop('maxfun')
                self.sol = minimize(fun=lambda z: -self.model.loglikelihood(self.G, z),
                                    x0=np.squeeze(self.x0),
                                    method='SLSQP',
                                    constraints={'fun':self.model.constraints, 'type':'ineq'},
                                    bounds=self.model.bounds,
                                    jac=J,
                                    tol=opts['ftol'],
                                    options=opts)
            else: # the model has only bound-constraints, hence use L-BFGS-B
                # Optimize using L-BFGS-B which typically returns good results
                opts.pop('xtol'), opts.pop('verbose')
                self.sol = minimize(fun=lambda z: -self.model.loglikelihood(self.G, z),
                                    x0=np.squeeze(self.x0),
                                    method='L-BFGS-B',
                                    jac=J,
                                    bounds=self.model.bounds,
                                    options=opts)
            if kwargs.get('verbose',0)>0:
                print(self.sol['message'])
                if self.sol['status'] != 0:
                    RuntimeWarning(self.sol['message'])
                #raise Exception('Method did not converge to maximum likelihood: ')
        elif method is 'saddle_point':
            if hasattr(self.model, 'constraints'):
                raise RuntimeError('Cannot solve saddle_point_equations with non-linear constraints')
            # Use the Dogbox method to optimize likelihood
            #ls_bounds = [np.min(np.ravel(self.model.bounds)), np.max(np.ravel(self.model.bounds))]
            ls_bounds = np.ravel(self.model.bounds).astype(float)
            ls_bounds[np.isnan(ls_bounds)] = np.inf
            ls_bounds = np.reshape(ls_bounds,[len(self.x0),2]).T.tolist() # it is required in a different format
            def basin_opt(*basin_opt_args, **basin_opt_kwargs):
                opt_result = least_squares(fun=basin_opt_args[0],
                                           x0=np.squeeze(basin_opt_args[1]),
                                           bounds=ls_bounds,
                                           method='trf',
                                           xtol=opts['xtol'],
                                           gtol=opts['gtol'],
                                           tr_solver='lsmr',
                                           max_nfev=opts['maxiter'],
                                           verbose=opts['verbose'])
                # use this form with linear loss as in the basinhopping
                # func argument to be consistent
                opt_result['fun'] = 0.5 * np.sum(opt_result['fun'] ** 2)
                return opt_result

            # If the model is convex, we simply run least_squares
            if not kwargs.get('basinhopping', False):
                self.sol = basin_opt(lambda z: self.model.saddle_point(self.G, z), self.x0)
            else:
                # otherwise combine local and global optimization with basinhopping
                nlineq = lambda z: self.model.saddle_point(self.G, z)
                from .basinhoppingmod import basinhopping, BHBounds, BHRandStepBounded
                bounds = BHBounds(xmin=ls_bounds[0], xmax=ls_bounds[1])
                bounded_step = BHRandStepBounded(ls_bounds[0], ls_bounds[1], stepsize=0.5)
                self.sol = basinhopping(func=lambda z: (nlineq(z)**2).sum(),
                                        x0=np.squeeze(self.x0),
                                        T=kwargs.get('T', 1),
                                        minimize_routine=basin_opt,
                                        minimizer_kwargs={'saddle_point_equations': nlineq },
                                        accept_test=bounds,
                                        take_step=bounded_step,
                                        niter=kwargs.get('basin_hopping_niter', 5),
                                        disp=bool(kwargs.get('verbose')))
        
        else:
            raise RuntimeError('Only MLE and saddle_point methods are supported')
        
        # Compute the corrected Akaike information and Bayes information criteria
        # http://downloads.hindawi.com/journals/complexity/2019/5120581.pdf
        K = len(self.sol['x'])
        N = len(self.G)
        n = N*(N-1)/2 # n is the sample size
        # Both AIC and BIC are minimum for the best explanatory model
        L = self.model.loglikelihood(self.G,self.sol['x'])
        self.sol['AIC'] = -2*L + 2*K + (2*K*(K+1)) / (n-K+1)
        self.sol['BIC'] = -2*L + K*np.log(n)
        return self.sol


class ContinuousOptimizer:
    """
    Continuos optimization method of spectral entropy in the continuous approximation S(rho, sigma(E[L])))
    """

    def __init__(self, A, x0, beta_range, model, **kwargs):
        """
        Initialization method, must provide the observed network in form of adjacency matrix,
        the initial optimization parameters and the range over which to span $\beta$.

        args:
            A (numpy.array): The observed adjacency matrix
            x0 (numpy.array): The initial value of the optimization parameters (also called θ_0)
            beta_range (numpy.array, list): The values for which to run optimization
        """
        self.A = A
        self.x0 = x0
        self.beta_range = beta_range
        self.L = graph_laplacian(A)
        self.model = model
        self.step_callback = None
        super().__init__(**kwargs)

    def gradient(self, x, rho, beta):
        """
        This method computes the gradient as 
        
        :math:`\\frac{s(\\boldsymbol \\rho \| \\boldsymbol \\sigma)}{\\partial \\boldsymbol \\theta_k} = \\beta \textrm{Tr}\left \lbrack \left( \rho - \sigma(\theta)\right) \frac{\mathbb{E}\mathbf{L}(\theta)}{\partial \theta_k} \right \rbrack`
        
        args:
            x (numpy.array): the current parameters
            rho (numpy.array): the observed density matrix
            beta (float): the beta, a positive real.

        Returns:
            the gradient as a three index numpy array. The last index is the one pertaining to the k-th component of x
        """
        sigma = compute_vonneuman_density(graph_laplacian(self.model.expected_adjacency(x)), beta)
        # Here instead of tracing over the matrix product, we just sum over the entrywise product of the two matrices
        # (rho-sigma) and the ∂E_θ[L]/.
        return np.array([np.sum(np.multiply(rho - sigma, self.model.expected_laplaplacian_grad(x)[:, i, :].T)) for i in
                         range(0, len(self.x0))])

    def run(self, **kwargs):
        """
        Starts the optimization.

        Args:
            method (string): 'BFGS'
            if the optimization problem is bounded, instead use one of the scipy constrained optimizer,
            which are 'L-BFGS-B', 'TNC', 'SLSQP' or 'least_squares'
        
        Kwargs:

            gtol: (float)
                gradient tolerance to be used in optimization (default 1E-12)

            maxfun (int):
                maximum number of function evaluations

            maxiter (int):
                maximum number of iterations of gradient descent

            xtol (float):
                tolerance in the solution change

        Output:
            sol: (scipy.optimize.OptimizeResult) parameters at optimal likelihood
        """

        self.method = kwargs.get('method', 'BFGS')
        if self.model.bounds is not None and self.method not in ['L-BFGS-B', 'TNC', 'SLSQP', 'least_squares']:
            raise RuntimeWarning('This model has bounds. BFGS cannot handle constraints nor bounds, switch to either '
                                 'L-BFGS-B, TNC or SLSQP')

        # Populate the solution list as function of beta
        # the list sol contains all optimization points
        sol = []
        # Iterate over all beta provided by the user
        for beta in self.beta_range:
            # define the relative entropy function, dependent on current beta
            self.rel_entropy_fun = lambda x: SpectralDivergence(Lobs=self.L,
                                                                Lmodel=graph_laplacian(self.model.expected_adjacency(x)),
                                                                beta=beta).rel_entropy

            # Define initially fgrad as None, relying on numerical computation of it
            fgrad = None

            # Define observed density rho as none initially, if necessary it is computed
            rho = None

            # If user provides gradients of the model, use them, redefyining fgrad to pass to solver
            if self.model.expected_laplaplacian_grad is not None:
                # compute rho for each beta, just once
                rho = compute_vonneuman_density(L=self.L, beta=beta)
                # define the gradient of the Dkl, given rho at current beta
                fgrad = lambda x: self.gradient(x, rho, beta)

            # Append the solutions
            # Here we adopt two different strategies, either minimizing the residual of gradients of Dkl
            # using least_squares or minimizing the Dkl itself.
            # The least_squares approach requires the gradients of the model, if they are not implemented 
            # in the model, the algorithmic differentiation is used, which is slower

            sol.append(minimize(fun=self.rel_entropy_fun,
                                x0=self.x0,
                                jac=fgrad,
                                method=self.method,
                                options={'disp': kwargs.get('disp', False),
                                         'gtol': kwargs.get('gtol', 1E-12),
                                         'maxiter': kwargs.get('maxiter', 5E4),
                                         'maxfun': kwargs.get('maxfun', 5E4)},
                                bounds=self.model.bounds))

            # important to reinitialize from the last solution, solution is restarted at every step otherwise
            if kwargs.get('reinitialize', True):
                self.x0 = sol[-1].x
            # Call the step_callback function to print or display current solution
            if self.step_callback is not None:
                self.step_callback(beta, sol[-1].x)

            # Here creates the output data structure as a dictionary of the optimization parameters and variables
            spect_div = SpectralDivergence(Lobs=self.L, Lmodel=graph_laplacian(self.model.expected_adjacency(sol[-1].x)),
                                           beta=beta)
            sol[-1]['DeltaL'] = (np.trace(self.L) - np.trace(graph_laplacian(self.model.expected_adjacency(sol[-1].x)))) / 2
            if kwargs.get('compute_sigma', False):
                Lmodel = graph_laplacian(self.model.expected_adjacency(sol[-1].x))
                rho = compute_vonneuman_density(L=self.L, beta=beta)
                sigma = compute_vonneuman_density(L=Lmodel, beta=beta)
                sol[-1]['<DeltaL>'] = np.sum((rho-sigma)*Lmodel)
            sol[-1]['T'] = 1 / beta
            sol[-1]['beta'] = beta
            sol[-1]['loglike'] = spect_div.loglike
            sol[-1]['rel_entropy'] = spect_div.rel_entropy
            sol[-1]['entropy'] = spect_div.entropy
            sol[-1]['AIC'] = 2 * len(self.model.expected_adjacency.args_mapping) - 2 * sol[-1]['loglike']
            for i in range(0, len(self.model.expected_adjacency.args_mapping)):
                sol[-1][self.model.expected_adjacency.args_mapping[i]] = sol[-1].x[i]
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
            s = "{:>20} " * (len(self.model.expected_adjacency.args_mapping) + 1)
            print('=' * 20 * (len(self.model.expected_adjacency.args_mapping) + 1))
            print('Summary:')
            print('Model:\t' + str(self.model.expected_adjacency.formula))
            print('=' * 20 * (len(self.model.expected_adjacency.args_mapping) + 1))
            print('Optimization method: ' + self.method)
            print('Variables bounds: ')
            for i, b in enumerate(self.bounds):
                left = '-∞' if self.bounds[i][0] is None else str(
                    self.bounds[i][0])
                right = '+∞' if self.bounds[i][1] is None else str(
                    self.bounds[i][1])
                print("{: >1} {:} {: >10} {:} {: >1}".format(
                    left, '<=', self.model.expected_adjacency.args_mapping[i], '<=', right))
            print('=' * 20 * (len(self.model.expected_adjacency.args_mapping) + 1))
            print('Results:')
            print('=' * 20 * (len(self.model.expected_adjacency.args_mapping) + 1))
            print('Von Neumann Log Likelihood:\t' +
                  str(self.sol[-1]['loglike']))
            print('Von Neumann Entropy:\t\t' + str(self.sol[-1]['entropy']))
            print('Von Neumann Relative entropy:\t' +
                  str(self.sol[-1]['rel_entropy']))
            print('AIC:\t\t\t\t' + str(self.sol[-1]['AIC']))
            print('=' * 20 * (len(self.model.expected_adjacency.args_mapping) + 1))
            print('Estimate:')
            print('=' * 20 * (len(self.model.expected_adjacency.args_mapping) + 1))
            print(s.format('beta', *self.model.expected_adjacency.args_mapping))
            for i in range(0, len(self.sol)):
                row = [str(x) for x in self.sol[i].x]
                print(s.format(self.sol[i]['beta'], *row))


class StochasticOptimizer:
    """
    This class is at the base of implementation of methods based on stochastic gradient descent.
    The idea behind this class is to help the user in designing a nice stochastic gradient descent method,
    such as ADAM, AdaGrad or older methods, like the Munro-Robbins stochastic gradients optimizer.
    Working out the expression for the gradients of the relative entropy, one remains with the following:

    :math: `\nabla_{\theta}S(\rho \| \sigma) = \beta \textrm\biggl \lbrack \rho \nabla_{\theta}\mathbb{E}_{\theta}[L]} \biggr \rbrack`
        
    :math: `\frac{\partial S(\rho \| \sigma)}{\partial \theta_k} = \beta \textrm{Tr}\lbrack \rho \frac{\partial}{\partial \theta_k} \rbrack + \frac{\partial}{\partial \theta_k}\mathbb{E}_{\theta}\log \textrm{Tr} e^{-\beta L(\theta)}\lbrack \rbrack`
    
    This class requires either Tensorflow or Pytorch to support backpropagation in the eigenvalues routines.
    Alternatively you can use github.com/HIPS/autograd method for full CPU support.
    """

    def __init__(self, G: np.array, x0 : np.array, model : GraphModel, **kwargs):
        """
        Initialize the stochastic optimizer with the observed graph, an initial guess and the 
        model to optimize.

        Args:
            G (numpy.array) :is the empirical network to study. A N x N adjacency matrix as numpy.array.
            x0 (numpy.array): is the k-element array of initial parameters estimates. Typically set as random.
            model (nq.GraphModel): the graph model to optimize the likelihood of.
        """
        self.G = G
        self.L = graph_laplacian(self.G) # compute graph laplacian
        self.x0 = x0
        self.model = model
        self.sol = OptimizeResult(x=self.x0,
                                  success=False,
                                  status=None,
                                  message='',
                                  fun=None,
                                  jac=None,
                                  hess=None,
                                  hess_inv=None,
                                  nfev=0, # number of function evaluations
                                  njev=0, # number of jacobian evaluations
                                  nhev=0, # number of hessian evaluations
                                  nit=0,  # number of iterations
                                  maxcv=-1) # maximum constraint violation
        
        self.model.ls_bounds = np.ravel(self.model.bounds).astype(float)
        self.model.ls_bounds[np.isnan(self.model.ls_bounds)] = np.inf

        self.model.min_bounds = self.model.ls_bounds.reshape([len(self.x0),2])[:,0]
        self.model.max_bounds = self.model.ls_bounds.reshape([len(self.x0),2])[:,1]


    def gradient(self, x, rho, beta, batch_size=1):
        # Compute the relative entropy between rho and sigma, using tensors with autograd support
        # Entropy of rho is constant over the iterations
        # if beta is kept constant, otherwise a specific rho can be passed
        #lambd_rho = eigh(rho)[0]
        #entropy = -np.sum(lambd_rho * np.log(lambd_rho))

        def log_likelihood(z):
            N = len(self.G)
            # advanced broadcasting here!
            # Sample 'batch_size' adjacency matrices shape=[batch_size,N,N]
            Amodel = self.model.sample_adjacency(theta=z, batch_size=batch_size, with_grads=True)
            #print('Amodel nan?:', np.any(np.isnan(Amodel.ravel())))
            # Here exploit broadcasting to create batch_size diagonal matrices with the degrees
            Dmodel = np.eye(N) * np.transpose(np.zeros([1, 1, N]) + np.einsum('ijk->ik', Amodel, optimize=True), [1, 0, 2])
            Lmodel = Dmodel - Amodel  # returns a batch_size x N x N tensor
            # do average over batches of the sum of product of matrix elements (done with * operator)
            Emodel = np.mean(np.sum(np.sum(Lmodel * rho, axis=2), axis=1))
            lambd_model = eigh(Lmodel)[0]  # eigh is a batched operation, take the eigenvalues only
            Fmodel = - np.mean(logsumexp(-beta * lambd_model, axis=1) / beta)
            loglike = -beta * Emodel + Fmodel # - Tr[rho log(sigma)]
            #loglike  #- entropy # Tr[rho log(rho)] - Tr[rho log(sigma)]
            
            # Add a penalty term accouting for boundaries violation
            # Follows these ideas https://www.cs.jhu.edu/~ijwang/pub/01271742.pdf
            def quadratic_penalty(theta):
                z = np.zeros(theta.shape)
                # compute the penalty with respect to low bounds violation 
                penalty_low = np.sum(np.max(np.hstack([z, -theta + self.model.min_bounds]))**2)
                # compute the penalty with respect to high bounds violation 
                penalty_high = np.sum(np.max(np.hstack([z, theta - self.model.max_bounds]))**2)
                return 0.5 * (penalty_low + penalty_high)
            def absolute_value_penalty(theta):
                z = np.zeros(theta.shape)
                # compute the penalty with respect to low bounds violation 
                penalty_low = np.max(np.hstack([z, -theta + self.model.min_bounds]))
                # compute the penalty with respect to high bounds violation 
                penalty_high = np.max(np.hstack([z, theta - self.model.max_bounds]))
                return np.max(np.hstack(penalty_low,penalty_high))
            penalty = quadratic_penalty
            cost = -loglike + beta*penalty(z)
            return cost

        # value and gradient of relative entropy as a function
        dkl_and_dkldx = autograd.value_and_grad(log_likelihood)
        return dkl_and_dkldx(x)


class Adam(StochasticOptimizer):
    """
    Implements the ADAM stochastic gradient descent.
    Adam: A Method for Stochastic Optimization
    Diederik P. Kingma, Jimmy Ba

    https://arxiv.org/abs/1412.6980
    However here we use quasi-hyperbolic adam by default, with parameters nu1,nu2
    https://arxiv.org/pdf/1810.06801.pdf
    """
    def __init__(self, G: np.array, x0 : np.array, model : GraphModel, **kwargs):
        super().__init__(G, x0, model, **kwargs)
        self.logfile = open('adam.log','w')
        self.mt, self.vt = np.zeros(self.sol.x.shape), np.zeros(self.sol.x.shape)

    def run(self,
            beta,
            rho=None,
            maxiter=1E4,
            learning_rate=1E-3,
            beta1=0.9,
            beta2=0.999,
            nu1=0.7,
            nu2=1.0,
            epsilon=1E-8,
            batch_size=64,
            ftol=1E-10,
            gtol=1E-5,
            xtol=1E-8,
            last_iters = 100,
            quasi_hyperbolic = True,
            callback=None,
            **kwargs):
        
        opts = {'ftol': ftol,
                'gtol': gtol, # as default of LBFGSB
                'xtol': xtol,
                'maxiter': maxiter,
                'eps': kwargs.get('eps', 1E-8), # as default of LBFGSB
                'maxfun': kwargs.get('maxfun', 1E10),
                'verbose' : kwargs.get('verbose', 0),
                'disp':  bool(kwargs.get('verbose', 0)),
                'iprint': kwargs.get('verbose', 0)
                }

        # if rho is provided, user rho is used, otherwise is computed at every beta
        if rho is None:
            rho = compute_vonneuman_density(L=self.L, beta=beta)
        
        #logging.basicConfig(stream=sys.stderr, level=logging.INFO)
        #import logging
        #adam_logger = logging.getLogger('ADAM')
        #adam_logger.setLevel(logging.INFO)
        #all_dkl = []
        #from scipy.stats import linregress

        for t in range(1,int(maxiter)+1):
            self.sol.nit += 1
            # get the relative entropy value and its gradient w.r.t. variables
            dkl, grad_t = self.gradient(self.sol.x, rho, beta, batch_size=batch_size)
            print('\rIteration %d beta=%.3f' % (t,beta),end='')
            # Convergence status
            if np.linalg.norm(grad_t) < gtol:
                self.sol.success = True
                self.sol.message = 'Exceeded minimum gradient |grad| < %g' % gtol
                break
            #all_dkl.append(dkl)
            # check the convergence based on the slope of dkl of the last 10 iterations
            #if t > last_iters:
            #    dkl_iters = np.arange(last_iters)
            #    slope, intercept, r_value, p_value, std_err = linregress(x=dkl_iters, y=all_dkl[-last_iters:])
            #    if np.abs(slope - 1E-3) < 0:
            #        converged = True
            #        print('Optimization terminated for flatness')

            self.mt = beta1 * self.mt + (1.0 - beta1) * grad_t
            self.vt = beta2 * self.vt + (1.0 - beta2) * (grad_t * grad_t)
            mttilde = self.mt / (1.0 - (beta1 ** t))  # compute bias corrected first moment estimate
            vttilde = self.vt / (1.0 - (beta2 ** t))  # compute bias-corrected second raw moment estimate
            if quasi_hyperbolic: # quasi hyperbolic adam
                deltax = ((1-nu1) * grad_t + nu1 * mttilde)/(np.sqrt((1-nu2) * (grad_t**2) + nu2 * vttilde ) + epsilon)
            else: # vanilla Adam
                deltax = mttilde / np.sqrt(vttilde + epsilon)
            self.sol.x -= learning_rate * deltax
            self.logfile.write( u"%g\t%g\t%g\n" % (self.sol.x,beta,dkl) )
            if t % 50:
                self.logfile.flush()
        return self.sol

    # def run(self, **kwargs):
    #     x = self.x0
    #     batch_size = kwargs.get('batch_size', 1)
    #     max_iters = kwargs.get('max_iters', 1000)
    #     # ADAM parameters
    #     eta = kwargs.get('eta', 1E-3)
    #     gtol = kwargs.get('gtol', 1E-4)
    #     xtol = kwargs.get('xtol', 1E-3)
    #     beta1 = 0.9
    #     beta2 = 0.999
    #     epsilon = 1E-8 # avoid nan with large learning rates
    #     # Use quasi-hyperbolic adam by default
    #     # https://arxiv.org/pdf/1810.06801.pdf
    #     quasi_hyperbolic = kwargs.get('quasi_hyperbolic', True)
    #     nu1 = 0.7 # for quasi-hyperbolic adam
    #     nu2 = 1.0 # for quasi-hyperbolic adam
    #     # visualization options
    #     refresh_frames = kwargs.get('refresh_frames', 100)
    #     from drawnow import drawnow, figure
    #     figure(figsize=(8,8))
    #     import matplotlib.pyplot as plt
    #     #plt.figure(figsize=(8, 8))

    #     # Populate the solution list as function of beta
    #     # the list sol contains all optimization points
    #     sol = []
    #     # Iterate over all beta provided by the user
    #     mt, vt = np.zeros(self.x0.shape), np.zeros(self.x0.shape)
    #     all_dkl = []
        
    #     # TODO implement model boundaries in Adam
    #     frames = 0
    #     filename = 'adam.log'
    #     logfile = open(filename,'w')
    #     from scipy.stats import linregress
    #     import logging
    #     import sys
    #     logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    #     adam_logger = logging.getLogger('ADAM')
    #     adam_logger.setLevel(logging.INFO)

    #     for beta in self.beta_range:
    #         adam_logger.info('Changed beta to %g' % beta)
    #         # if rho is provided, user rho is used, otherwise is computed at every beta
    #         rho = kwargs.get('rho', compute_vonneuman_density(L=self.L, beta=beta))
    #         # initialize at x0
    #         x = self.x0
    #         converged = False
    #         t = 0  # t is the iteration number
    #         while not converged:
    #             t += 1
    #             # get the relative entropy value and its gradient w.r.t. variables
    #             dkl, grad_t = self.gradient(x, rho, beta, batch_size=batch_size)
    #             # Convergence status
    #             if np.linalg.norm(grad_t) < gtol:
    #                 converged = True
    #                 adam_logger.info('Exceeded minimum gradient |grad|<%g' % gtol)
                
    #             if t > max_iters:
    #                 adam_logger.info('Exceeded maximum iterations')
    #                 converged = True

    #             # check the convergence based on the slope of dkl of the last 10 iterations
    #             last_iters = 100
    #             if len(all_dkl) > last_iters:
    #                 dkl_iters = np.arange(last_iters)
    #                 slope, intercept, r_value, p_value, std_err = linregress(x=dkl_iters, y=all_dkl[-last_iters:])
    #                 if np.abs(slope - 1E-3) < 0:
    #                         converged = True
    #                         adam_logger.info('Optimization terminated for flatness')

    #             # TODO implement check boundaries in Adam
    #             # if np.any(np.ravel(self.model.bounds)):
    #             #    raise RuntimeError('variable bounds exceeded')
    #             mt = beta1 * mt + (1.0 - beta1) * grad_t
    #             vt = beta2 * vt + (1.0 - beta2) * (grad_t * grad_t)
    #             mttilde = mt / (1.0 - (beta1 ** t))  # compute bias corrected first moment estimate
    #             vttilde = vt / (1.0 - (beta2 ** t))  # compute bias-corrected second raw moment estimate
    #             if quasi_hyperbolic: # quasi hyperbolic adam
    #                 deltax = ((1-nu1) * grad_t + nu1 * mttilde)/(np.sqrt((1-nu2) * (grad_t**2) + nu2 * vttilde ) + epsilon)
    #             else: # vanilla Adam
    #                 deltax = mttilde / np.sqrt(vttilde + epsilon)
    #             x -= eta * deltax
    #             #all_dkl.append(dkl)
    #             #print(np.hstack([t, x,beta,dkl]))
    #             #logfile.write( u"%g\t%g\t%g\n" % (x,beta,dkl) )
    #             # if t % refresh_frames == 0:
    #             #     frames += 1
    #             #     def draw_fig():
    #             #         plot_beta_range = np.logspace(-3,3,100)
    #             #         sol.append({'x': x.copy()})
    #             #         #plt.figure(figsize=(8, 8))
    #             #         A0 = np.mean(self.model.sample_adjacency(theta=x, batch_size=batch_size), axis=0)
    #             #         plt.subplot(2, 2, 1)
    #             #         im = plt.imshow(self.A)
    #             #         plt.colorbar(im)
    #             #         plt.title('Data')
    #             #         plt.subplot(2, 2, 2)
    #             #         im = plt.imshow(A0)
    #             #         plt.colorbar(im)
    #             #         plt.title('<Model>')
    #             #         plt.subplot(2, 2, 3)
    #             #         plt.plot(all_dkl)
    #             #         plt.xlabel('iteration')
    #             #         plt.ylabel('$S(\\boldsymbol \\rho,\\boldsymbol \\sigma)$')
    #             #         plt.subplot(2, 2, 4)
    #             #         plt.semilogx(plot_beta_range, batch_compute_vonneumann_entropy(self.L, plot_beta_range), '.-', label='data')
    #             #         plt.semilogx(plot_beta_range, batch_compute_vonneumann_entropy(graph_laplacian(A0), plot_beta_range), '.-', label='model')
    #             #         plt.plot(beta, batch_compute_vonneumann_entropy(graph_laplacian(A0), [beta]), 'ko', label='model')
    #             #         plt.xlabel('$\\beta$')
    #             #         plt.ylabel('$S$')
    #             #         plt.title('Entropy')
    #             #         plt.legend(loc='best')
    #             #         plt.suptitle('$\\beta=$' + '{0:0>3}'.format(beta))
    #             #         #plt.tight_layout()
    #             #     drawnow(draw_fig)
    #     self.sol = sol
    #     adam_logger.info('Optimization finished\n' + str(self.sol))
    #     return sol

