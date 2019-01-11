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

from abc import ABC, abstractmethod
import numdifftools as nd
import numpy as np
from scipy.linalg import eigvalsh
from scipy.misc import logsumexp
from scipy.optimize import minimize, least_squares, fsolve
from networkqit.graphtheory import *
from networkqit.graphtheory import graph_laplacian as graph_laplacian
from networkqit.infotheory.density import VonNeumannDensity, SpectralDivergence, compute_vonneuman_density


class ModelOptimizer(ABC):
    def __init__(self, A, **kwargs):
        """
        This represents the base abstract class from which to inherit all the possible model optimization classes
        """
        super().__init__()

    @abstractmethod
    def setup(self, **kwargs):
        """
        The setup method must be implemented by each single inherited class.
        Here all the details and additional optimization variables are inserted as kwargs.
        """
        pass

    @abstractmethod
    def gradient(self, **kwargs):
        """
        The gradient method must be implemented by each single inherited class
        Here all the details and additional optimization variables are inserted as kwargs.
        The gradient method muust return a NxNxk array where N is the dimension of the network
        to optimize and k is the number of free parameters of the model.
        """
        pass

    @abstractmethod
    def run(self, **kwargs):
        """
        The  method to call to start optimization
        """
        pass


###############################################
## Standard maximum likelihood optimization ###
###############################################
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

    def gradient(self, **kwargs):
        """
        Not implemented for this class
        """
        pass

    def setup(self, **kwargs):
        """
        Not implemented for this class
        """
        pass

    def run(self, model, **kwargs):
        """
        Optimize the likelihood of the model given the observation A. 
        """
        eps = np.finfo(float).eps
        bounds = ((eps) * len(self.A), (np.inf) * len(self.x0))
        # Use the Trust-Region reflective algorithm to maximize loglikelihood
#        self.sol = least_squares(fun=lambda z : -model.loglikelihood(A,z), x0=np.squeeze(self.x0),
#                                 method='dogbox',
#                                 bounds=bounds,
#                                 xtol=kwargs.get('xtol', 1E-12),
#                                 gtol=kwargs.get('gtol', 1E-12),
#                                 max_nfev=kwargs.get('max_nfev', len(self.x0) * 100000),
#                                 verbose=kwargs.get('verbose', 1))

        opts = {'ftol':kwargs.get('ftol', 1E-20),
                'xtol':kwargs.get('xtol', 1E-20)}
        self.sol = minimize(fun=lambda z : -model.loglikelihood(A,z), x0=np.squeeze(self.x0),
                            method='SLSQP',
                            bounds=model.bounds,
                            options=opts)

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
                                       verbose=kwargs.get('verbose',1))
            # use this form with linear loss as in the basinhopping
            # func argument to be consistent
            opt_result['fun'] = 0.5*np.sum(opt_result['fun']**2)
            return opt_result

        # If the model is convex, we simply run least_squares
        if kwargs.get('basinhopping',False)==False:
            self.sol = basin_opt(lambda z : model.saddle_point(self.G, z), self.x0)
        else: # otherwise combine local and global optimization with basinhopping
            from .basinhoppingmod import basinhopping, BHBounds, BHRandStepBounded
            xmin = np.zeros_like(self.x0) + 1E-9 #np.finfo(float).eps
            xmax = xmin + np.inf # broadcast infinity
            bounds = BHBounds(xmin = xmin)
            bounded_step = BHRandStepBounded(xmin, xmax, stepsize=0.5)
            self.sol = basinhopping(func = lambda z: 0.5*(model.saddle_point(self.G, z)**2).sum(),
                                    x0 = np.squeeze(self.x0),
                                    T=kwargs.get('T',1),
                                    minimize_routine = basin_opt,
                                    minimizer_kwargs = {'saddle_point_equations': lambda z : model.saddle_point(self.G, z)},
                                    accept_test = bounds,
                                    take_step = bounded_step,
                                    niter = kwargs.get('basin_hopping_niter',5),
                                    disp = bool(kwargs.get('verbose')))
        return self.sol


####################################
## Spectral entropy optimization ###
####################################
class ExpectedModelOptimizer(ModelOptimizer):
    """
    This class is at the base of the continuos optimization method
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

    def setup(self, expected_adj_fun, expected_lapl_grad_fun=None, step_callback=None):
        """
        Setup the optimizer. Must specify the model function, as a function that returns the expected adjacency matrix,
        Optionally one can also provide the modelfun_grad, a function that returns the gradient of the expected Laplacian.

        args:
            expected_adj_fun: a function in the form f(θ) that once called returns the expected adjacency matrix of a model E_θ[A].
            expected_lapl_grad_fun: a function in the form f(θ) that once called returns the gradients of the expected Laplacian of a model ∇_θ[E_θ[A]]. 
            step_callback: a callback function to control the current status of optimization.
        """
        self.modelfun = expected_adj_fun
        self.modelfun_grad = expected_lapl_grad_fun
        self.step_callback = step_callback
        self.bounds = expected_adj_fun.bounds

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
        sigma = compute_vonneuman_density(graph_laplacian(self.modelfun(x)), beta)
        # Here instead of tracing over the matrix product, we just sum over the entrywise product of the two matrices
        # (rho-sigma) and the ∂E_θ[L]/.
        return np.array(
            [np.sum(np.multiply(rho - sigma, self.modelfun_grad(x)[:, i, :].T)) for i in range(0, len(self.x0))])

    def hessian(self, x, beta):
        """
        If required by an optimization algorithms, here we relyi on the numdifftools Python3 library
        to compute the Hessian of the model at given parameters x and beta
        """
        import numdifftools as nd
        H = nd.Hessian(lambda y: SpectralDivergence(
            Lobs=self.L, Lmodel=graph_laplacian(self.modelfun(y)), beta=beta).rel_entropy)(x)
        return H  # (x)

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
            self.rel_entropy_fun = lambda x: SpectralDivergence(Lobs=self.L, Lmodel=graph_laplacian(self.modelfun(x)),
                                                                beta=beta).rel_entropy

            # Define initially fgrad as None, relying on numerical computation of it
            fgrad = None

            # Define observed density rho as none initially, if necessary it is computed
            rho = None

            # If user provides gradients of the model, use them, redefyining fgrad to pass to solver
            if self.modelfun_grad is not None:
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
            spect_div = SpectralDivergence(Lobs=self.L, Lmodel=graph_laplacian(self.modelfun(sol[-1].x)), beta=beta)
            sol[-1]['DeltaL'] = (np.trace(self.L) - np.trace(graph_laplacian(self.modelfun(sol[-1].x)))) / 2
            if kwargs.get('compute_sigma', False):
                Lmodel = graph_laplacian(self.modelfun(sol[-1].x))
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
            sol[-1]['AIC'] = 2 * len(self.modelfun.args_mapping) - 2 * sol[-1]['loglike']
            for i in range(0, len(self.modelfun.args_mapping)):
                sol[-1][self.modelfun.args_mapping[i]] = sol[-1].x[i]
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
            s = "{:>20} " * (len(self.modelfun.args_mapping) + 1)
            print('=' * 20 * (len(self.modelfun.args_mapping) + 1))
            print('Summary:')
            print('Model:\t' + str(self.modelfun.formula))
            print('=' * 20 * (len(self.modelfun.args_mapping) + 1))
            print('Optimization method: ' + self.method)
            print('Variables bounds: ')
            for i, b in enumerate(self.bounds):
                left = '-∞' if self.bounds[i][0] is None else str(
                    self.bounds[i][0])
                right = '+∞' if self.bounds[i][1] is None else str(
                    self.bounds[i][1])
                print("{: >1} {:} {: >10} {:} {: >1}".format(
                    left, '<=', self.modelfun.args_mapping[i], '<=', right))
            print('=' * 20 * (len(self.modelfun.args_mapping) + 1))
            print('Results:')
            print('=' * 20 * (len(self.modelfun.args_mapping) + 1))
            print('Von Neumann Log Likelihood:\t' +
                  str(self.sol[-1]['loglike']))
            print('Von Neumann Entropy:\t\t' + str(self.sol[-1]['entropy']))
            print('Von Neumann Relative entropy:\t' +
                  str(self.sol[-1]['rel_entropy']))
            print('AIC:\t\t\t\t' + str(self.sol[-1]['AIC']))
            print('=' * 20 * (len(self.modelfun.args_mapping) + 1))
            print('Estimate:')
            print('=' * 20 * (len(self.modelfun.args_mapping) + 1))
            print(s.format('beta', *self.modelfun.args_mapping))
            for i in range(0, len(self.sol)):
                row = [str(x) for x in self.sol[i].x]
                print(s.format(self.sol[i]['beta'], *row))


