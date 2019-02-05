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
    
        \\frac{\\partial S(\\rho \\| \\sigma(\\mathbb{E}_{\\theta}[L]))}{\\partial \\theta_k} = \\beta \\textrm{Tr}\\biggl \\lbrack \\left(\\rho - \\sigma(\\mathbb{E}_{\\theta}[L])\\right)\\frac{\\partial \\mathbb{E}_{\\theta}[L]}{\\partial \\theta_k} \\biggr \\rbrack

In the `StochasticOptimizer` class we instead address the issue to implement stochastic gradient descent methods.
In these methods the gradients are defined as:

.. math::
   
   \\frac{\\partial \\mathbb{E}_{\\theta}[S(\\rho \\| \\sigma)]}{\\partial \\theta_k} = \\beta \\textrm{Tr}\\biggl \\lbrack \\rho \\frac{\\partial  \\mathbb{E}_{\\theta}[L]}{\\partial \\theta_k}\\biggr \\rbrack + \\frac{\\partial}{\\partial \\theta_k}\\mathbb{E}_{\\theta}\\biggl \\lbrack \\log \\left( \\textrm{Tr}\\left \\lbrack e^{-\\beta L(\\theta)} \\right \\rbrack \\right) \\biggr \\rbrack

The stochastic optimizer is the **correct** optimizer, as it makes no approximation on the Laplacian eigenvalues.
It is more suitable for small graphs and intermediate $\\beta$, where the differences between the random matrix s
pectrum and its expected counterpart are non-neglibile.
For large and dense enough graphs however the `ContinuousOptimizer` works well and yields deterministic results,
as the optimization landscape is smooth.

In order to minimize the expected relative entropy then, we need both the expected Laplacian formula, which is simple
to get, and a way to estimate the second summand in the gradient, that involves averaging over different realizations
of the log trace of  $e^{-\\beta L(\\theta)}$.
A good the approximation to the expected logtrace $\\mathbb{E}_{\\theta}[\\log \\textrm{Tr}[\\exp{(-\\beta L)}]]$ is,
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
from scipy.optimize import minimize, least_squares
from networkqit.graphtheory import *
from networkqit.infotheory.density import *
import logging
logger = logging.getLogger('optimize')
logger.setLevel(logging.INFO)


class ModelOptimizer:
    def __init__(self, **kwargs):
        """
        This represents the base abstract class from which to inherit all the possible model optimization classes
        """
        super().__init__()

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

    def __init__(self, G: np.array, x0: np.array, model: GraphModel, **kwargs):
        """
        Init the optimizer.

        Args:
            G (numpy.array) :is the empirical network to study. A N x N adjacency matrix as numpy.array.
            x0 (numpy.array): is the k-element array of initial parameters estimates. Typically set as random.
        """
        self.G = G
        self.x0 = x0
        self.model = model
        self.sol = None
        super().__init__(**kwargs)

    def run(self, **kwargs):
        """
        Maximimize the likelihood of the model given the observed network G.

        Kwargs:
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

            maxfun:     (int) set the maximum number of evaluations of the cost.
                        Default value is very large ten thousand (1E4)

            maxiter:    (int) set the maximum number of iteration. Default 10.000 iterations
            verbose:    (int) Verbosity level. No output=0, 2=iterations output.

            basin_hopping: (bool) whether to run global optimization with local
                optimization by least_squares (only for saddle point method)

            basin_hopping_niter: (int) number of basin hopping repetitions

        Outputs:
            sol: (scipy.optimize.OptimizeResult) parameters at optimal likelihood
        """

        opts = {'ftol': kwargs.get('ftol', 1E-10),
                'gtol': kwargs.get('gtol', 1E-5), # as default of LBFGSB
                'eps': kwargs.get('eps', 1E-8), # as default of LBFGSB
                'xtol': kwargs.get('xtol', 1E-8),
                'maxfun': kwargs.get('maxfun', 1E10),
                'maxiter': kwargs.get('maxiter', 1E4),
                'verbose' : kwargs.get('verbose', 2),
                'disp':  bool(kwargs.get('verbose', 2)),
                'iprint': kwargs.get('verbose', 1)
                }

        if kwargs.get('method', 'MLE') is 'MLE':
            # The model has non-linear constraints, must use Sequential Linear Square Programming SLSQP
            from autograd import jacobian
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

            print(self.sol['message'])
            if self.sol['status'] != 0:
                RuntimeWarning(self.sol['message'])
                #raise Exception('Method did not converge to maximum likelihood: ')
            return self.sol

        elif kwargs.get('method', 'saddle_point') is 'saddle_point':
            if hasattr(self.model, 'constraints'):
                raise RuntimeError('Cannot solve saddle_point_equations with non-linear constraints')
            # Use the Dogbox method to optimize likelihood
            ls_bounds = [np.min(np.ravel(self.model.bounds)), np.max(np.ravel(self.model.bounds))]
            def basin_opt(*basin_opt_args, **basin_opt_kwargs):
                opt_result = least_squares(fun=basin_opt_args[0],
                                           x0=np.squeeze(basin_opt_args[1]),
                                           bounds=ls_bounds,
                                           method='dogbox',
                                           xtol=opts['xtol'],
                                           gtol=opts['gtol'],
                                           max_nfev=opts['maxfun'],
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
                xmin = np.zeros_like(self.x0) + np.finfo(float).eps
                xmax = xmin + np.inf  # broadcast infinity
                bounds = BHBounds(xmin=xmin, xmax=np.inf)
                bounded_step = BHRandStepBounded(xmin, xmax, stepsize=0.5)
                self.sol = basinhopping(func=lambda z: (nlineq(z)**2).sum(),
                                        x0=np.squeeze(self.x0),
                                        T=kwargs.get('T', 1),
                                        minimize_routine=basin_opt,
                                        minimizer_kwargs={'saddle_point_equations': nlineq },
                                        accept_test=bounds,
                                        take_step=bounded_step,
                                        niter=kwargs.get('basin_hopping_niter', 5),
                                        disp=bool(kwargs.get('verbose')))
            return self.sol


class ContinuousOptimizer(ModelOptimizer):
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
        
        :math:`\\frac{s(\\rho \| \\sigma)}{\\partial \\theta_k} = \\beta \textrm{Tr}\left \lbrack \left( \rho - \sigma(\theta)\right) \frac{\mathbb{E}\mathbf{L}(\theta)}{\partial \theta_k} \right \rbrack`
        
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


class StochasticOptimizer(ModelOptimizer):
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

    def __init__(self, A, x0, beta_range, model, **kwargs):
        self.A = A
        self.L = graph_laplacian(self.A)
        self.x0 = x0
        self.beta_range = beta_range
        self.model = model

    def gradient(self, x, rho, beta, batch_size=1):
        # Compute the relative entropy between rho and sigma, using tensors with autograd support
        # Entropy of rho is constant over the iterations
        # if beta is kept constant, otherwise a specific rho can be passed
        #lambd_rho = eigh(rho)[0]
        #entropy = -np.sum(lambd_rho * np.log(lambd_rho))

        def log_likelihood(z):
            N = len(self.A)
            # advanced broadcasting here!
            # Sample 'batch_size' adjacency matrices shape=[batch_size,N,N]
            Amodel = self.model.sample_adjacency(z, batch_size=batch_size)
            #print('Amodel nan?:', np.any(np.isnan(Amodel.ravel())))
            # Here exploit broadcasting to create batch_size diagonal matrices with the degrees
            Dmodel = np.eye(N) * np.transpose(np.zeros([1, 1, N]) + np.einsum('ijk->ik', Amodel), [1, 0, 2])
            Lmodel = Dmodel - Amodel  # returns a batch_size x N x N tensor
            # do average over batches of the sum of product of matrix elements (done with * operator)
            Emodel = np.mean(np.sum(np.sum(Lmodel * rho, axis=2), axis=1))
            lambd_model = eigh(Lmodel)[0]  # eigh is a batched operation, take the eigenvalues only
            Fmodel = - np.mean(logsumexp(-beta * lambd_model, axis=1) / beta)
            loglike = beta * (Emodel - Fmodel) # - Tr[rho log(sigma)]
            dkl = loglike # - entropy # Tr[rho log(rho)] - Tr[rho log(sigma)]
            return dkl

        # value and gradient of relative entropy as a function
        dkl_and_dkldx = autograd.value_and_grad(log_likelihood)
        return dkl_and_dkldx(x)


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
        epsilon = 1E-8 # avoid nan with large learning rates
        # Use quasi-hyperbolic adam by default
        # https://arxiv.org/pdf/1810.06801.pdf
        quasi_hyperbolic = kwargs.get('quasi_hyperbolic', True)
        nu1 = 0.7 # for quasi-hyperbolic adam
        nu2 = 1.0 # for quasi-hyperbolic adam
        # visualization options
        refresh_frames = kwargs.get('refresh_frames', 100)
        #from drawnow import drawnow, figure
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))

        # Populate the solution list as function of beta
        # the list sol contains all optimization points
        sol = []
        # Iterate over all beta provided by the user
        mt, vt = np.zeros(self.x0.shape), np.zeros(self.x0.shape)
        all_dkl = []
        # TODO implement model boundaries in Adam
        frames = 0
        for beta in self.beta_range:
            logger.info('Changed beta to %g' % beta)
            # if rho is provided, user rho is used, otherwise is computed at every beta
            rho = kwargs.get('rho', compute_vonneuman_density(L=self.L, beta=beta))
            # initialize at x0
            x = self.x0
            converged = False
            t = 0  # t is the iteration number
            while not converged:
                t += 1
                # get the relative entropy value and its gradient w.r.t. variables
                dkl, grad_t = self.gradient(x, rho, beta, batch_size=batch_size)
                # Convergence status
                if np.linalg.norm(grad_t) < gtol:
                    converged = True
                    logger.info('Exceeded minimum gradient |grad|<%g' % gtol)
                if t > max_iters:
                    logger.info('Exceeded maximum iterations')
                    converged = True
                # TODO implement check boundaries in Adam
                # if np.any(np.ravel(self.model.bounds)):
                #    raise RuntimeError('variable bounds exceeded')
                mt = beta1 * mt + (1.0 - beta1) * grad_t
                vt = beta2 * vt + (1.0 - beta2) * (grad_t * grad_t)
                mttilde = mt / (1.0 - (beta1 ** t))  # compute bias corrected first moment estimate
                vttilde = vt / (1.0 - (beta2 ** t))  # compute bias-corrected second raw moment estimate
                if quasi_hyperbolic: # quasi hyperbolic adam
                    deltax = ((1-nu1) * grad_t + nu1 * mttilde)/(np.sqrt((1-nu2) * (grad_t**2) + nu2 * vttilde ) + epsilon)
                else: # vanilla Adam
                    deltax = mttilde / np.sqrt(vttilde + epsilon)
                x -= eta * deltax
                all_dkl.append(dkl)
                if t % refresh_frames == 0:
                    frames += 1
                    def draw_fig():
                        sol.append({'x': x.copy()})
                        plt.figure(figsize=(8, 8))
                        A0 = np.mean(self.model.sample_adjacency(x, batch_size=batch_size), axis=0)
                        plt.subplot(2, 2, 1)
                        plt.imshow(self.A)
                        plt.title('Data')
                        plt.subplot(2, 2, 2)
                        plt.imshow(A0)
                        plt.title('<Model>')
                        plt.subplot(2, 2, 3)
                        plt.plot(all_dkl)
                        plt.xlabel('iteration')
                        plt.ylabel('$S(\\rho,\\sigma)$')
                        plt.subplot(2, 2, 4)
                        plt.semilogx(self.beta_range, batch_compute_vonneumann_entropy(self.L, np.logspace(3,-3,100)), '.-', label='data')
                        plt.semilogx(self.beta_range,
                                     batch_compute_vonneumann_entropy(graph_laplacian(A0), np.logspace(3,-3,100)), '.-', label='model')
                        plt.plot(beta, batch_compute_vonneumann_entropy(graph_laplacian(A0), [beta]), 'ko', label='model')
                        plt.xlabel('$\\beta$')
                        plt.ylabel('$S$')
                        plt.title('Entropy')
                        plt.legend(loc='best')
                        plt.suptitle('$\\beta=$' + '{0:0>3}'.format(beta))
                        #plt.tight_layout()
                        print('frame_' + '{0:0>5}'.format(frames) + '.png')
                        plt.savefig('frame_' + '{0:0>5}'.format(frames) + '.png')
                        plt.cla()
                    draw_fig()
        self.sol = sol
        return sol

