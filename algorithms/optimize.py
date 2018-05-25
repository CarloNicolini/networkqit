#!/usr/bin/env python
"""

Define the base and inherited classes for model optimization, both in the continuous approximation
and for the stochastic optimization.

"""
#    Copyright (C) 2018 by
#    Carlo Nicolini <carlo.nicolini@iit.it>
#    All rights reserved.
#    BSD license.

from abc import ABC, abstractmethod
from scipy.optimize import minimize, least_squares, fsolve
from networkqit.infotheory.density import VonNeumannDensity, SpectralDivergence, compute_vonneuman_density
from networkqit.graphtheory import graph_laplacian as graph_laplacian
import numpy as np
from numpy import triu, nan_to_num, log, inf
import numdifftools as nd
#from networkqit.graphtheory.models.GraphModel import *


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

    
################################################
## Stochastic optimzation from random samples ##
################################################

class StochasticOptimizer(ModelOptimizer):
    def __init__(self, A, x0, beta_range, **kwargs):
        """
        This class is at the base of possible implementation of methods based
        on stochastic gradient descent. Here not implemented. 
        The idea behind this class is to help the user in designing a nice stochastic gradient descent method,
        such as ADAM, AdaGrad or older methods, like the Munro-Robbins stochastic gradients optimizer.

        """
        pass

    def gradient(self, x, rho, beta):
        pass

    def setup(self, expected_adj, modelfun_grad=None, step_callback=None):
        pass

    def run(self, model, **kwargs):
        pass
    

class RobbinsMonro(StochasticOptimizer):
    """
    Implements the Robbins Monro optimizer described in the paper:
    A stochastic optimization approach to coarse-graining using a relative-entropy framework
    by Ilias Bilionis and Nicholas Zabaras
    The Journal of Chemical Physics  138, 044313 (2013); doi: 10.1063/1.4789308
    Available at:
    https://www.predictivesciencelab.org/uploads/4/8/9/6/48962987/bilionis_2013a.pdf
    """
    def __init__(self, A, x0, beta_range, **kwargs):
        self.A = A
        self.x0 = x0
        self.beta_range = beta_range

    def setup(self, expected_adj, modelfun_grad=None, step_callback=None):
        """
        Setup the optimizer. Must specify the model function, as a function that returns the expected adjacency matrix,
        Differently from the expected model optimizer here we need not only the expected Laplacian, but also the expectation
        of the log(Tr(exp(-betaL))). This can only be compu

        args:
            modelfun: a function in the form f(x) that once called returns the expected adjacency matrix of a model.
            modelfun_grad: a function in the form f(x) that once called returns the gradients of the expected Laplacian of a model.
            step_callback: a callback function to control the current status of optimization.
        """
        self.modelfun = modelfun
        self.modelfun_grad = modelfun_grad
        self.step_callback = step_callback
        self.bounds = modelfun.bounds


    def gradient(self, x, rho, beta):
        pass
        
    def run(self,**kwargs):
        num_iters = 100
        x = self.x0
        alpha_k = np.array([alpha/((A+k)**r) for k in range(0,num_iters)])
        for k in range(0,num_iters):
            x = x - alpha[k]

###############################################
## Standard maximum likelihood optimization ###
###############################################

class MLEOptimizer(ModelOptimizer):
    """
    This class, inheriting from the model optimizer class solves the problem of 
    maximum likelihood parameters estimation in the classical settings.
    """
    def __init__(self, A, x0, **kwargs):
        """
        Init the optimizer.
        
        Args:
            A (numpy.array) :is the empirical network to study. A N x N adjacency matrix as numpy.array. Can be weighted or unweighted.
            x0 (numpy.array): is the k-element array of initial parameters estimates. Typically set as random.    
        """
        self.A = A
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
        n = len(self.x0) #  number of parameters of the model is recovered by the size of x0
        eps = np.finfo(float).eps
        bounds = ((0) * len(self.A), (inf) * len(self.x0))
        A = (self.A > 0).astype(float) # binarize the input adjacency matrix
        W = self.A
        
        # See the definition here:
        # Garlaschelli, D., & Loffredo, M. I. (2008). Maximum likelihood: Extracting unbiased information from complex networks. 
        # PRE 78(1), 1–4. https://doi.org/10.1103/PhysRevE.78.015101
        def likelihood(x):
            pijx = model(x)
            one_min_pij = 1.0 - pijx
            one_min_pij[one_min_pij <= 0] = eps
            l = triu(W * (log(pijx) - log(one_min_pij)) + log(one_min_pij), 1).sum()
            return l

        # Use the Trust-Region reflective algorithm to optimize likelihood
        self.sol = least_squares(fun=likelihood, x0=np.squeeze(self.x0),
                                 method='trf',
                                 bounds=bounds,
                                 xtol=kwargs.get('xtol', 2E-10),
                                 gtol=kwargs.get('gtol', 2E-10),
                                 max_nfev=kwargs.get('max_nfev',len(self.x0)*100000))
        return self.sol


    def runfsolve(self, **kwargs):
        """
        Alternative method to estimate model parameters based on 
        the solution of a non-linear system of equation.
        The gradients with respect to the parameters are set to zero.
        
        kwargs keys:
            UBCM (string): specify the undirected binary configuration model. xixj/(1-xixj)
            UWCM (string): specify the undirected weighted configuration model. (yiyj)/(1-yiyj)
        """
        n = len(self.x0)
        eps = np.finfo(float).eps
        bounds = ((eps) * len(self.A), (inf) * len(self.x0))
        pij = None
        A = (self.A > 0).astype(float)
        W = self.A
        f = None
        if kwargs['model'] is 'UBCM':
            kstar = A.sum(axis=0)
            M = UBCM(N=len(A))
            f = lambda x : np.abs(kstar - M(x).sum(axis=0))
        elif kwargs['model'] is 'UWCM':
            sstar = W.sum(axis=0)
            M = UWCM(N=len(self.A))
            f = lambda x : np.abs(sstar - M(x).sum(axis=0))
        elif kwargs['model'] is 'UECM':
            kstar = A.sum(axis=0)
            sstar = W.sum(axis=0)
            M = UECM(N=len(A))
            f = lambda x : np.abs(np.hstack( [ kstar - M.adjacency(*x).sum(axis=0), sstar - M.adjacency_weighted(*x).sum(axis=0) ] ))
        
        # Use the Trust-Region reflective algorithm to optimize likelihood
        self.sol = fsolve(func=f, x0=np.squeeze(self.x0), xtol=1E-16)

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
        self.bounds = modelfun.bounds

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
        return np.array([np.sum(np.multiply(rho-sigma, self.modelfun_grad(x)[:, i, :].T)) for i in range(0, len(self.x0))])

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
            self.rel_entropy_fun = lambda x : SpectralDivergence(Lobs=self.L, Lmodel=graph_laplacian(self.modelfun(x)), beta=beta).rel_entropy
            
            # Define initially fgrad as None, relying on numerical computation of it
            fgrad = None
            
            # Define observed density rho as none initially, if necessary it is computed
            rho = None
            
            # If user provides gradients of the model, use them, redefyining fgrad to pass to solver
            if self.modelfun_grad is not None:
                # compute rho for each beta, just once
                rho = VonNeumannDensity(A=None, L=self.L, beta=beta).density
                # define the gradient of the Dkl, given rho at current beta
                fgrad = lambda x : self.gradient(x, rho, beta)
            
            # Append the solutions
            # Here we adopt two different strategies, either minimizing the residual of gradients of Dkl
            # using least_squares or minimizing the Dkl itself.
            # The least_squares approach requires the gradients of the model, if they are not implemented 
            # in the model, the algorithmic differentiation is used, which is slower
            if self.method is 'least_squares': 
                if rho is None: # avoid recomputation of rho
                    rho = VonNeumannDensity(A=None, L=self.L, beta=beta).density
                sol.append( least_squares(lambda x : self.gradient(x,rho,beta), x0=self.x0, bounds=kwargs.get('bounds',(0,np.inf)),
                                      loss=kwargs.get('loss','soft_l1'), # robust choice for the loss function of the residuals
                                      xtol = kwargs.get('xtol',1E-9),
                                      gtol = kwargs.get('gtol',1E-10)))
            else: # otherwise directly minimize the relative entropy function
                sol.append(minimize(fun=self.rel_entropy_fun,
                                    x0=self.x0,
                                    jac=fgrad,
                                    method=self.method,
                                    options={'disp': kwargs.get(
                                        'disp', False), 'gtol': kwargs.get('gtol', 1E-12)},
                                    bounds=self.bounds))
            
            # important to reinitialize from the last solution
            if kwargs.get('reinitialize',False):
                self.x0 = sol[-1].x
            # Call the step_callback function to print or display current solution
            if self.step_callback is not None:
                self.step_callback(beta, sol[-1].x)
            
            # Here creates the output data structure as a dictionary of the optimization parameters and variables
            spect_div = SpectralDivergence(Lobs=self.L, Lmodel=graph_laplacian(self.modelfun(sol[-1].x)), beta=beta)
            sol[-1]['DeltaL'] = (np.trace(self.L) - np.trace(graph_laplacian(self.modelfun(sol[-1].x))))/2
            if kwargs.get('compute_sigma',False):
                from scipy.linalg import expm
                Lmodel = graph_laplacian(self.modelfun(sol[-1].x))
                rho = VonNeumannDensity(A=None, L=self.L, beta=beta).density
                sigma = VonNeumannDensity(A=None, L=Lmodel, beta=beta).density
                sol[-1]['<DeltaL>'] = np.trace(np.dot(rho,Lmodel)) - np.trace(np.dot(sigma,Lmodel))
                #from scipy.integrate import quad
                #from numpy import vectorize
                #Q1 = vectorize(quad)(lambda x : expm(-x*beta*self.L)@(self.L-Lmodel)@expm(x*beta*self.L),0,1)
                #sol[-1]['<DeltaL>_1'] = beta*( np.trace(rho@Q1@Lmodel) - np.trace(rho@Q1)*np.trace(rho@Lmodel) )
            sol[-1]['T'] = 1/beta
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
            print(s.format('beta', * self.modelfun.args_mapping))
            for i in range(0, len(self.sol)):
                row = [str(x) for x in self.sol[i].x]
                print(s.format(self.sol[i]['beta'], *row))
