#!/usr/bin/env python
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


Finally, the `MLEOptimizer` maximizes the standard likelihood of a model and it is not related to the spectral entropies framework introduced in the paper on which `networkqit` is based.

"""
#    Copyright (C) 2018 by
#    Carlo Nicolini <carlo.nicolini@iit.it>
#    All rights reserved.
#    BSD license.

from abc import ABC, abstractmethod
import numpy as np
from numpy import triu, nan_to_num, log, inf
from scipy.linalg import expm, logm, eigvalsh
from scipy.optimize import minimize, least_squares, fsolve
from networkqit.infotheory.density import VonNeumannDensity, SpectralDivergence, compute_vonneuman_density
from networkqit.graphtheory import graph_laplacian as graph_laplacian
from scipy.misc import logsumexp
import numdifftools as nd


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


################################################
## Stochastic optimzation from random samples ##
################################################
class StochasticOptimizer(ModelOptimizer):
    """
    This class is at the base of possible implementation of methods based
    on stochastic gradient descent. Here not implemented. 
    The idea behind this class is to help the user in designing a nice stochastic gradient descent method,
    such as ADAM, AdaGrad or older methods, like the Munro-Robbins stochastic gradients optimizer.
    Working out the expression for the gradients of the relative entropy, one remains with the following:

    :math: `\nabla_{\theta}S(\rho \| \sigma) = \beta \textrm\biggl \lbrack \rho \nabla_{\theta}\mathbb{E}_{\theta}[L]} \biggr \rbrack`
        
    :math: `\frac{\partial S(\rho \| \sigma)}{\partial \theta_k} = \beta \textrm{Tr}\lbrack \rho \frac{\partial}{\partial \theta_k} \rbrack + \frac{\partial}{\partial \theta_k}\mathbb{E}_{\theta}\log \textrm{Tr} e^{-\beta L(\theta)}\lbrack \rbrack`
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


    def gradient(self, x, rho, beta):
        """
        This method must be defined from inherited classes.
        It returns the gradients of the expected relative entropy at given parameters x.
        The expectation can either be estimated numerically by repeated sampling or via some smarter methods.
        """
        pass


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


    def gradient(self, x, rho, beta,num_samples=1):
        # Compute the first part of the gradient, the one depending linearly on the expected laplacian, easy to get
        grad = np.array([beta*np.sum(np.multiply(rho,self.modelfun_grad(x)[:,i,:])) for i in range(0,len(self.x0))])
        # Now compute the second part, dependent on the gradient of the expected log partition function
        def estimate_gradient_log_trace(x, rho, beta, num_samples=1):
            logZ = lambda y : logsumexp(-beta*eigvalsh(graph_laplacian(self.samplingfun(y))))
            meanlogZ = lambda w: np.mean([ logZ(w) for i in range(0,num_samples)])
            return nd.Gradient(meanlogZ)(x)
        gradlogtrace = estimate_gradient_log_trace(x,rho,beta)
        return grad + gradlogtrace

    def run(self,**kwargs):
        x = self.x0
        num_samples = kwargs.get('num_samples',1)
        clip_gradients = kwargs.get('clip_gradients',None)
        max_iters = kwargs.get('max_iters',1000)       
        eta = kwargs.get('eta',1E-3)
        tol = kwargs.get('tol',1E-5)
        # Populate the solution list as function of beta
        # the list sol contains all optimization points
        sol = []
        # Iterate over all beta provided by the user
        for beta in self.beta_range:
            rho = VonNeumannDensity(A=None, L=self.L, beta=beta).density
            converged = False
            t = 0
            while not converged:
                t += 1
                grad_t = None
                if clip_gradients is None:
                    grad_t = self.gradient(x,rho,beta)
                else:
                    grad_t = np.clip(self.gradient(x,rho,beta),clip_gradients[0],clip_gradients[1]) # clip the gradients
                x_old = x.copy()
                x -= eta/t*grad_t
                if self.step_callback is not None:
                    self.step_callback(beta,x)
<<<<<<< HEAD
                if t > max_iters or np.linalg.norm(x_old-x) < tol:
=======
                if t > max_iters:
>>>>>>> 560adf2cb92cc0a150f692ea358e474aa25566f3
                    break
            sol.append({'x':x})
            # Here creates the output data structure as a dictionary of the optimization parameters and variables
            spect_div = SpectralDivergence(Lobs=self.L, Lmodel=graph_laplacian(self.samplingfun(sol[-1]['x'])), beta=beta)
            sol[-1]['DeltaL'] = (np.trace(self.L) - np.trace(graph_laplacian(self.samplingfun(sol[-1]['x']))))/2
            if kwargs.get('compute_sigma',False):
                Lmodel = graph_laplacian(self.modelfun(sol[-1]['x']))
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
            #for i in range(0, len(self.modelfun.args_mapping)):
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


    def gradient(self, x, rho, beta,num_samples=1):
        # Compute the first part of the gradient, the one depending linearly on the expected laplacian, easy to get
        grad = np.array([beta*np.sum(np.multiply(rho,self.modelfun_grad(x)[:,i,:])) for i in range(0,len(self.x0))])
        # Now compute the second part, dependent on the gradient of the expected log partition function
        def elogz(x):
            elogz = 0
            for i in range(0,num_samples):
                L = graph_laplacian(self.samplingfun(x))
                l = eigvalsh(L)
                elogz += logsumexp(-beta*l)/num_samples
            return elogz
        
        nablaelogz = nd.Gradient(elogz)
        return grad + nablaelogz(x)
                
    def run(self,**kwargs):
        x = self.x0
        num_samples = kwargs.get('num_samples',1)
        clip_gradients = kwargs.get('clip_gradients',None)
        max_iters = kwargs.get('max_iters',1000)
        alpha = kwargs.get('alpha',1E-3)
        tol = kwargs.get('tol',1E-5)
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1E-8
        
        # Populate the solution list as function of beta
        # the list sol contains all optimization points
        sol = []
        # Iterate over all beta provided by the user
        mt,vt = np.zeros(self.x0.shape),np.zeros(self.x0.shape)
        all_x = []
        for beta in self.beta_range:
            rho = VonNeumannDensity(A=None, L=self.L, beta=beta).density
            converged = False
            t = 0
            while not converged:
                t += 1
                grad_t = None
                if clip_gradients is None:
                    grad_t = self.gradient(x,rho,beta,num_samples)
                else:
                    grad_t = np.clip(self.gradient(x,rho,beta),clip_gradients[0],clip_gradients[1]) # clip the gradients
                mt = beta1 * mt + (1.0-beta1) * grad_t
                vt = beta2 * vt  + (1.0-beta2) * grad_t*grad_t
                mttilde = mt/(1.0-(beta1**t)) # compute bias corrected first moment estimate
                vttilde = vt/(1.0-(beta2**t)) # compute bias-corrected second raw moment estimate
                x_old = x.copy()
                x -= alpha * mttilde / np.sqrt(vttilde + epsilon)
                if t > max_iters or np.linalg.norm(x_old-x) < tol:
                    break
                if self.step_callback is not None:
                    self.step_callback(beta,x)
                all_x.append(x[0])
            sol.append({'x':x.copy()})
            # Here creates the output data structure as a dictionary of the optimization parameters and variables
            spect_div = SpectralDivergence(Lobs=self.L, Lmodel=graph_laplacian(self.samplingfun(sol[-1]['x'])), beta=beta)
            sol[-1]['DeltaL'] = (np.trace(self.L) - np.trace(graph_laplacian(self.samplingfun(sol[-1]['x']))))/2
            sol[-1]['T'] = 1/beta
            sol[-1]['beta'] = beta
            sol[-1]['loglike'] = spect_div.loglike
            sol[-1]['rel_entropy'] = spect_div.rel_entropy
            sol[-1]['entropy'] = spect_div.entropy
            sol[-1]['AIC'] = 2 * len(self.modelfun.args_mapping) - 2 * sol[-1]['loglike']
            for i in range(0, len(self.modelfun.args_mapping)):
                sol[-1][self.modelfun.args_mapping[i]] = sol[-1]['x'][i]
                
            if kwargs.get('compute_sigma',False):
                Lmodel = graph_laplacian(self.modelfun(sol[-1]['x']))
                rho = VonNeumannDensity(A=None, L=self.L, beta=beta).density
                sigma = VonNeumannDensity(A=None, L=Lmodel, beta=beta).density
                sol[-1]['<DeltaL>'] = np.trace(np.dot(rho,Lmodel)) - np.trace(np.dot(sigma,Lmodel))
            
            
        self.sol = sol
        return sol
