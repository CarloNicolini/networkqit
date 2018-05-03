#!/usr/bin/python

from abc import ABC, abstractmethod
from scipy.optimize import minimize, least_squares, fsolve
from networkqit.infotheory.density import VonNeumannDensity, SpectralDivergence, compute_vonneuman_density
from networkqit.graphtheory import graph_laplacian as graph_laplacian
import numpy as np
from numpy import triu, nan_to_num, log, inf
import numdifftools as nd
from networkqit.graphtheory.models.GraphModel import *

class ModelOptimizer(ABC):
    def __init__(self, A, **kwargs):
        super().__init__()

    @abstractmethod
    def setup(self, **kwargs):
        pass

    @abstractmethod
    def gradient(self, **kwargs):
        pass

    @abstractmethod
    def run(self, **kwargs):
        pass

    
################################################
## Stochastic optimzation from random samples ##
################################################

class StochasticOptimizer(ModelOptimizer):
        def __init__(self, A, x0, beta_range, **kwargs):
            pass
    
###############################################
## Standard maximum likelihood optimization ###
###############################################


class MLEOptimizer(ModelOptimizer):
    def __init__(self, A, x0, **kwargs):
        self.A = A
        self.x0 = x0

    def gradient(self, **kwargs):
        pass

    def setup(self, **kwargs):
        pass

    def run(self, model, **kwargs):
        n = len(self.x0)
        eps = np.finfo(float).eps
        bounds = ((0) * len(self.A), (inf) * len(self.x0))
        A = (self.A > 0).astype(float)
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


class ContinuousModelOptimizer(ModelOptimizer):
    def __init__(self, A, x0, beta_range, **kwargs):
        self.A = A
        self.L = graph_laplacian(A)
        self.beta_range = beta_range
        self.x0 = x0

    def setup(self, modelfun, modelfun_grad=None, step_callback=None):
        self.modelfun = modelfun
        self.modelfun_grad = modelfun_grad
        self.step_callback = step_callback
        self.bounds = modelfun.bounds

    def gradient(self, x, rho, beta):
        sigma = compute_vonneuman_density(
            graph_laplacian(self.modelfun(x)), beta)
        print(self.modelfun_grad(x))
        #return np.array([np.trace(np.dot(rho-sigma, self.modelfun_grad(x)[:, i, :])) for i in range(0, len(self.x0))])
        return np.array([np.sum(np.multiply(rho-sigma, self.modelfun_grad(x)[:, i, :].T)) for i in range(0, len(self.x0))])

    def hessian(self, x, beta):
        import numdifftools as nd
        H = nd.Hessian(lambda y: SpectralDivergence(
            Lobs=self.L, Lmodel=graph_laplacian(self.modelfun(y)), beta=beta).rel_entropy)(x)
        return H  # (x)

    def run(self, **kwargs):
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
            else: # otherwise minimize the Dkl function
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
