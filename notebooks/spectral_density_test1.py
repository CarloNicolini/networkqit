import sys
sys.path.append('..')
import numpy as np
from numpy.random import random as rand

import networkqit as nq
import networkx as nx

from networkqit import graph_laplacian as GL

import scipy.optimize
import scipy.linalg

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

import sympy as sp
import sympyprinting as spp
from sympy import init_printing

import mpmath as mp
import networkx as nx
init_printing()

def planted_partition_graph(n,b,pin,pout):
    nb = int(n/b)
    A = (np.random.random((n,n)) < pout).astype(float)
    for i in range(0,b):
        T = np.triu((np.random.random((nb,nb))<pin).astype(float))
        T = T+T.T
        A[i*nb:(i+1)*nb,i*nb:(i+1)*nb] = T
        
    np.fill_diagonal(A,0)
    A = np.triu(A)
    A = A+A.T
    return A


def compute_tr(z, t0, sigma2, nr, pr_, tol=1E-16, maxsteps=100):
    b = sigma2.shape[0]
    if len(nr) != b:
        raise 'sigma2rs and nr must have same dimensions'
    
    # Define the variables $t_r(z)$ which are the Stieltjes transforms to look for
    tr_ = [ sp.symbols('t'+str(i),complex=True) for i in range(0,b)]
    Eq2 = []
    for r in range(0,b):
        c_ = list(pr_[r].free_symbols)[0]
        sum_sigma2rs_ns_ts = np.array([sigma2[r,s]*nr[s]*tr_[s] for s in range(0,b)]).sum()
        #print(sum_sigma2rs_ns_ts)
        fund_eq = tr_[r] -  sp.summation( pr_[r]/(z - c_ - sum_sigma2rs_ns_ts ) , (c_,0,sp.oo))
        # Simplify the fundamental Equation
        fund_eq = sp.simplify(fund_eq)
        # Generate its lambdified expression, for evaluation with mpmath
        F = sp.lambdify(tr_,fund_eq.replace('exp_polar','exp'), modules='mpmath')
        # Append in the list of equations to be solved simulateously
        Eq2.append(F)
    # Now look for the solutions with respect to [t0,t1,...,tb] at this given z
    t = mp.findroot(Eq2, x0=t0, solver='muller', tol=tol, maxsteps=maxsteps)    
    return [t[i] for i in range(0,b)]

def compute_detached_tr(M, z0, t0, nr, eps=1E-12,maxsteps=100):
    IB = sp.eye(b)
    NB = sp.diag(*nr)
    
    def TzB(z):
        val = sp.diag(*compute_tr(z+eps*1j, t0, M, nr, [sp.KroneckerDelta(sp.Symbol('c',integer=True),0)] * b))
        return val
    
    def eq4(z): # Equation 4 of Peixoto paper
        det = sp.simplify((IB - TzB(z)*M*NB).det())
        return mp.mpc(det)
    
    def trsum(z):
        tr = compute_tr(z+eps*1j, t0, M, nr, [sp.KroneckerDelta(sp.Symbol('c',integer=True),0)] * b)
        return np.sum([t.imag for t in tr])
    z_detached =  mp.findroot(lambda z : eq4(z), x0= mp.mpc(z0), solver='muller', verbose=False, maxsteps=maxsteps)
    return z_detached

def compute_rho(allz, M, nr, matrix, eps=1E-12, maxsteps=150, include_isolated=False):
    allzlist = allz.tolist()
    rho = np.zeros_like(allz) # initialize rho as zeros
    rholist = rho.tolist()
    t0 = [0 + 0.01j] * b # initial value for the stieltjes transform tr
    pr = None
    c = sp.Symbol('c',integer=True) # symbolic variable to sum over in the fundamental equation
    N = np.sum(nr) # total number of nodes
    dr = M.sum()/np.array(nr) # average within module degree
    print(dr)
    # Define the pr variable depending on the matrix to investigate
    if matrix is 'laplacian':
        pr = [sp.exp(-dr[r]) * (dr[r]**c) / sp.gamma(c + 1) for r in range(0,b)]
    elif matrix is 'adjacency':
        pr = [sp.KroneckerDelta(c,0) for i in range(0,b)]
    elif matrix is 'norm_laplacian':
        pr = [sp.KroneckerDelta(c,1) for i in range(0,b)]
    print(pr)
    print('Continuos band')
    for i,z in enumerate(allz):
        t = compute_tr(z + eps*1j, t0, M, nr, pr)
        t0 = t
        rho[i] = -(1.0 / (N*np.pi)) * np.sum([nr[r]*(t[r].imag) for r in range(0,b)])
        print('\rPercent done=%.1f %%\tz=%.2f\trho(z)=%g\t%s' % (float(i+1)/len(allz)*100,z,rho[i],t), end='')

    # Then look for the detached eigenvalues by looking for the z such that it solves Eq.4=0
    # where rho is 0 look for separate eigenvalues
    if include_isolated:
        zi,ti = None,None
        print('\nDiscrete band')
        zi = mp.mpc(np.min(allz))
        for z in allz:
            if mp.re(zi) <= z:
                lastzi = zi
                zi = compute_detached_tr(M, z, [1E-5j]*b, nr, maxsteps=maxsteps)
                if np.abs(mp.re(lastzi)-mp.re(zi)) > eps:
                    allzlist.append(mp.re(zi))
                    rholist.append(0.05) # to avoid considering same solution twice
                    print('found isolated eigenvalue zi=%g' % (zi.real))

        return allzlist, rholist
    else:
        return allz, rho


if __name__=='__main__':
    b = 2
    nr = [ 100 ] * b
    pin, pout = 1.0, 0.2

    M = pout*(np.ones(b)-np.eye(b)) + pin*np.eye(b)
    
    eigs_l,eigs_a = [], []
    for i in range(0,50):
        print('\r',i,end='')
        A = planted_partition_graph(np.sum(nr),b,pout,pin)
        L=np.diag(A.sum(axis=0))-A
        eigs_a.append(scipy.linalg.eigvalsh(A))
        eigs_l.append(scipy.linalg.eigvalsh(L))
    eigs_a = np.array(eigs_a).flatten()
    eigs_l = np.array(eigs_l).flatten()

    allz, rho = compute_rho(np.linspace(-30,30,200), M, nr, 'adjacency', eps=1E-12, maxsteps=50)
    
    plt.hist(eigs_a,200)
    plt.plot(allz,rho)
    plt.show()