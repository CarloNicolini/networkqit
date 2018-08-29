#!/usr/bin/python3
import sys
sys.path.append('..')
import numpy as np
from networkqit import graph_laplacian as GL
from networkqit import normalized_graph_laplacian as NGL
import scipy.linalg
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import sympy as sp
import mpmath as mp
from etaprogress.progress import ProgressBar
    
def planted_partition_graph(n, b, pin, pout):
    nb = int(n/b)
    A = (np.random.random((n, n)) < pout).astype(float)
    for i in range(0, b):
        T = np.triu((np.random.random((nb, nb)) < pin).astype(float))
        T = T+T.T
        A[i*nb:(i+1)*nb, i*nb:(i+1)*nb] = T

    np.fill_diagonal(A, 0)
    A = np.triu(A)
    A = A+A.T
    return A


def hierarchical_random_graph(sigma2rs, nr):
    N = np.sum(nr)
    A = np.zeros([N, N])
    b = len(sigma2rs)
    idx = np.cumsum([0]+nr)
    for i in range(0, b):
        ri = np.array(range(idx[i], idx[i+1]+1))
        for j in range(0, b):
            rj = np.array(range(idx[j], idx[j+1]+1))
            A[ri.min():ri.max(), rj.min():rj.max()] = np.random.random(
                [len(ri)-1, len(rj)-1]) < sigma2rs[i, j]
    A = np.triu(A, 1)
    A += A.T
    return A

def Gamma(a, z0, z1): # this version of Gamma does not have infinite recursion problems
    return mp.gammainc(a, z0) - mp.gammainc(a, z1)

def compute_tr(z, t0, sigma2, nr, matrix, tol=1E-16, maxsteps=100):
    b = len(sigma2)
    if len(nr) != b:
        raise 'sigma2rs and nr must have same dimensions'

    # Define the variables $t_r(z)$ which are the Stieltjes transforms to look for
    #tr = [sp.symbols('t'+str(i), complex=True) for i in range(0, b)]
    
    # nrns = np.array([[(nr[r]*(nr[s]-1)) if r == s else nr[r]*nr[s] for s in range(0, b)] for r in range(0, b)])
    # ers = np.multiply(M, nrns)  # elementwise multiplication
    # Define the intrablock average degrees dr
    dr = np.diagonal(np.diag(nr)*sigma2)
    def S(r, T):
        val = np.sum([sigma2[r, s]*nr[s]*T[s] for s in range(0, b)])
        return mp.mpc(val)
    Eq2 = []
    for r in range(0, b):
        #S = mp.mpfnp.sum([sigma2[r, s]*nr[s]*tr[s] for s in range(0, b)])
        fund_eq = None # initialize to none
        #S = lambda *T : mp.mpc(np.sum([sigma2[r, s]*nr[s]*T[s] for s in range(0, b)])) # convert to mpmath
        if matrix is 'laplacian':
            def fund_eq(r,T):
                return ( mp.power(dr[r],(z-S(r,T)))*Gamma(S(r,T)-z,0,-dr[r])+T[r]*mp.exp(dr[r]+1j*mp.pi*(S(r,T)-z)))*mp.exp(-dr[r]-1j*mp.pi*(S(r,T)-z))
            #fund_eq = lambda *T : ( mp.power(dr[r],(z-S(r,T)))*Gamma(S(r,T)-z,0,-dr[r])+T[r]*mp.exp(dr[r]+1j*mp.pi*(S(r,T)-z)))*mp.exp(-dr[r]-1j*mp.pi*(S(r,T)-z))
        elif matrix is 'adjacency':
            #fund_eq = tr[r] + 1/(S-z)
            fund_eq = lambda T : T[r]+mp.mpc(1/(S(r,T)-z))
        elif matrix is 'norm_laplacian':
            fund_eq = lambda *T : T[r]+mp.mpc(1/(S(*T)-z+1))
        # Append in the list of equations to be solved simulateously
        Eq2.append(fund_eq)
        
    print(Eq2[0](mp.mpc(1j),mp.mpc(1j)))
    print(mp.findroot(lambda x,y : [ Eq2[0](x,y),Eq2[1](x,y) ],x0=[1,2] ))
    # Now look for the solutions with respect to [t0,t1,...,tb] at this given z
    #trsol = mp.findroot(Eq2, x0=[mp.mpc(tr) for tr in t0], solver='muller', tol=tol, maxsteps=maxsteps)
    #return [trsol[r] for r in range(0, b)]

############################################################
############################################################
def compute_detached_tr(M, z0, t0, nr, matrix, eps=1E-12, maxsteps=100):
    b = M.shape[0]
    if len(nr) != b:
        raise 'Incompatible size of matrix M and total number of blocks'
    IB = sp.eye(b)
    NB = sp.diag(*nr)

    def TzB(z):
        val = sp.diag(*compute_tr(z+eps*1j, t0, M, nr, matrix))
        return val

    def eq4(z):  # Equation 4 of Peixoto paper
        det = sp.simplify((IB - TzB(z)*M*NB).det())
        return mp.mpc(det)

    def trsum(z):
        tr = compute_tr(z+eps*1j, t0, M, nr, [sp.KroneckerDelta(sp.Symbol('c_', integer=True), 0)] * b)
        return np.sum([t.imag for t in tr])
    z_detached = mp.findroot(lambda z: eq4(z), x0=mp.mpc(
        z0), solver='muller', verbose=False, maxsteps=maxsteps)
    return z_detached

def compute_rho(allz, M, nr, matrix, eps=1E-12, maxsteps=150, include_isolated=False, t0=None):

    N = np.sum(nr)  # total number of nodes
    
    bar = ProgressBar(len(allz), max_width=40)

    allzlist = allz.tolist()
    rho = np.zeros_like(allz)  # initialize rho as zeros
    rholist = rho.tolist()
    
    b = M.shape[0]
    if t0 is None:
        # use a default initial value for the stieltjes transform tr
        t0 = [0 + 0.001j] * b

    # symbolic variable to sum over in the fundamental equation
    
    print('Continuos band')
    for i, z in enumerate(allz):
        bar.numerator = i + 1
        # print('\r',bar,end='')
        t = compute_tr(z + eps*1j, t0, M, nr, matrix)
        t0 = t
        rho[i] = -(1.0 / (N*np.pi)) * np.sum([nr[r]*(t[r].imag) for r in range(0, b)])
        print('\r%s Percent done=%.1f %%\tz=%.2f\trho(z)=%g' % (bar, float(i+1)/len(allz)*100, z, rho[i]), end='')

    # Then look for the detached eigenvalues by looking for the z such that it solves Eq.4=0
    # where rho is 0 look for separate eigenvalues
    if include_isolated:
        zi = None
        print('\nDiscrete band')
        zi = mp.mpc(np.min(allz))
        for z in allz:
            if mp.re(zi) <= z:
                lastzi = zi
                zi = compute_detached_tr(M, z, [1E-5j] * b, nr, matrix, maxsteps=maxsteps)
                if np.abs(mp.re(lastzi)-mp.re(zi)) > eps:
                    allzlist.append(mp.re(zi))
                    # to avoid considering same solution twice
                    rholist.append(0.05)
                    print('found isolated eigenvalue zi=%g' % (zi.real))

        return allzlist, rholist
    else:
        return allz, rho

def poisspdf(k,l):
    return sp.exp(-l)*(k**l)/sp.factorial(l)
    
def test1():
    b0 = 2
    pin, pout = 0.8, 0.2

    M0 = pout*(np.ones(b0)-np.eye(b0)) + pin*np.eye(b0)
    #M0 = np.array([[0.8,0.2],[0.3,0.5]])
    M = M0 #np.kron(M0, M0)
    nr = [100, 100]
    #b = len(nr)
    reps = 50

    #eigs_a = np.array([scipy.linalg.eigvalsh(hierarchical_random_graph(M, nr)) for i in range(0, reps)]).flatten()
    eigs_l = np.array([scipy.linalg.eigvalsh(GL(hierarchical_random_graph(M, nr))) for i in range(0, reps)]).flatten()
    #eigs_nl = np.array([scipy.linalg.eigvalsh(NGL(hierarchical_random_graph(M, nr))) for i in range(0, reps)]).flatten()
    plt.hist(eigs_l, density=1, bins=200)
    plt.show()
    #allz, rho = compute_rho(np.linspace(0, 110, 10), M, nr, 'laplacian', eps=1E-12, maxsteps=2000)
    #print('')
    #plt.hist(eigs_l, density=1, bins=200)
    #plt.plot(allz, rho)
    #plt.show()


if __name__ == '__main__':
    #print('done')
    compute_tr(100,[1+0.01j]*2,np.array([[0.8,0.2],[0.2,0.8]]),[100,100],'laplacian')
    #test1()
