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
    nb = int(n / b)
    A = (np.random.random((n, n)) < pout).astype(float)
    for i in range(0, b):
        T = np.triu((np.random.random((nb, nb)) < pin).astype(float))
        T = T + T.T
        A[i * nb:(i + 1) * nb, i * nb:(i + 1) * nb] = T

    np.fill_diagonal(A, 0)
    A = np.triu(A)
    A = A + A.T
    return A


def hierarchical_random_graph2(ers, nr):
    N = np.sum(nr)
    b = len(ers)
    nrns = np.reshape(np.kron(nr, nr), [b, b])
    M = ers / nrns
    A = np.zeros([N, N])
    idx = np.cumsum([0] + nr)
    for i in range(0, b):
        ri = np.array(range(idx[i], idx[i + 1] + 1))
        for j in range(0, b):
            rj = np.array(range(idx[j], idx[j + 1] + 1))
            R = np.random.random([len(ri) - 1, len(rj) - 1])
            #R = np.triu(R)
            #R += R.T
            A[ri.min():ri.max(), rj.min():rj.max()] = (nrns[i, j] * R) < ers[i, j]
    A = np.triu(A, 1)
    A += A.T
    return A


def hierarchical_random_graph(sigma2rs, nr):
    N = np.sum(nr)
    A = np.zeros([N, N])
    b = len(sigma2rs)
    idx = np.cumsum([0] + nr)
    for i in range(0, b):
        ri = np.array(range(idx[i], idx[i + 1] + 1))
        for j in range(0, b):
            rj = np.array(range(idx[j], idx[j + 1] + 1))
            R = np.random.random([len(ri) - 1, len(rj) - 1])
            A[ri.min():ri.max(), rj.min():rj.max()] = R < sigma2rs[i, j]
    A = np.triu(A, 1)
    A += A.T
    return A


def gammadiff(a, z0, z1):  # this version of Gamma does not have infinite recursion problems
    return mp.gammainc(a, z0) - mp.gammainc(a, z1)


def compute_tr(z, t0, ers, nr, matrix, tol=1E-9, maxsteps=50000):
    b = ers.shape[0]
    if len(nr) != b:
        raise Exception('Incompatible dimensions')
    # Their definition changes for every different submatrix
    sigma2rs, M = None, None
    nrns = np.reshape(np.kron(nr,nr),[b,b])
    er = ers.sum(axis=1)
    if matrix is 'laplacian':
        sigma2rs = ers/nrns
        M = -sigma2rs
    elif matrix is 'adjacency':
        sigma2rs = ers/nrns
        M = sigma2rs
    elif matrix is 'norm_laplacian':
        sigma2rs = ers/np.reshape(np.kron(er,er),[b,b])
        M = -ers/np.sqrt(np.multiply(nrns,ers))

    # Define the intrablock average degrees dr
    # since we work with undirected graphs
    dr = er/np.array(nr)

    Eq2 = [None] * b

    # This is the supercomplicated way to avoid shallow copy of lambda
    # functions around
    for r in range(0, b):
        if matrix is 'laplacian':
            Eq2[r] = lambda rr, T: (mp.power(dr[rr], (z - np.sum([sigma2rs[rr, ss] * nr[ss] * T[ss] for ss in range(0, b)]))) * gammadiff(np.sum([sigma2rs[rr, ss] * nr[ss] * T[ss] for ss in range(0, b)]) - z, 0, -dr[rr]) + T[
                                    rr] * mp.exp(dr[rr] + 1j * mp.pi * (np.sum([sigma2rs[rr, ss] * nr[ss] * T[ss] for ss in range(0, b)]) - z))) * mp.exp(-dr[rr] - 1j * mp.pi * (np.sum([sigma2rs[rr, ss] * nr[ss] * T[ss] for ss in range(0, b)]) - z))
        elif matrix is 'adjacency':
            Eq2[r] = lambda rr, T: T[rr] + 1 / (np.sum([ sigma2rs[rr, ss] * nr[ss] * T[ss] for ss in range(0, b) ]) - z)
        elif matrix is 'norm_laplacian':
            Eq2[r] = lambda rr, T: T[rr] + 1 / (np.sum([sigma2rs[rr, ss] * nr[ss] * T[ss] for ss in range(0, b)]) - z + 1)

    try:
        trsol = mp.findroot(lambda *t: [Eq2[rr](rr, [*t]) for rr in range(0, b)], x0=t0, solver='muller', maxsteps=maxsteps)
        return [trsol[r] for r in range(0, b)]
    except:
        raise Exception('Value of z was ' + str(z))
        return t0

############################################################
############################################################


def compute_detached_tr(ers, z0, t0, nr, matrix, eps=1E-7, maxsteps=50000):
    b = ers.shape[0]
    sigma2rs, M = None, None

    nrns = np.reshape(np.kron(nr,nr),[b,b])
    er = ers.sum(axis=1)
    if matrix is 'laplacian':
        sigma2rs = ers/nrns
        M = -sigma2rs
    elif matrix is 'adjacency':
        sigma2rs = ers/nrns
        M = sigma2rs
    elif matrix is 'norm_laplacian':
        sigma2rs = ers/np.reshape(np.kron(er,er),[b,b])
        M = -ers/np.sqrt(np.multiply(nrns,ers))

    IB = sp.eye(b)
    NB = sp.diag(*nr)

    def TzB(z):
        val = sp.diag(*compute_tr(z + eps * 1j, t0, ers, nr, matrix))
        return val

    def eq4(z):  # Equation 4 of Peixoto paper
        det = sp.simplify((IB - TzB(z) * M * NB).det())
        return mp.mpc(det)

    def trsum(z):
        tr = compute_tr(z + eps * 1j, t0, ers, nr, [sp.KroneckerDelta(sp.Symbol('c_', integer=True), 0)] * b)
        return np.sum([t.imag for t in tr])
    z_detached = mp.findroot(eq4, x0=mp.mpc(z0), tol=1E-9, solver='muller', maxsteps=maxsteps)
    return z_detached.real


def compute_rho(allz, ers, nr, matrix, eps=1E-7, maxsteps=150, include_isolated=False, t0=None):

    N = np.sum(nr)  # total number of nodes

    bar = ProgressBar(len(allz), max_width=40)

    b = ers.shape[0]
    if t0 is None:
        # use a default initial value for the stieltjes transform tr
        t0 = [mp.mpc('0')] * b

    print('Continuos band')
    io = open('data.dat','w')
    rho = np.zeros_like(allz)  # initialize rho as zeros
    for i, z in enumerate(allz):
        bar.numerator = i + 1
        t = compute_tr(z + eps * 1j, t0, ers, nr, matrix)
        for tt in t:
            io.write('%g %g ' % (float(tt.real),float(tt.imag)))
        io.write('\n')
        t0 = t # initialize from previous solution
        rho[i] = -(1.0 / (N * np.pi)) * np.sum([nr[r] * t[r].imag for r in range(0, b)])
        print('\r', bar, end='')
        #print('\r%s Percent done=%.1f %%\tz=%.2f\trho(z)=%g ' % (bar, float(i+1)/len(allz)*100, z, rho[i]), end='')
    io.close()
    # Then look for the detached eigenvalues by looking for the z such that it solves Eq.4=0
    # where rho is 0 look for separate eigenvalues
    if include_isolated:
        allzlist = allz.tolist()
        rholist = rho.tolist()
        print('\nDiscrete band')
        zi = 0
        for z in reversed(allz):
            if zi >= z:
                lastzi = zi
                zi = compute_detached_tr(ers, z, t0, nr, matrix, maxsteps=maxsteps)
                if np.abs(mp.re(lastzi) - mp.re(zi)) > eps:
                    allzlist.append(mp.re(zi))
                    # to avoid considering same solution twice
                    rholist.append(0.05)
                    print('\rfound isolated eigenvalue zi=%g' % (zi.real))

        return allzlist, rholist
    else:
        return allz, rho

# def test1(matrix):
#     b0 = 2
#     pin, pout = 0.8, 0.2
#     sigma2rs = pout*(np.ones(b0)-np.eye(b0)) + pin*np.eye(b0)
#     #M0 = np.array([[0.8,0.2],[0.3,0.5]])
#     #sigma2rs = sigma2rs#np.kron(M0, M0)

#     nr = [100,50]
#     b = len(nr)
#     nrns = np.reshape(np.kron(nr,nr),[b,b])
#     ers = np.multiply(sigma2rs,nrns)

#     # Empirical eigenvalues
#     reps = 50
#     if matrix is 'laplacian':
#         eigs = np.array([scipy.linalg.eigvalsh(GL(hierarchical_random_graph2(ers, nr))) for i in range(0, reps)]).flatten()
#     elif matrix is 'adjacency':
#         eigs = np.array([scipy.linalg.eigvalsh(hierarchical_random_graph(ers, nr)) for i in range(0, reps)]).flatten()
#     elif matrix is 'norm_laplacian':
#         eigs = np.array([scipy.linalg.eigvalsh(NGL(hierarchical_random_graph(ers, nr))) for i in range(0, reps)]).flatten()

#     allz, rho = compute_rho(np.linspace(-50, 50, 1000), sigma2rs, nr, matrix, eps=1E-12, t0=[1E-5j]*ers.shape[0])
#     #plt.figure()
#     #plt.hist(eigs, bins=200)
#     #plt.figure()
#     #plt.plot(allz, rho)

#     plt.figure()
#     plt.hist(eigs, density=1,bins=200)
#     plt.plot(allz, rho)
#     plt.show()


def test2(matrix):
    ers = np.array([[5000, 25000], [25000, 5000]])*5
    #ers = np.kron(ers,np.array([[0.5,0.3],[0.8,0.2]]))
    print(ers)
    nr = [512, 256]
    b = len(nr)
    nrns = np.reshape(np.kron(nr, nr), [b, b])
    reps = 500
    print(ers/nrns)
    if matrix is 'laplacian':
        eigs = np.array([scipy.linalg.eigvalsh(GL(hierarchical_random_graph2(ers, nr))) for i in range(0, reps)]).flatten()
    elif matrix is 'adjacency':
        eigs = np.array([scipy.linalg.eigvalsh(hierarchical_random_graph2(ers, nr)) for i in range(0, reps)]).flatten()
    elif matrix is 'norm_laplacian':
        eigs = np.array([scipy.linalg.eigvalsh(NGL(hierarchical_random_graph2(ers, nr))) for i in range(0, reps)]).flatten()

    zmin, zmax = eigs.min(), eigs.max()
    if matrix is not 'adjacency':
        zmax = -zmin
        zmin = 0

    plt.subplot(1, 3, 1)
    plt.imshow(hierarchical_random_graph2(ers, nr))
    plt.grid(False)
    plt.subplot(1, 3, 2)
    plt.imshow(ers)
    plt.grid(False)
    plt.subplot(1,3,3)
    plt.hist(eigs, density=1, bins=500)
    
    plt.show()
    np.savetxt('ers.dat',ers)
    np.savetxt('nrns.dat',nrns)
    np.savetxt('eigs.dat',eigs)
    allz, rho = compute_rho(np.linspace(0, 200, 10000), ers, nr, matrix, eps=mp.mpf(1E-7), t0=[mp.mpc('0')] * len(nr), include_isolated=False)
    # save for further analyses
    np.savetxt('allz.dat',allz)
    np.savetxt('rho.dat',rho)
    plt.hist(eigs, density=1, bins=2000)
    #plt.vlines(x=zmin, ymin=0, ymax=0.05, color='b')
    #plt.vlines(x=zmax, ymin=0, ymax=0.05, color='b')
    plt.plot(allz, np.abs(rho))
    plt.show()
    print()
    A=np.loadtxt('data.dat')
    for i in np.arange(0,2*len(nr),2):
        plt.plot(A[:,i],A[:,i+1])
    plt.show()


if __name__ == '__main__':
    test2('laplacian')