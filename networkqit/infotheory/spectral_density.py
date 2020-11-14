"""
Method and functions based on the information theory of networks
This file contains function to estimate the spectral density of random modular
networks (adjacency, laplacian, normalized laplacian)
"""

import numpy as np
from networkqit.graphtheory.matrices import graph_laplacian as GL
from networkqit.graphtheory.matrices import normalized_graph_laplacian as NGL
import sympy as sp
import mpmath as mp
from networkqit.graphtheory.matrices import sbm


def gammadiff(
    a, z0, z1
):  # this version of Gamma does not have infinite recursion problems
    return mp.gammainc(a, z0) - mp.gammainc(a, z1)


def compute_tr(z, t0, ers, nr, matrix, tol=1e-9, maxsteps=50000):
    """
    Solves a system of self-consistent equations, given the eigenvalue z
    to get the real and imaginary part of the non-linear function t(z)

    Args:
        z (float): the eigenvalue we are looking the value of t(z) for
        t0 (float): an initial estimate of t(z), which has to be iterated
        ers (nd.array): the square numpy array with the number of links between modules r and s
        nr (list or nd.array): the list of number of nodes in each block
        tol (float): the termination tolerance of the findroot method
        maxsteps (int): maximum number of iterations of the mpmath findroot method
    Output:
        t(z) (float): the value of the function t(z), to be used in the calculation
        of the spectral density of either the adjacency, laplacian or norm. laplacian
    """

    b = ers.shape[0]  # number of blocks, ers must be square
    if len(nr) != b:
        raise Exception("Incompatible dimensions")
    # Their definition changes for every different submatrix
    sigma2rs, M = None, None
    nrns = np.reshape(np.kron(nr, nr), [b, b])
    er = np.sum(ers, axis=1)  # reduced one dimension
    if matrix is "laplacian":
        sigma2rs = ers / nrns
        M = -sigma2rs
    elif matrix is "adjacency":
        sigma2rs = ers / nrns
        M = sigma2rs
    elif matrix is "norm_laplacian":
        sigma2rs = ers / np.reshape(np.kron(er, er), [b, b])
        M = -ers / np.sqrt(np.multiply(nrns, ers))

    # Define the intrablock average degrees dr
    # since we work with undirected graphs
    dr = [er[i] / nr[i] for i in range(b)]

    Eq2 = [None] * b

    # This is the supercomplicated way to avoid shallow copy of lambda
    # functions around
    for r in range(0, b):
        if matrix is "laplacian":
            #  to get this equation use sympy and solve symbolically
            Eq2[r] = lambda rr, T: (
                mp.power(
                    dr[rr],
                    (
                        z
                        - np.sum(
                            [sigma2rs[rr, ss] * nr[ss] * T[ss] for ss in range(0, b)]
                        )
                    ),
                )
                * gammadiff(
                    np.sum([sigma2rs[rr, ss] * nr[ss] * T[ss] for ss in range(0, b)])
                    - z,
                    0,
                    -dr[rr],
                )
                + T[rr]
                * mp.exp(
                    dr[rr]
                    + 1j
                    * mp.pi
                    * (
                        np.sum(
                            [sigma2rs[rr, ss] * nr[ss] * T[ss] for ss in range(0, b)]
                        )
                        - z
                    )
                )
            ) * mp.exp(
                -dr[rr]
                - 1j
                * mp.pi
                * (
                    np.sum([sigma2rs[rr, ss] * nr[ss] * T[ss] for ss in range(0, b)])
                    - z
                )
            )
        elif matrix is "adjacency":
            Eq2[r] = lambda rr, T: T[rr] + 1 / (
                np.sum([sigma2rs[rr, ss] * nr[ss] * T[ss] for ss in range(0, b)]) - z
            )
        elif matrix is "norm_laplacian":
            Eq2[r] = lambda rr, T: T[rr] + 1 / (
                np.sum([sigma2rs[rr, ss] * nr[ss] * T[ss] for ss in range(0, b)])
                - z
                + 1
            )

    try:
        trsol = mp.findroot(
            lambda *t: [Eq2[rr](rr, [*t]) for rr in range(0, b)],
            x0=t0,
            solver="muller",
            maxsteps=maxsteps,
        )
        return [trsol[r] for r in range(0, b)]
    except:
        raise Exception("findroot did not converge. Last value of z was %g" % z)
        return t0


def compute_detached_tr(ers, z0, t0, nr, matrix, eps=1e-7, maxsteps=50000):
    """
    Gets an estimate of the eigenvalues which are detached from the bulk spectrum

    Args:
        ers (nd.array): the numpy array with the number of links between modules r and s
        z0 (float): the initial eigenvalue around which we are looking the detached one
        t0 (float): an initial estimate of t(z), which has to be iterated
        nr (list or nd.array): the list of number of nodes in each block
        matrix (str): can be 'adjacency', 'norm_laplacian', 'laplacian'
        eps (float): a termination tolerance
        maxsteps (int): maximum number of iterations of the Muller solver.
    """
    b = ers.shape[0]
    sigma2rs, M = None, None

    nrns = np.reshape(np.kron(nr, nr), [b, b])
    er = ers.sum(axis=1)
    if matrix is "laplacian":
        sigma2rs = ers / nrns
        M = -sigma2rs
    elif matrix is "adjacency":
        sigma2rs = ers / nrns
        M = sigma2rs
    elif matrix is "norm_laplacian":
        sigma2rs = ers / np.reshape(np.kron(er, er), [b, b])
        M = -ers / np.sqrt(np.multiply(nrns, ers))

    IB = sp.eye(b)
    NB = sp.diag(*nr)

    def TzB(z):
        val = sp.diag(*compute_tr(z + eps * 1j, t0, ers, nr, matrix))
        return val

    def eq4(z):  # Equation 4 of Peixoto paper
        det = sp.simplify((IB - TzB(z) * M * NB).det())
        return mp.mpc(det)

    def trsum(z):
        tr = compute_tr(
            z + eps * 1j,
            t0,
            ers,
            nr,
            [sp.KroneckerDelta(sp.Symbol("c_", integer=True), 0)] * b,
        )
        return np.sum([t.imag for t in tr])

    z_detached = mp.findroot(
        eq4, x0=mp.mpc(z0), tol=1e-9, solver="muller", maxsteps=maxsteps
    )
    return z_detached.real


def compute_rho(
    allz, ers, nr, matrix, eps=1e-7, maxsteps=150, include_isolated=False, t0=None
):
    """
    Estimates the spectral density of a random matrix ensemble of modular graphs.
    First an initial estimate of t(z) is done on, and the estimate is propagated
    'adiabatically' to the next point. In this way solver needs less steps than
    starting from random for every eigenvalue in allz.
    The denser the array allz, the closer the continuos bulk of the spectral
    density is to the correct one, at the cost of increased computation time.

    Args:
        allz (list or nd.array): the eigevalues range to calculate the spectral density of.
        ers (nd.array): the numpy array with the number of links between modules r and s
        nr (list or nd.array): the list of number of nodes in each block
        matrix (str): can be 'adjacency', 'norm_laplacian', 'laplacian'
        eps (float): a termination tolerance
        maxsteps (int): maximum number of iterations of the Muller solver.
        include_isolated (bool): whether to compute detached eigenvalues
        t0 (float): initial estimate of t(z) for a given z0
    """

    N = np.sum(nr)  # total number of nodes

    b = ers.shape[0]
    if t0 is None:
        # use a default initial value for the stieltjes transform tr
        t0 = [mp.mpc("0")] * b

    print("Continuos band")
    io = open("allt.dat", "w")
    rho = np.zeros_like(allz)  # initialize rho as zeros
    from tqdm import tqdm

    for i, z in tqdm(enumerate(allz), ascii=True):
        t = compute_tr(z + eps * 1j, t0, ers, nr, matrix)
        for tt in t:
            io.write("%g\t%g\t" % (float(tt.real), float(tt.imag)))
        io.write("\n")
        t0 = t  # initialize from previous solution
        rho[i] = -(1.0 / (N * np.pi)) * np.sum([nr[r] * t[r].imag for r in range(0, b)])
        # print('\r%s Percent done=%.1f %%\tz=%.2f\trho(z)=%g ' % (bar, float(i+1)/len(allz)*100, z, rho[i]), end='')
    io.close()

    # Then look for the detached eigenvalues by looking for the z such that it solves Eq.4=0
    # where rho is 0 look for separate eigenvalues
    if include_isolated:
        allzlist = allz.tolist()
        rholist = rho.tolist()
        print("\nDiscrete band")
        zi = 0
        for z in reversed(allz):
            if zi >= z:
                lastzi = zi
                zi = compute_detached_tr(ers, z, t0, nr, matrix, maxsteps=maxsteps)
                if np.abs(mp.re(lastzi) - mp.re(zi)) > eps:
                    allzlist.append(mp.re(zi))
                    # to avoid considering same solution twice
                    rholist.append(0.05)
                    print("\rfound isolated eigenvalue zi=%g" % (zi.real))

        return allzlist, rholist
    else:
        return allz, rho


def adiabatic_rho(z, ers, nr, matrix, eps):
    z0 = mp.mpc(0 + eps * 1j)
    t0 = [mp.mpc("0")] * len(nr)
    N = np.sum(nr)  # total number of nodes
    b = ers.shape[0]
    while mp.fabs(z0 - z) > 1e-3:
        t = compute_tr(z0 + eps * 1j, t0, ers, nr, matrix)
        t0 = t
        z0 += 1
        print(
            z0,
            z,
            -(1.0 / (N * np.pi)) * np.sum([nr[r] * t0[r].imag for r in range(0, b)]),
        )
    tf = t0
    rho = -(1.0 / (N * np.pi)) * np.sum([nr[r] * tf[r].imag for r in range(0, b)])
    return rho


def expected_partition_function(ers, nr, beta, reps):
    Z = 0
    for i in range(0, reps):
        L = GL(sbm(ers, nr))
        Z += np.exp(-beta * scipy.linalg.eigvalsh(L)).sum()
    Z /= reps
    return Z


def analytical_partition_function(
    ers,
    nr,
    beta,
):
    """
    Compute the ensemble averaged partition function analytically by numerical
    quadrature methods.

    """
    f = lambda z: rho_z(z, ers, nr, "laplacian", t0=[mp.mpc("0")] * len(nr))
    I = mp.quad(f, [0, mp.inf])
    return I * np.sum(nr)


def benchmark(matrix):
    ers = np.array([[5000, 25000], [25000, 5000]])
    ers = np.kron(ers, np.array([[1, 0.8], [0.8, 0.5]]) * 2)
    nr = [512, 256] * 2
    b = len(nr)
    nrns = np.reshape(np.kron(nr, nr), [b, b])
    reps = 5

    if matrix is "laplacian":
        eigs = np.array(
            [scipy.linalg.eigvalsh(GL(sbm(ers, nr))) for i in range(0, reps)]
        ).flatten()
    elif matrix is "adjacency":
        eigs = np.array(
            [scipy.linalg.eigvalsh(sbm(ers, nr)) for i in range(0, reps)]
        ).flatten()
    elif matrix is "norm_laplacian":
        eigs = np.array(
            [scipy.linalg.eigvalsh(NGL(sbm(ers, nr))) for i in range(0, reps)]
        ).flatten()
    return eigs, ers, nr, nrns


def test2(matrix):
    eigs, ers, nr, nrns = benchmark(matrix)

    #    plt.subplot(1, 3, 1)
    #    plt.imshow(sbm(ers, nr))
    #    plt.grid(False)
    #    plt.subplot(1, 3, 2)
    #    plt.imshow(ers)
    #    plt.grid(False)
    #    plt.subplot(1,3,3)
    #    plt.hist(eigs, density=1, bins=500)
    #    plt.show()

    np.savetxt("ers.dat", ers)
    np.savetxt("nrns.dat", nrns)
    np.savetxt("eigs.dat", eigs)
    allz, rho = compute_rho(
        np.linspace(0, 400, 1000),
        ers,
        nr,
        matrix,
        eps=mp.mpf(1e-7),
        t0=[mp.mpc("0")] * len(nr),
        include_isolated=False,
    )
    # save for further analyses
    np.savetxt("allz.dat", allz, fmt="%g")
    np.savetxt("rho.dat", rho, fmt="%g")
    plt.hist(eigs, density=1, bins=200)
    # plt.vlines(x=zmin, ymin=0, ymax=0.05, color='b')
    # plt.vlines(x=zmax, ymin=0, ymax=0.05, color='b')
    plt.plot(allz, np.abs(rho))
    plt.show()

    print()


#    A=np.loadtxt('data.dat')
#    for i in np.arange(0,2*len(nr),2):
#        plt.plot(A[:,i],A[:,i+1])
#    plt.show()


def test3(matrix):
    eigs, ers, nr, nrns = benchmark(matrix)

    plt.hist(eigs, density=1, bins=500)

    beta = 1e-2
    f = lambda z: rho_z(z, ers, nr, "laplacian", t0=[mp.mpc("0")] * len(nr))
    print(f(200))
    # print('E[Tr[exp(-bL)]]=', expected_partition_function(ers,nr,beta,reps))
    # print('Integral=', analytical_partition_function(ers,nr,beta))
