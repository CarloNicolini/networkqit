import numpy as np
from scipy.linalg import expm, logm
#from networkqit.utils.matfun import expm
from scipy.linalg import eigvalsh
from scipy.stats import entropy
from networkqit.graphtheory.matrices import graph_laplacian


def compute_vonneuman_density(L, beta):
    """ Get the von neumann density matrix :math:`e^{-bL}` """
    rho = expm(-beta*L)
    return rho / np.trace(rho)


def compute_vonneumann_entropy(**kwargs):
    """ Get the von neumann entropy of the density matrix :math:`Tr[]` """        
    def S(L, beta):
        l = eigvalsh(L)
        #l = l[l > 0] # introduces an error, entropy doesn't tend to 0 in beta=inf
        lrho = np.exp(-beta*l)
        Z = lrho.sum()
        return np.log(Z) + beta * (l*lrho).sum()/Z 
        # alternatively use scipy.entropy
        #return entropy(lrho/lrho.sum())

    if 'density' in kwargs.keys():
        l = eigvalsh(kwargs['density'])
        return entropy(l[l > 0])

    elif 'A' in kwargs.keys() and 'beta' in kwargs.keys():
        A = kwargs['A']
        L = graph_laplacian(A)
        return S(L, kwargs['beta'])

    elif 'L' in kwargs.keys() and 'beta' in kwargs.keys():
        return S(kwargs['L'], kwargs['beta'])

def compute_vonneumann_entropy_beta_deriv(**kwargs):
    """ Get the derivative of entropy with respect to inverse temperature """
    if 'A' in kwargs.keys() and 'beta' in kwargs.keys():
        A = kwargs['A']
        L = graph_laplacian(A)
        rho = compute_vonneuman_density(L, kwargs['beta'])
        logmrho = logm(rho)
        # Tr [ Lρ log ρ ] − Tr [ ρ log ρ ] Tr [ Lρ ]
        return np.trace(L@rho@logmrho) - np.trace(rho@logmrho)*np.trace(L@rho)

    elif 'L' in kwargs.keys() and 'beta' in kwargs.keys():
        rho = compute_vonneuman_density(L, kwargs['beta'])
        return np.trace(L@rho@logm(rho)) - np.trace(rho@logm(rho))*np.trace(L@rho)
    
class VonNeumannDensity(object):
    def __init__(self, A, L, beta, **kwargs):
        self.L = L
        self.A = A
        self.beta = beta
        self.density = compute_vonneuman_density(self.L, beta)


class SpectralDivergence(object):
    def __init__(self, Lobs: np.array, Lmodel: np.array, beta: float, **kwargs):
        self.Lmodel = Lmodel
        self.Lobs = Lobs
        self.fast_mode = kwargs.get('fast_mode', True)
        if 'rho' in kwargs.keys():
            self.rho = kwargs['rho']
        else:
            self.rho = compute_vonneuman_density(Lobs, beta)

        # Average energy of the observation and model
        if self.fast_mode:
            # use the property of hadamard product
            # Trace of matrix product can be simplified by using sum of hadamard product
            # if matrices are symmetric
            # Tr(A B) =  Sum(A_ij B_ij)
            self.Em = (Lmodel*self.rho).sum()
            self.Eo = (Lobs*self.rho).sum()
        else:  # otherwise use correct version, slower for large graphs
            self.Em = np.trace(np.dot(Lmodel, self.rho))
            self.Eo = np.trace(np.dot(Lobs,   self.rho))

        # Computation of partition functions
        if self.fast_mode:  # prefer faster implementation based on eigenvalues
            lm = eigvalsh(Lmodel)
            lo = eigvalsh(Lobs)
            self.Zm = np.exp(-beta*lm).sum()
            self.Zo = np.exp(-beta*lo).sum()
        else:  # otherwise compute with matrix exponentials
            self.Zm = np.trace(expm(-beta*Lmodel))
            self.Zo = np.trace(expm(-beta*Lobs))

        # Computation of free energies from partition functions
        self.Fm = -np.log(self.Zm)/beta
        self.Fo = -np.log(self.Zo)/beta
        
        # Loglikelihood betweeen rho (obs) and sigma (model)
        self.loglike = beta*(-self.Fm + self.Em)
        
        # Entropy of observation (rho)
        self.entropy = beta*(-self.Fo + self.Eo)
        
        # use abs ONLY for numerical issues, as DKL must be positive
        #self.rel_entropy = np.trace(np.dot(compute_vonneuman_density(Lobs,beta),(logm(compute_vonneuman_density(Lobs,beta)) - logm(compute_vonneuman_density(Lmodel,beta)))))
        self.rel_entropy = np.abs(self.loglike - self.entropy)

        if kwargs.get('compute_js', False):
            l = eigvalsh(self.rho + self.sigma)
            self.jensen_shannon = entropy(
                l[l > 0]) - 0.5*(entropy(eigvals(self.rho) + entropy(eigvals(self.sigma))))
        self.options = kwargs
