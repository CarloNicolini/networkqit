import numpy as np

def erdos_renyi(N,p):
    A = np.ones([N,N]) * p
    np.fill_diagonal(A,0)
    return A

def erdos_renyi_gradient(N,p):
    dL = np.eye(N)-1
    np.fill_diagonal(dL,N-1)
    return np.reshape(dL,[N,N,1])


def planted_partition_model(memb,pin,pout):
    return

def undirected_weighted_configuration_model(y):
    A = np.outer(y,y)
    A = A/(1.0 - A)
    return A

def undirected_binary_configuration_model(x):
    A = np.outer(x,x)
    A = A/(1.0 + A)
    return A

def undirected_binary_configuration_model_gradient(x):
    N = len(x)
    g = np.zeros([N,N,N])
    for l in range(0,N):
        degL = 0
        v = set(range(0,N))-set([l])
        for k in v:
            degL += x[k]/((1+x[l]*x[k])**2)
        for i in range(0,N):
            g[i,i,l] = x[i]/((1+x[i]*x[l])**2)
        g[l,l,l] = degL
        for i in v:
            g[l,i,l] = -x[i]/((1+x[i]*x[l])**2)
            g[:,l,l] = g[l,:,l]
    return g
