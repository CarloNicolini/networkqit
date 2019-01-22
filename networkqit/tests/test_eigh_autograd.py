import autograd.numpy as np
from autograd import grad,elementwise_grad
from autograd.numpy.linalg import eigh

def f(x):
    A = np.ones([2,2])
    L = np.diag(np.sum(A,axis=0)) - A
    return np.multiply(L, x)

F = lambda x : np.sum(eigh(f(x))[0])
G = grad(lambda x: F(x))
print(F([1.0]))
print(G([1.0]))