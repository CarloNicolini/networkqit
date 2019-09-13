import autograd.numpy as np
from autograd import grad,elementwise_grad
from autograd.numpy.linalg import eigh

def f(x):
    A = np.ones([2,2])
    L = np.diag(np.sum(A,axis=0)) - A
    I = np.eye(2)
    return np.sum(np.log(eigh(x*I + L)[0])) - np.log(x)

g = grad(lambda x: f(x))
print(g(1.0))