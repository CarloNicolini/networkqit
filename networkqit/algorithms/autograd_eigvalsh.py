# Some formulas are from
# "An extended collection of matrix derivative results
#  for forward and reverse mode algorithmic differentiation"
# by Mike Giles
# https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf

from __future__ import absolute_import
from functools import partial
import numpy.linalg as npla
from autograd.numpy.numpy_wrapper import wrap_namespace
import autograd.numpy.numpy_wrapper as anp
from autograd.extend import defvjp

wrap_namespace(npla.__dict__, globals())


def grad_eigvalsh(ans, x, UPLO='L'):
    """Gradient for eigenvalues of a symmetric matrix."""
    N = x.shape[-1]
    w = ans              # w are the eigenvalues
    dot = anp.dot if x.ndim == 2 else partial(anp.einsum, '...ij,...jk->...ik')
    def vjp(g):
        wg = g          # Gradient w.r.t. w eigenvalues
        w_repeated = anp.repeat(w[..., anp.newaxis], N, axis=-1)
        off_diag = 1.0 - anp.eye(N)
        F = off_diag / (T(w_repeated) - w_repeated + anp.eye(N))
        return dot(v * wg[..., anp.newaxis, :] + dot(v, F * dot(T(v), vg)), T(v))
    return vjp

defvjp(anp.eigvalsh, grad_eigvalsh)

if __name__=='__main__':
    import autograd.numpy as np
    import autograd
    A = np.reshape(range(16),[4,4])
    A = A + A.T
    def f(x):
        return autograd.numpy.linalg.eigvalsh(A + x)

    from autograd import value_and_grad
    g = value_and_grad(f)
    g(1)
    from scipy.linalg import eigvalsh
    eigvalsh