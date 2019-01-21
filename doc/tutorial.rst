..  -*- coding: utf-8 -*-

Tutorial
========

.. currentmodule:: networkqit

This guide can help you start working with networkqit.
In this first tutorial we show how to compute the spectral entropy of a small graph over a large range of $\beta$
parameter.

Computing the spectral entropy
------------------------------

Let us start by studying the spectral entropy of the Laplacian of the karate club graph. This example shows how to
 generate the spectral entropy plots shown in our main paper.

.. nbplot::

   >>> import networkx as nx
   >>> G = nx.karate_club_graph()
   >>> A = nx.to_numpy_array(G)
   >>> import networkqit as nq
   >>> import numpy as np
   >>> import matplotlib.pyplot as plt
   >>> plt.style.use('ggplot')
   >>> beta_range = np.logspace(-3,3,200)
   >>> Sbeta = nq.batch_compute_vonneumann_entropy(L=nq.graph_laplacian(A),beta_range=beta_range)
   >>> plt.semilogx(beta_range, Sbeta)
   >>> plt.xlabel('$\\beta$')
   >>> plt.ylabel('$S$')
   >>> plt.title('Unnormalized spectral entropy')
   >>> plt.show()

The spectral entropy is always in the range $[0,\log N]$, so if we simply divide by $\log N$ where $N$ is the number
of nodes, we renormalize it in the $[0,1]$ range.


Model optimization
------------------

Networkqit can also work with network model fitting. The utilities to define model optimization are defined in the
`algorithms` package.

The following lines describe an example describing optimization of the expected laplacian model.
We first create the object ErdosRenyi that describes the expected laplacian, adjacency and laplacian gradient with
respect to the only continuous parameters $p$.

.. nbplot::

   >>> import networkx as nx
   >>> import networkqit as nq
   >>> import numpy as np
   >>> G = nx.karate_club_graph()
   >>> A = nx.to_numpy_array(G)
   >>> L = nq.graph_laplacian(A)
   >>> beta_range = np.logspace(-3,3,20)
   >>> ermodel = nq.ErdosRenyi(N=len(A))


We can call the `ermodel` as if it is a simple function:

.. nbplot::

   >>> print(ermodel([0.5]))

This returns the expected adjacency matrix, a 34x34 matrix with 0.5 off diagonal and 0 on diagonal.
The `ermodel` also allows to call methods like the expected laplacian or the expected laplacian gradient.

.. nbplot::

   >>> print(ermodel.expected_laplacian_grad([0.5]))

Otherwise we can sample 10 independent networks from the Erdos-Renyi model, with the method `sample_adjacency`:

.. nbplot::

   >>> print(ermodel.sample_adjacency([0.5]), batch_size=10)

Once the `ermodel` object is defined, we can use it within the `ContinuousOptimizer`.
We initialize the observed adjacency matrix as `A` and the initial solution `x0=np.array([0.2])`. It is important to define `x0` as a numpy array.
Finally the `beta_range` is a numpy array with the range of `beta` over which to optimize the model.

.. nbplot::

   >>> solver = nq.ContinuousOptimizer(A=A, x0=np.array([0.2]), beta_range=beta_range)
   >>> solver.setup(ermodel, ermodel.expected_laplacian_grad, step_callback=None)
   >>> sol = solver.run()
   >>> print(sol)


Alternatively we can do stochastic optimization and find the parameters of the empirical graph:

.. nbplot::

   >>> solver = opt = Adam(A=A, L=L, x0=x0, beta_range=beta_range, model=ermodel)
   >>> sol = opt.run(refresh_frames=100, eta=0.001, max_iters=5000, gtol=1E-5, batch_size=128)

