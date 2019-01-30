..  -*- coding: utf-8 -*-

Tutorial
========

.. currentmodule:: networkqit

This guide can help you start working with networkqit.
In this first tutorial we show how to compute the spectral entropy of a small graph over a large range of $\beta$
parameter.

Computing the spectral entropy of a network
-------------------------------------------

Let us start by studying the spectral entropy of the density matrix of the famous karate club graph.
This example shows how to generate the spectral entropy plots shown in our main paper.

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

Generative network models
-------------------------

**networkqit** features a large number of network models, mainly those obtained from within the Maximum Entropy
framework. You can create a number of models and call methods on them by a consistent nomenclature:
Here we create an instance of the Undirected Binary Configuration model (UBCM), and sample 10 random networks from this
ensemble, based on the fitness parameters .. :math: x_i created at  random in the [0,1] domain.

.. nbplot::

   >>> import networkqit as nq
   >>> N=10
   >>> model = nq.UBCM(N=10)
   >>> xi = np.random.random([N,])
   >>> adj_graphs = model.sample_adjacency(xi, batch_size=5)
   >>> print(adj_graphs.shape)

The call returns a [5,N,N] numpy array, where the last two dimensions embed with the adjacency matrix of the 5 random graphs.
The method `sample_adjacency` is available for every generative model implemented in **networkqit** and is at the base  of the *maximize and sample* approach used for the optimization of the spectral entropy. 

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

Stochastic optimization of spectral entropy
-------------------------------------------

Most of the models do not allow an analytical treatment of their spectral likelihood. This involves knowing exact analytical formulations of the laplacian eigenvalues spectral density and this is still an unsolved problem for most of the useful network models.
To perform spectral likelihood optimization we can do stochastic optimization and find the parameters of the empirical graph such that the expected relative entropy averaged over the model ensemble 
is minimized. 
Doing this in **networkqit** is simple, as the library relies on the `autograd` package for the automatic calculation of complicated gradients of stochastic functions.

With the optimization method `Adam` which is borrowed from machine learning, we can optimize the expected relative entropy of the Erdos-Renyi model with respect to our model karate-club network,represented by **A** and see the effects of changing the inverse temperature parameter `beta` on the optimization process.

At every iteration, a number of random networks (batch_size) are sampled from the model, and their average spectral properties are used to compute an approximation to the expected relative entropy.


.. nbplot::

   >>> solver = opt = Adam(A=A, L=L, x0=x0, beta_range=beta_range, model=ermodel)
   >>> sol = opt.run(refresh_frames=100, eta=0.001, max_iters=5000, gtol=1E-5, batch_size=128)

