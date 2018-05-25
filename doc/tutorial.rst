..  -*- coding: utf-8 -*-

Tutorial
========

.. currentmodule:: networkqit

This guide can help you start working with networkqit.
In this first tutorial we show how to compute the spectral entropy of a small graph over a large range of $\beta$ parameter.

Computing the spectral entropy
------------------------------

Let us start by studying the spectral entropy of the Laplacian of the karate club graph. This example shows how to generate the spectral entropy plots shown in our main paper.

.. nbplot::

    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> A = nx.to_numpy_array(G)
    >>> import networkqit as nq
    >>> import numpy as np
    >>> import seaborn as sns
    >>> sns.set()
    >>> beta_range = np.logspace(-3,3,20)
    >>> Sbeta = [nq.compute_vonneumann_entropy(L=nq.graph_laplacian(A),beta=beta) for beta in beta_range]
    >>> import matplotlib.pyplot as plt
    >>> plt.semilogx(1.0/beta_range,Sbeta)
    >>> plt.xlabel('$1/\\beta$')
    >>> plt.ylabel('$S$')
    >>> plt.title('Unnormalized spectral entropy')
    >>> plt.show()

The spectral entropy is always in the range $[0,\log N]$, so if we simply divide by $\log N$ where $N$ is the number of nodes, we renormalize it in the $[0,1]$ range.


Model optimization
------------------

Networkqit can also work with network model fitting. The utilities to define model optimization are defined in the `algorithms` package