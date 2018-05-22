..  -*- coding: utf-8 -*-

Tutorial
========

.. currentmodule:: networkqit

This guide can help you start working with networkqit.

Computing the spectral entropy
------------------------------

In this guide we try to compute the spectral entropy of a small graph over a large range of $\beta$ parameter.

.. nbplot::

    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> A = nx.to_numpy_array(G)
    >>> import networkqit as nq
    >>> import numpy as np
    >>> 
    >>> beta_range = np.logspace(-3,3,20)
    >>> a=1

