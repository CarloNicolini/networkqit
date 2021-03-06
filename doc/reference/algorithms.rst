.. _algorithms:

**********
Algorithms
**********

This submodule contains definition of various algorithms applied to the spectral entropy framework.
They are mostly based on optimization algorithms for model fitting.

.. automodule:: networkqit.algorithms

Optimization methods
--------------------
.. automodule:: networkqit.algorithms.optimize
.. autosummary::
   :toctree: generated/

   MLEOptimizer
   ContinuousOptimizer
   StochasticOptimizer
   Adam

Basinhopping modified
---------------------
.. automodule:: networkqit.algorithms.basinhoppingmod
.. autosummary::
   :toctree: generated/

   Storage
   BasinHoppingRunner
   AdaptiveStepsize
   RandomDisplacement
   MinimizerWrapper
   Metropolis
   BHBounds
   BHRandStepBounded


Community detection utilities
-----------------------------

.. automodule:: networkqit.algorithms.community
.. autosummary::
   :toctree: generated/

   comm_mat
   comm_assortativity
   reindex_membership
   reassign_singletons