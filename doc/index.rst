..  -*- coding: utf-8 -*-

.. _contents:

Overview
========
.. only:: html

    :Release: |version|
    :Date: |today|

**networkqit** is a Python package for working within the spectral entropy framework of complex networks.
It allows to compute spectral entropies and to fit different models to your observed network easily.
**networkqit** is heavily based on existing packages, such as:

  - networkx
  - numpy
  - scipy
  - pandas

In the current state of development, **networkqit** features two methods of optimization for the relative entropy between observation and model.
The first is based on an approximation that makes the problem tractable analytically.
The second is based on stochastic optimization and allows one to obtain correct estimates of the model parameters, with no approximations, however at the expenses of speed and precision.


Documentation
----------------

.. toctree::
  :maxdepth: 3

  install
  tutorial
  reference/infotheory/
  reference/graphtheory/
  reference/algorithms/
  license
  citing

.. automodule:: networkqit


Free software
-------------

**networkqit** is free software; you can redistribute it and/or modify it under the
terms of the :doc:`3-clause BSD License </license>`.  I welcome contributions. Join me on `Bitbucket <https://bitbucket.org/carlonicolini/networkqit>`_.
