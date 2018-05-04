..  -*- coding: utf-8 -*-

Tutorial
========

.. currentmodule:: networkqit

This guide can help you start working with networkqit.

Creating a graph
----------------

Create an empty graph with no nodes and no edges.

.. nbplot::

    >>> import networkqit as nx
    >>> G = nx.Graph()

By definition, a :class:`Graph` is a collection of nodes (vertices) along with
identified pairs of nodes (called edges, links, etc).  In networkqit, nodes can
be any hashable object e.g., a text string, an image, an XML object, another
Graph, a customized node object, etc.

.. note:: Python's ``None`` object should not be used as a node as it determines
   whether optional function arguments have been assigned in many functions.

