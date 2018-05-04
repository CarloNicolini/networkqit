#!/usr/bin/env python
"""
==========
Properties
==========

Compute some network properties for the lollipop graph.
"""
#    Copyright (C) 2004-2018 by
#    Aric Hagberg <hagberg@lanl.gov>
#    Dan Schult <dschult@colgate.edu>
#    Pieter Swart <swart@lanl.gov>
#    All rights reserved.
#    BSD license.

import matplotlib.pyplot as plt
from networkx import nx

G = nx.lollipop_graph(4, 6)
