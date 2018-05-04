"""
networkqit
========

    NetworkQIT (NX) is a Python package for the fitting of complex generative models to networks using quantum information theory.

    https://networkqit.github.io

Using
-----

    Just write in Python

    >>> import networkqit as nq
    >>> import networkx as nx
    >>> G=nx.karate_club_graph()
    >>> (A,beta)=(nx.to_numpy_matrix(),0.5)
    >>> spdens = nq.SpectralDensity(A,beta)
    >>> print(spdens.rho)
"""
#    Copyright (C) 2018 by
#    Carlo Nicolini <carlo.nicolini@iit.it>
#    All rights reserved.
#    BSD license.
#
# Add platform dependent shared library path to sys.path
#

from __future__ import absolute_import

import sys
if sys.version_info[:2] < (2, 7):
    m = "Python 2.7 or later is required for networkqit (%d.%d detected)."
    raise ImportError(m % sys.version_info[:2])
del sys

# Release data
from networkqit import release

__author__ = '%s <%s>\n%s <%s>\n' % \
    (release.authors['Nicolini'] + release.authors['Vlasov'])
__license__ = release.license

__date__ = release.date
__version__ = release.version

__bibtex__ = """xxx"""

import numpy as np
import networkx as nx

# These are import orderwise
import networkqit.graphtheory
from networkqit.graphtheory import *

import networkqit.infotheory
from networkqit.infotheory import *

import networkqit.algorithms
from networkqit.algorithms import *

import networkqit.utils
from networkqit.utils import *