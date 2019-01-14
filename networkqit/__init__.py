#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# networkqit -- a python module for manipulations of spectral entropies framework
#
# Copyright (C) 2017-2018 Carlo Nicolini <carlo.nicolini@iit.it>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
networkqit
==========

    networkqit (nq) is a Python package for the fitting of complex generative models to networks using quantum information theory.

    https://carlonicolini.github.io/networkqit

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

from __future__ import absolute_import

import sys
if sys.version_info[:2] < (2, 7):
    m = "Python 2.7 or later is required for networkqit (%d.%d detected)."
    raise ImportError(m % sys.version_info[:2])
del sys

# Release data
from networkqit import release

__author__ = '%s <%s>\n%s <%s>\n' % (release.authors['Nicolini'] + release.authors['Vlasov'])
__license__ = release.license

__date__ = release.date
__version__ = release.version

__bibtex__ = """@article{nicolini2018thermodynamics,
title={Thermodynamics of network model fitting with spectral entropies},
author={Nicolini, Carlo and Vlasov, Vladimir and Bifone, Angelo},
journal={arXiv preprint arXiv:1801.06009},
year={2018}}"""

name = "networkqit"

import autograd.numpy as np
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
