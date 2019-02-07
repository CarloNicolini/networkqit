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
Some Maximum entropy graph models, inherit from GraphModel.
See Tiziano Squartini thesis for more details
Also reference:
Garlaschelli, D., & Loffredo, M. I. (2008).
Maximum likelihood: Extracting unbiased information from complex networks.
PRE 78(1), 1–4. https://doi.org/10.1103/PhysRevE.78.015101

G here is the graph adjacency matrix, A is the binary adjacency, W is the weighted
"""

import autograd.numpy as np
from .GraphModel import GraphModel
from .GraphModel import expit, batched_symmetric_random, multiexpit

EPS = np.finfo(float).eps

class Weibull(GraphModel):
    """
    Weibull distribution
    Eq. 4.33 Barabasi - Network Science (advanced topis, power laws)
    """
    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = ['c_wei', 'mu_wei', 'gamma_wei']
        self.model_type = 'spatial'        
        self.formula = r'$c_{wei} d_{ij}^{\gamma_{wei}-1} e^{-(\mu_{wei} d_{ij})^\gamma_{wei}}$'
        self.bounds = [(0, None), (0, None), (0, None)]

    def expected_adjacency(self, *args):
        P = args[0]*np.exp(-(args[1]*self.parameters['dij'])**args[2])
        np.fill_diagonal(P, 0)
        return P


class LevyFligth(GraphModel):
    """
    Levy law for mobile phone users (Gonzalez et al, 2009)
    Understanding individual human mobility patterns, Nature 453 (2009) 779–782
    # https://ac.els-cdn.com/S037015731000308X/1-s2.0-S037015731000308X-main.pdf
    """

    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = ['c_levy', 'delta_r_0_levy',
                             'gamma_levy', 'cutoff_levy']
        self.model_type = 'spatial'
        self.formula = r'$c_{levy} $'
        self.bounds = [(0, None), (0, None), (0, None), (0,None)]

    def expected_adjacency(self, *args):
        P = args[0]/((args[1]+self.parameters['dij'])**args[2]) * np.exp(-self.parameters['dij']/args[3])
        np.fill_diagonal(P, 0)
        return P


class S1(GraphModel):
   def __init__(self, **kwargs):
       if kwargs is not None:
           super().__init__(**kwargs)
       self.args_mapping = [
           'x_' + str(i) for i in range(0, kwargs['N'])] + ['beta', 'mu']
       self.model_type = 'topological'
       self.formula = '$\frac{1}{1+\left(\frac{d_{ij}}{\mu k_i k_j}\right)^\beta}$'
       self.bounds = [(0, None) for i in range(0, kwargs['N'])]

   def expected_adjacency(self, *args):
       beta = args[-2]
       mu = args[-1]
       O = args[-1]*np.outer(args[:-2])
       return 1.0/(1.0 + O**args[-2])