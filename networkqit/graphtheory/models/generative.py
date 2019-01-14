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

import autograd.numpy as np

class StochasticModel(object):
    def __init__(self, **kwargs):
        super().__init__()
        
    def generate_adjacency():
        pass

        
class StochasticEdr(StochasticModel):
    def __init__(self, **kwargs):
        self.dij = kwargs['dij']
        self.decay = kwargs['decay']
    
    def generate_adjacency():
        N = len(self.dij)
        A = np.zeros([N,N])
        while A.sum() < L:
            x = np.random.exponential(1/self.decay,(N,N))
            A += x * (( self.dij<(5/4*x) ) & ( self.dij>(1/4*x))).astype(float)
            np.fill_diagonal(A, 0)
        return A#*(mu**2)

