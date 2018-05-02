#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 14:48:46 2018

@author: carlo2
"""

from networkqit.graphtheory.models import *
from networkqit.graphtheory import graph_laplacian as graph_laplacian
from abc import abstractmethod
import numpy as np
import math

def StochasticModel(Model):
    def __init__(self, **kwargs):
        self.dij = kwargs['dij']
        
    def random_instance(self):
        