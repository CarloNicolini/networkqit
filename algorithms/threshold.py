#!/usr/bin/python

from abc import ABC, abstractmethod
from scipy.optimize import minimize as minimize
from networkqit.infotheory.density import VonNeumannDensity, SpectralDivergence, compute_vonneuman_density
from networkqit.graphtheory import graph_laplacian as graph_laplacian
import numpy as np
import numdifftools as nd

class Threshold(object):
    def __init__(self, A, **kwargs):
        self.A = A
        #self.L = graph_laplacian(A)
        super().__init__()
        
    def __call__(self, threshold):
        self.At = (self.A*(self.A>threshold)).astype(float)
        #self.Lt = graph_laplacian(self.At)
        return self.At