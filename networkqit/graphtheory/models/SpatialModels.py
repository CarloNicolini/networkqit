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


class SpatialCM(GraphModel):
    """
    Implements the random graph model with spatial constraints from:
    Ruzzenenti, F., Picciolo, F., Basosi, R., & Garlaschelli, D. (2012).
    Spatial effects in real networks: Measures, null models, and applications.
    Physical Review E - Statistical, Nonlinear, and Soft Matter Physics, 86(6),
    1–13. https://doi.org/10.1103/PhysRevE.86.066110
    """

    def __init__(self, **kwargs):
        if kwargs is not None:
            super().__init__(**kwargs)
        self.args_mapping = ["x_" + str(i) for i in range(0, kwargs["N"])] + [
            "gamma",
            "z",
        ]
        # self.formula = '$\frac{z x_i x_j e^{-\gamma d_{ij}}}{ 1 + z x_i x_j e^{-\gamma d_{ij}}  }$'
        self.bounds = [(EPS, None) for i in range(0, kwargs["N"])] + [
            (EPS, None),
            (EPS, None),
        ]
        self.dij = kwargs["dij"]
        self.expdij = np.exp(-kwargs["dij"])
        self.is_weighted = kwargs.get("is_weighted", False)

    def expected_adjacency(self, *args):
        O = args[-1] * np.outer(args[0:-2], args[0:-2]) * self.expdij ** (args[-2])
        if self.is_weighted:
            return O / (1 + O)
        return O / (1 - O)
