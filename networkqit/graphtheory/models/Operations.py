"""
Some Maximum entropy graph models, inherit from GraphModel.
See Tiziano Squartini thesis for more details
Also reference:
Garlaschelli, D., & Loffredo, M. I. (2008).
Maximum likelihood: Extracting unbiased information from complex networks.
PRE 78(1), 1–4. https://doi.org/10.1103/PhysRevE.78.015101

G here is the graph adjacency matrix, A is the binary adjacency, W is the weighted
"""

from .GraphModel import GraphModel


class Operator(GraphModel):
    def __init__(self, left, right):
        self.left = left
        self.right = right
        # unique elements, order preserving

        def unique_ordered(seq):
            seen = set()
            seen_add = seen.add
            return [x for x in seq if not (x in seen or seen_add(x))]

        self.args_mapping = unique_ordered(left.args_mapping + right.args_mapping)
        self.n = self.args_mapping
        self.idx_args_left = [
            i for i, e in enumerate(self.args_mapping) if e in set(left.args_mapping)
        ]
        self.idx_args_right = [
            i for i, e in enumerate(self.args_mapping) if e in set(right.args_mapping)
        ]
        self.bounds = left.bounds + right.bounds
        self.formula = left.formula[0:-1] + str(self) + right.formula[1:]
        # print(self.args_mapping,self.idx_args_left,self.idx_args_right,self.right.args_mapping)


class Mul(Operator):
    def __call__(self, x):
        return self.left(x[self.idx_args_left]) * self.right(x[self.idx_args_right])

    def __str__(self):
        return "*"


class Add(Operator):
    def __call__(self, x):
        return self.left(x[self.idx_args_left]) + self.right(x[self.idx_args_right])

    def __str__(self):
        return "+"


class Div(Operator):
    def __call__(self, x):
        return self.left(x[self.idx_args_left]) / self.right(x[self.idx_args_right])

    def __str__(self):
        return "/"
