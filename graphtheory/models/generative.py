import numpy as np

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

