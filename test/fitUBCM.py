import sys
sys.path.append('/home/carlo2/workspace/networkqit')
from networkqit.graphtheory import UBCM, UWCM
from networkqit.algorithms import MLEOptimizer
import numpy as np

def binarize(W):
    return (W>0).astype(int)

if __name__=='__main__':
    W = np.loadtxt('/home/carlo2/workspace/communityalg/data/Coactivation_matrix_weighted.adj')[0:20,0:20]
    A = binarize(W)
    x0 = np.random.random([len(A),1])
    opt = MLEOptimizer(G=A,x0=x0)
    sol = opt.runfsolve(model=UWCM(N=len(A)))
    
    print()