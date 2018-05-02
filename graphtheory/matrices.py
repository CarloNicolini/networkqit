import numpy as np

def graph_laplacian(A):
    D = np.zeros(A.shape)
    np.fill_diagonal(D,A.sum(axis=0))
    return D - A

def normalized_graph_laplacian(A):
    D = np.zeros(A.shape)
    np.fill_diagonal(D,np.sum(A,1))
    D = np.diag((1.0/np.sqrt(np.diag(D))))
    return np.eye(A.shape[0]) - D*A*D