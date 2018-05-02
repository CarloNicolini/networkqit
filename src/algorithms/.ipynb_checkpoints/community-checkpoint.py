def comm_mat(W, ci):
    u = np.unique(ci)
    C = np.zeros([len(u),len(ci)])
    for i in range(C.shape[0]):
        for j in range(0,len(ci)):
            C[i,j] = ci[j]==u[i]
    B = np.dot(np.dot(C,W),C.T)
    K = B.sum(axis=1)
    np.fill_diagonal(B,B.diagonal()/2)
    n = len(W)
    m = np.triu(W,1).sum()
    p = n * (n - 1) / 2
    commsizes = C.sum(axis=1)
    commpairs = np.dot(np.diag(commsizes),np.diag(commsizes-1)) / 2
    commpairs2 = np.dot(np.dot(np.diag(commsizes) , np.ones([len(commsizes),len(commsizes)])) , np.diag(commsizes))
    blockpairs = np.multiply(commpairs2,(1-np.eye(len(commsizes)))) + commpairs;
    Bnorm = B / blockpairs
    return B, Bnorm

def reindex_membership(membership):
    """
    This function has the membership as input and output the membership
    where the communities number are ordered by the number of nodes in that community
    """
    ds = {}
    for u, v in enumerate(membership):
        if v not in ds.keys():
            ds[v] = []
        ds[v].append(u)

    S = dict(
        zip(range(0, len(ds)), sorted(ds.values(), key=len, reverse=True)))

    M = [-1]*len(membership)
    for u, vl in S.items():
        for v in vl:
            M[v] = u
    return M