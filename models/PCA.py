import numpy as np
from numpy import dot
from models.normalization import normalize_meandev

def PCA_solve(X,require_normalize=False,unbiased=True):
    """
    Input:
        X (ndarray): [m,n] [samples,dims], where E(X)=0, D(X)=1
    Return:
        eig_val (ndarray): [n] eigen values, sorted monotonically non-descending
        D (ndarray): [n,n] eigen vectors, each column corresponding to eig_val.
        info_con (ndarray): [n] information contribution
        info_con_cum (ndarray): [n] information contribution (cumulative)
        R (ndarray): [n,n] correlation matrix
    """
    if require_normalize:
        X = normalize_meandev(X,unbiased=unbiased)
    m = len(X[:,0])
    if unbiased:
        R = dot(np.transpose(X),X) / (m-1)
    else:
        R = dot(np.transpose(X),X) / m
    eigen_val, eigen_vector = np.linalg.eigh(R)
    idx = np.argsort(eigen_val)
    eig_val = np.zeros_like(eigen_val)
    D = np.zeros_like(eigen_vector)
    info_con = np.zeros_like(eigen_val)
    info_con_cum = np.zeros_like(eigen_val)
    n = len(idx)
    for j in range(n):
        eig_val[n-1-j] = eigen_val[idx[j]]
        info_con[n-1-j] = eigen_val[idx[j]]
        if eigen_vector[:,idx[j]].sum() < 0:
            D[:,n-1-j] = - eigen_vector[:,idx[j]]
        else:
            D[:,n-1-j] = eigen_vector[:,idx[j]]
    den = eig_val.sum()
    for j in range(n):
        info_con[j] = info_con[j] / den
        info_con_cum[j] = info_con_cum[j-1] + info_con[j]
    return eig_val, D, info_con, info_con_cum, R

def PCA_eval(X,l,utils=None):
    """
    evaluate objects based on selected principle conponents.
    Input: 
        X (ndarray): [m,n] [samples,dims]
        l (int): number of principle conponent you want to preserve. (l < n)
        utils (tuple(eig_val, D, info_con)) | None: if None, we will recalculate.
    Return:
        C (ndarray): [m,l] compacted metric matrix
        score (ndarray): [m] score computed with info_con
    """
    if utils == None:
        eig_val, D, info_con, _ = PCA_solve(X)
    else:
        eig_val, D, info_con = utils
    assert l < len(eig_val)
    C = dot(X,D[:,:l])
    weight = info_con[:l].reshape(-1,1)
    score = dot(C,weight).reshape(-1)
    return C, score