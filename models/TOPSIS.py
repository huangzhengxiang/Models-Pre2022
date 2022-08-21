from models.normalization import normalize,normalize2pos
import numpy as np
import torch
import math
from models.EWM import EWM

def TOPSIS(data,weight_method=None,norm_cfg=None):
    """
    Input:
        data [m,n] (numpy.ndarray)
        weight_method (str | None)
            None for no weighting
            'EWM' for entropy weight method
            other methods aren't supported.
        norm_cfg (dict)
            at least consists of "normalization" and "options"
    Output:
        score [m] (numpy.ndarray) score for each object
    """
    m,n = data.shape
    data = normalize2pos(data=data,method=norm_cfg['normalization'],options=norm_cfg['options'])
    data = normalize(data=data,method='l2norm').solve()
    weight = np.ones(n,float)
    if weight_method == None:
        pass
    elif weight_method == "EWM":
        weight = EWM(data=data,require_normalize=False)
    else:
        print("MethodError: method not supported!")
        exit(-1)
    weight = torch.tensor(data=weight,dtype=float,device='cpu').reshape(1,-1).repeat(m,1)
    data = torch.tensor(data=data,dtype=float,device='cpu')
    Z_pos = torch.max(data,dim=0,keepdim=True)[0].repeat(m,1)
    Z_neg = torch.min(data,dim=0,keepdim=True)[0].repeat(m,1)
    D_pos = torch.sqrt((weight * ((Z_pos - data)**2)).sum(dim=1))
    D_neg = torch.sqrt(((Z_neg - data)**2).sum(dim=1))
    S = D_neg / (D_pos + D_neg)
    S = normalize(S.reshape(-1,1).numpy(),'sum').solve()
    return S.reshape(-1)