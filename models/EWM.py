import numpy as np
import torch
import math
from models.normalization import normalize

def EWM(data,require_normalize=True,epsilon=1e-7):
    """
    We suppose your raw data has already been normalized.
    epsilon is added to prevent nan error.
    input:
        data: [m,n] m objects, n criteria. (np.array)
    output:
        weight: [n] weights for each criteria. (np.array)
    """
    if require_normalize:
        data =  normalize(data=data,method="maxmin").solve()
    m = data.shape[0]
    n = data.shape[1]
    data = torch.tensor(data,dtype=float,device='cpu')
    p = data / data.sum(dim=0,keepdim=True).repeat(m,1) + epsilon
    e = - (p * torch.log(p) / math.log(m)).sum(dim=0,keepdim=False)
    weight = (1 - e) / (n - e.sum(dim=0,keepdim=False).item())
    return weight.numpy()