import numpy as np
import torch
import math

class normalize():
    def __init__(self,data,method='maxmin',unbiased=True):
        """_summary_

        Args:
            data (np.array): [m,n] Your dataset which needs to be normalized
            method (str, optional): normalization method, Defaults to 'maxmin'.
                'maxmin', 'meandev', sum, l2norm
        """
        if method == 'maxmin':
            self.y = normalize_maxmin(data)
        elif method == 'meandev':
            self.y = normalize_meandev(data,unbiased=unbiased)
        elif method == 'sum':
            self.y = normalize_sum(data)
        elif method == 'l2norm':
            self.y = normalize_l2norm(data)
        else:
            print("MethodError: method not supported!")
            exit(-1)
            
    def solve(self):
        return self.y
    
def normalize_maxmin(data):
    """_summary_

    Args:
        data (np.array): [m,n] Your dataset which needs to be normalized. -> [0,1]
    """
    m,n = data.shape
    data = torch.tensor(data,dtype=float,device='cpu')
    y = (data - data.min(dim=0,keepdim=True)[0].repeat(m,1)) / (data.max(dim=0,keepdim=True)[0].repeat(m,1) - data.min(dim=0,keepdim=True)[0].repeat(m,1))
    return y.numpy()

def normalize_meandev(data,unbiased=True):
    """_summary_

    Args:
        data (np.array): [m,n] Your dataset which needs to be normalized. -> mu=0,stddev=1
    """
    m,n = data.shape
    data = torch.tensor(data,dtype=float,device='cpu')
    y = (data - data.mean(dim=0,keepdim=True)[0].repeat(m,1)) / data.std(dim=0,unbiased=unbiased,keepdim=True)[0].repeat(m,1)
    return y.numpy()

def normalize_sum(data):
    """_summary_

    Args:
        data (np.array): [m,n] Your dataset which needs to be normalized. -> sum=1
    """
    m,n = data.shape
    data = torch.tensor(data,dtype=float,device='cpu')
    y = data / data.sum(dim=0,keepdim=True).repeat(m,1)
    return y.numpy()

def normalize_l2norm(data):
    """_summary_

    Args:
        data (np.array): [m,n] Your dataset which needs to be normalized. -> /l2norm
    """
    m,n = data.shape
    data = torch.tensor(data,dtype=float,device='cpu')
    y = data / torch.sqrt(torch.sum(data ** 2,dim=0,keepdim=True)).repeat(m,1)
    return y.numpy()

def normalize2pos(data,method,options):
    """_summary_

    Args:
        data (np.ndarray): [m,n] Your dataset which needs to be normalized.
        method (str list): 'max', 'min', 'med', 'range'. [n]
        options (list): None or "" for 'max' and 'min', float for 'med', tuple for 'range'. [n]
    """
    m,n = data.shape
    data = torch.tensor(data,dtype=float,device='cpu')
    for j in range(n):
        if method[j] == 'max':
            continue
        elif method[j] == 'min':
            data[:,j] = data[:,j].max().item() - data[:,j]
        elif method[j] == 'med':
            best = torch.tensor(options[j],dtype=float,device='cpu').reshape(1).repeat(m)
            M = torch.abs(data[:,j]-best).max(dim=0,keepdim=True)[0].repeat(m)
            data[:,j] = 1 - torch.abs(data[:,j]-best) / M
        elif method[j] == 'range':
            a,b = options[j]
            M = max((a - data[:,j].min()).item(),(data[:,j].max()-b).item())
            mask1 = torch.tensor([(1-(a-x)/M) if x < a else 0 for x in data[:,j]])
            mask2 = torch.tensor([1 if (a <= x) and (x <= b) else 0 for x in data[:,j]])
            mask3 = torch.tensor([(1-(x-b)/M) if x > b else 0 for x in data[:,j]])
            data[:,j] = mask1+mask2+mask3
        else:
            print("MethodError: Wrong type!")
            exit(-1)
    return data.numpy()