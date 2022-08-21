import torch
import numpy as np
from math import sqrt

class ClusterAlg():
    def __init__(self,m,n,k,method='K-means',require_report=False,tol=1e-6) -> None:
        """
        m: dataset size
        n: dimension of data
        k: cluster number
        method: 
            'K-means'
        require_report: (Not supported yet)
            the report of clustering result
            metrics
        """
        self.m = m
        self.n = n
        self.k = k
        self.method = method
        self.require_report = require_report
        self.tol = tol
        self.report = {}

    def __solveK_means(self,x,max_iter=500):
        """
        x: [m,n], np.array
        max_iter: int
        Return:
            centroid [k,n], indices [m] np.array
        """
        x = torch.tensor(x,dtype=float,device='cuda')
        centroid = x.gather(dim=0,index=torch.randperm(self.m,device='cuda').reshape(-1,1).repeat(1,self.n)[:self.k,:])
        pre_cen = centroid
        for j in range(max_iter):
            dist = (x.reshape(self.m,1,self.n).repeat(1,self.k,1) - centroid.reshape(1,self.k,self.n).repeat(self.m,1,1)) ** 2 # [m,k,n]
            dist = dist.sum(dim=2,keepdim=False) # [m,k]
            _,indices = dist.min(dim=1,keepdim=False) # [m]
            num_points = torch.tensor(np.linspace(0,self.k,self.k,endpoint=False),dtype=int,device='cuda').reshape(1,-1).repeat(self.m,1) # [m,k]
            num_points = (num_points == indices.reshape(-1,1).repeat(1,self.k)).float()
            pre_cen = centroid
            centroid = (num_points.reshape(self.m,self.k,1).repeat(1,1,self.n) * x.reshape(self.m,1,self.n).repeat(1,self.k,1)).sum(dim=0,keepdim=False) / num_points.sum(dim=0,keepdim=False).view(self.k,1).repeat(1,self.n) # [k,n]
            if sqrt(((pre_cen - centroid) ** 2).max().item()) <= self.tol:
                self.report['Convergence'] = True
                self.report['iter'] = j
                break
        return centroid.cpu().numpy(), indices.cpu().numpy()

    def solve(self,x,max_iter=500):
        """
        x: [m,n], np.array
        max_iter: int
        Return:
            centroid [k,n], indices [m] np.array
        """
        if self.method == 'K-means':
            cenroid, indices = self.__solveK_means(x,max_iter)
        else:
            print("Not supported method!")
            exit(-1)
        if self.require_report:
            print(self.report)
        return cenroid, indices

    def get_report(self):
        if self.require_report:
            return self.report
        else:
            return None