import torch

class LinearRegression():
    """
    The class that performs Linear Regression, 
        using GPU for gd (deprecated).
    pred_y = Wx + b, x[n], y[m]
    """
    def __init__(self,n,m,gpu_enable = True) -> None:
        """
        pred_y = Wx + b, x[n], y[m]
        n: dimension for x
        m: dimenson for y
        gpu_enable: use gpu or not, when calling solve().
            solve_gd automatically set to use gpu
            solve_1d automatically set to use cpu
            both deprecated!
        """
        self.n = n
        self.m = m
        self.gpu_enable = gpu_enable
        return

    def solve_gd(self,x,label_y,alpha=1e-6,epoch=500,W=None,b=None):
        """
        x: [s,n], s for sample size, iterable
        y: [s,m], s for sample size, iterable
        (pred_y = xWT + bT.repeat)
        (Gradient descent):
            epoch: epoch
            alpha: learning rate
        (initial value):
            W: [m,n], iterable
            b: [m,1], iterable
        Return:
            pred_y [s,m], pred_W [m,n], pred_b [m,1] (np.array)
        """
        x = torch.tensor(x,dtype=float,device='cuda')
        x = x.reshape(x.shape[0],-1) # walk around when n = 1
        y = torch.tensor(label_y,dtype=float,device='cuda')
        y = y.reshape(y.shape[0],-1) # walk around when m = 1

        if W == None:
            pred_W = torch.randn(self.m,self.n,device='cuda',dtype=float)
        else:
            pred_W = torch.tensor(W,dtype=float,device='cuda')
            pred_W = pred_W.reshape(pred_W.shape[0],-1) # walk around when m or n = 1
        if b == None:
            pred_b = torch.randn(self.m,1,device='cuda',dtype=float)
        else:
            pred_b = torch.tensor(b,dtype=float,device='cuda')
            pred_b = pred_b.reshape(pred_b.shape[0],-1) # walk around when m or n = 1

        for _ in range(epoch):
            pred_y = torch.matmul(x,torch.transpose(pred_W,1,0)) + torch.transpose(pred_b,1,0).repeat(x.shape[0],1)
            pred_W = pred_W - alpha * torch.matmul(torch.transpose((pred_y - y),1,0),x)
            pred_b = pred_b - alpha * (pred_y - y).sum(dim=0,keepdim=True).transpose(1,0)

        return pred_y.cpu().numpy(), pred_W.cpu().numpy(), pred_b.cpu().numpy()

    def solve_1d(self,x,label_y):
        """
        optimal solution for 1 dim.
        x: [s]
        y: [s]
        Return:
            pred_y (np.array), pred_a, pred_b  (float)
        """
        if self.n != 1 or self.m != 1:
            print("Error: solve_1d can only solve x,y both 1d.")
            exit(-1)
        x = torch.tensor(x,dtype=float,device='cuda').view(-1)
        y = torch.tensor(label_y,dtype=float,device='cuda').view(-1)
        n = x.shape[0]

        pred_a = (n * torch.matmul(x,y) - x.sum() * y.sum()) / ((n * torch.matmul(x,x)) - (x.sum() ** 2))
        pred_b = (y.sum() - pred_a * x.sum()) / n
        pred_y = pred_a.item() * x + pred_b.item()

        return pred_y.cpu().numpy(), pred_a.item(), pred_b.item()

    def solve(self,x,label_y):
        """
        x: [s,n], s for sample size, iterable
        y: [s,m], s for sample size, iterable
        (pred_y = xWT + bT.repeat)
        Return:
            pred_y [s,m], pred_W [m,n], pred_b [m,1] (np.array)
        """
        x = torch.tensor(x,dtype=float,device='cpu')
        x = x.reshape(x.shape[0],-1) # walk around when n = 1
        y = torch.tensor(label_y,dtype=float,device='cpu')
        y = y.reshape(y.shape[0],-1) # walk around when m = 1

        if self.gpu_enable:
            x.cuda()
            y.cuda()
        
        pred_W = torch.matmul(x.shape[0] * torch.matmul(y.transpose(1,0),x) \
            -  torch.matmul(y.transpose(1,0).sum(dim=1,keepdim=True),x.sum(dim=0,keepdim=True)),
            (x.shape[0] * torch.matmul(x.transpose(1,0),x)  
                - torch.matmul(x.transpose(1,0).sum(dim=1,keepdim=True),x.sum(dim=0,keepdim=True))).inverse()
        )
        pred_b = (1 / x.shape[0]) * (y.transpose(1,0).sum(dim=1,keepdim=True) 
             - torch.matmul(pred_W,x.transpose(1,0).sum(dim=1,keepdim=True))
        )

        pred_y = torch.matmul(x,pred_W.transpose(1,0)) + pred_b.transpose(1,0).repeat(x.shape[0],1)
        
        if self.gpu_enable:
            return pred_y.cpu().numpy(), pred_W.cpu().numpy(), pred_b.cpu().numpy()
        else:
            return pred_y.numpy(), pred_W.numpy(), pred_b.numpy()