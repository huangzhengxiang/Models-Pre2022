import numpy as np
from models.LR import LinearRegression

class ExpSmoothing():
    """
    class for Exponential Smoothing.
    support 4 variants.
        'simple': p[t+1] = alpha * z[t] + p[t]
        'hzx':  p[t+1] = alpha * z[t] + p[t] + a
        'trend': p[t+1] = e[t] + b[t]
            e[t] = alpha * z[t] + (1 - alpha) * p[t], (no trend)
            b[t] = beta * (e[t] - e[t-1]) + (1 - beta) * b[t-1], (trend only)
        'season': sp[t+1] = e[t] + b[t] + s[t+1-m]
            p[t+1] = e[t] + b[t] (trend)
            e[t] = alpha * (z[t] - s[t-m]) + (1 - alpha) * p[t],  (no trend and season)
            b[t] = beta * (e[t] - e[t-1]) + (1 - beta) * b[t-1], (trend only)
            s[t] = gamma * (z[t] - p[t]) + (1 - gamma) * s[t-m], (season only)
    """
    def __init__(self,method="simple",alpha=0.6,beta=0.6,gamma=0.6,season_len=None) -> None:
        """
        T: the number that takes into account.
        the bigger the parameter is, the more important the nearby property is.
        alpha: all required
        beta: not required by 'simple', 'hzx'
        gamma: not required by 'simple', 'hzx', and 'trend'
        season_len: used only for season.
        method: 'simple', 'trend'
        """
        self.alpha = alpha
        self.beta = beta
        self.method = method
        self.gamma = gamma
        self.season_len = season_len

    def pred_simple(self,x,y,l):
        """
        x: [s], s for sample size, np.array
        y: [s], s for sample size, np.array
        Return:
            pred_y: [s+l], np.array
        """
        s = x.shape[0]
        pred_y = np.ones(s+l,dtype=float)
        pred_y[0] = y[0]
        for j in range(1,s):
            pred_y[j] = self.alpha * y[j-1] + (1 - self.alpha) * pred_y[j-1]
        for j in range(s,s+l):
            pred_y[j] = pred_y[j-1]
        return pred_y

    def pred_hzx(self,x,y,l):
        """
        x: [s], s for sample size, np.array
        y: [s], s for sample size, np.array
        Return:
            pred_y: [s+l], np.array
        """
        _,a,_ = LinearRegression(1,1).solve_1d(x,y)
        s = x.shape[0]
        pred_y = np.ones(s+l,dtype=float)
        pred_y[0] = y[0]
        for j in range(1,s):
            pred_y[j] = self.alpha * y[j-1] + (1 - self.alpha) * pred_y[j-1] + a
        for j in range(s,s+l):
            pred_y[j] = a + pred_y[j-1]
        return pred_y


    def pred_trend(self,x,y,l):
        """
        x: [s], s for sample size, np.array
        y: [s], s for sample size, np.array
        Return:
            pred_y: [s+l], np.array
        """
        s = x.shape[0]
        e = np.ones(s+l,dtype=float)
        b = np.ones(s+l,dtype=float)
        pred_y = np.ones(s+l,dtype=float)
        pred_y[0] = pred_y[1] = y[0]
        e[0] = e[1] = y[0]
        b[0] = b[1] = 0
        for j in range(2,s):
            e[j-1] = self.alpha * y[j-1] + (1 - self.alpha) * pred_y[j-1]
            b[j-1] = self.beta * (e[j-1] - e[j-2]) + (1 - self.beta) * b[j-2]
            pred_y[j] = e[j-1] + b[j-1]
        for j in range(s,s+l):
            e[j-1] = pred_y[j-1]
            b[j-1] = self.beta * (e[j-1] - e[j-2]) + (1 - self.beta) * b[j-2]
            pred_y[j] = e[j-1] + b[j-1]
        return pred_y

    def pred_season(self,x,y,l):
        """
        x: [s], s for sample size, np.array
        y: [s], s for sample size, np.array
        Return:
            pred_y: [s+l], np.array
        """
        s = x.shape[0]
        m = self.season_len
        if m == None:
            print("ValueError: season_len is not provided!")
        if m >= s+1:
            print("Error: season_len is too short for your input dataset!")
            exit(-1)
        e = np.ones(s+l,dtype=float)
        b = np.ones(s+l,dtype=float)
        S = np.ones(s+l,dtype=float)
        p = np.ones(s+l,dtype=float)
        pred_y = np.ones(s+l,dtype=float)
        p[0] = p[1] = pred_y[0] = pred_y[1] = y[0]
        e[0] = e[1] = y[0]
        S[0] = b[0] = b[1] = 0
        for j in range(2,m+1):
            e[j-1] = self.alpha * (y[j-1] - S[j-1-m]) + (1 - self.alpha) * p[j-1] # (no trend and season)
            b[j-1] = self.beta * (e[j-1] - e[j-2]) + (1 - self.beta) * b[j-2] # (trend only)
            S[j-1] = self.gamma * (y[j-1] - p[j-1]) # (season only)
            p[j] = pred_y[j] = e[j-1] + b[j-1] # (trend)
        for j in range(m+1,s):
            e[j-1] = self.alpha * (y[j-1] - S[j-1-m]) + (1 - self.alpha) * p[j-1] # (no trend and season)
            b[j-1] = self.beta * (e[j-1] - e[j-2]) + (1 - self.beta) * b[j-2] # (trend only)
            S[j-1] = self.gamma * (y[j-1] - p[j-1]) + (1 - self.gamma) * S[j-1-m] # (season only)
            p[j] = e[j-1] + b[j-1] # (trend)
            pred_y[j] = p[j] + S[j-m]
        for j in range(s,s+l):
            e[j-1] = p[j-1] # (no trend and season)
            b[j-1] = self.beta * (e[j-1] - e[j-2]) + (1 - self.beta) * b[j-2] # (trend only)
            S[j-1] = S[j-1-m] # (season only)
            p[j] = e[j-1] + b[j-1] # (trend)
            pred_y[j] = p[j] + S[j-m]
        return pred_y

    def solve(self, x, label_y, pred_len):
        """
        x: [s], s for sample size, iterable
        y: [s], s for sample size, iterable
        pred_len: pediction length
        Return:
            pred_y: [s+l], (numpy.array)
        """
        x = np.array(x,dtype=float).reshape(-1)
        y = np.array(label_y,dtype=float).reshape(-1)
        pred_y = np.array(0,dtype=float).reshape(-1)
        if self.method == "simple":
            pred_y = self.pred_simple(x,y,pred_len)
        elif self.method == "trend":
            pred_y = self.pred_trend(x,y,pred_len)
        elif self.method == 'hzx':
            pred_y = self.pred_hzx(x,y,pred_len)
        elif self.method == 'season':
            pred_y = self.pred_season(x,y,pred_len)
        else:
            print("Method Error: method not supported!")
            exit(-1)
        return pred_y