import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

SQRT_CONST = 1e-10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def mmd2_lin(X,t,p):
    ''' Linear MMD '''

    it = torch.where(t>0)[0]
    ic = torch.where(t<1)[0]

    Xc = X[ic]
    Xt = X[it]

    mean_control = torch.mean(Xc, dim=0)
    mean_treated = torch.mean(Xt, dim=0)

    # 确保p是tensor类型
    if not isinstance(p, torch.Tensor):
        p = torch.tensor(p, device=X.device, dtype=X.dtype)
    
    mmd = torch.sum(torch.square(2.0*p*mean_treated - 2.0*(1.0-p)*mean_control))

    return mmd

def mmd2_rbf(X,t,p,sig):
    """ Computes the l2-RBF MMD for X given t """

    it = torch.where(t>0)[0]
    ic = torch.where(t<1)[0]

    Xc = X[ic]
    Xt = X[it]

    # 确保p和sig是tensor类型
    if not isinstance(p, torch.Tensor):
        p = torch.tensor(p, device=X.device, dtype=X.dtype)
    if not isinstance(sig, torch.Tensor):
        sig = torch.tensor(sig, device=X.device, dtype=X.dtype)

    Kcc = torch.exp(-pdist2sq(Xc,Xc)/torch.square(sig))
    Kct = torch.exp(-pdist2sq(Xc,Xt)/torch.square(sig))
    Ktt = torch.exp(-pdist2sq(Xt,Xt)/torch.square(sig))

    m = float(Xc.shape[0])
    n = float(Xt.shape[0])

    mmd = torch.square(1.0-p)/(m*(m-1.0))*(torch.sum(Kcc)-m)
    mmd = mmd + torch.square(p)/(n*(n-1.0))*(torch.sum(Ktt)-n)
    mmd = mmd - 2.0*p*(1.0-p)/(m*n)*torch.sum(Kct)
    mmd = 4.0*mmd

    return mmd

def pdist2sq(X,Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2*torch.matmul(X,torch.transpose(Y, 0, 1))
    nx = torch.sum(torch.square(X), 1, keepdim=True)
    ny = torch.sum(torch.square(Y), 1, keepdim=True)
    D = (C + torch.transpose(ny, 0, 1)) + nx
    return D
def safe_sqrt(x, lbound=SQRT_CONST):
    ''' Numerically safe version of PyTorch sqrt '''
    return torch.sqrt(torch.clamp(x, min=lbound, max=float('inf')))

def pdist2(X,Y):
    """ Returns the pytorch pairwise distance matrix """
    return safe_sqrt(pdist2sq(X,Y))

def pop_dist(X,t):
    it = torch.where(t>0)[0]
    ic = torch.where(t<1)[0]
    Xc = X[ic]
    Xt = X[it]
    nc = float(Xc.shape[0])
    nt = float(Xt.shape[0])

    ''' Compute distance matrix'''
    M = pdist2(Xt,Xc)
    return M

def wasserstein(X,t,p,lam=10,its=10,sq=False,backpropT=False):
    """ Returns the Wasserstein distance between treatment groups """

    it = torch.where(t>0)[0]
    ic = torch.where(t<1)[0]
    Xc = X[ic]
    Xt = X[it]
    nc = float(Xc.shape[0])
    nt = float(Xt.shape[0])

    # 确保p是tensor类型
    if not isinstance(p, torch.Tensor):
        p = torch.tensor(p, device=X.device, dtype=X.dtype)

    ''' Compute distance matrix'''
    if sq:
        M = pdist2sq(Xt,Xc)
    else:
        M = safe_sqrt(pdist2sq(Xt,Xc))

    ''' Estimate lambda and delta '''
    M_mean = torch.mean(M)
    M_drop = F.dropout(M, p=1.0-10/(nc*nt), training=True)
    delta = torch.max(M).detach()
    eff_lam = (lam/M_mean).detach()

    ''' Compute new distance matrix '''
    Mt = M
    row = delta*torch.ones(M[0:1,:].shape, device=M.device, dtype=M.dtype)
    col = torch.cat([delta*torch.ones(M[:,0:1].shape, device=M.device, dtype=M.dtype), torch.zeros((1,1), device=M.device, dtype=M.dtype)], dim=0)
    Mt = torch.cat([M,row], dim=0)
    Mt = torch.cat([Mt,col], dim=1)

    ''' Compute marginal vectors '''
    # 确保维度匹配：将1维张量转换为2维
    treated_indices = torch.where(t>0)[0]
    control_indices = torch.where(t<1)[0]
    
    a = torch.cat([p*torch.ones((treated_indices.shape[0], 1), device=X.device, dtype=X.dtype)/nt, (1-p)*torch.ones((1,1), device=X.device, dtype=X.dtype)], dim=0)
    b = torch.cat([(1-p)*torch.ones((control_indices.shape[0], 1), device=X.device, dtype=X.dtype)/nc, p*torch.ones((1,1), device=X.device, dtype=X.dtype)], dim=0)

    ''' Compute kernel matrix'''
    Mlam = eff_lam*Mt
    K = torch.exp(-Mlam) + 1e-6 # added constant to avoid nan
    U = K*Mt
    ainvK = K/a

    u = a
    for i in range(0,its):
        u = 1.0/(torch.matmul(ainvK,(b/torch.transpose(torch.matmul(torch.transpose(u, 0, 1),K), 0, 1))))
    v = b/(torch.transpose(torch.matmul(torch.transpose(u, 0, 1),K), 0, 1))

    T = u*(torch.transpose(v, 0, 1)*K)

    if not backpropT:
        T = T.detach()

    E = T*Mt
    D = 2*torch.sum(E)

    return D, Mlam

class EarlyStopper:
    """
    通用的早停器类，用于神经网络训练
    
    参考DragonNet的实现方式，提供简单有效的早停机制
    """
    def __init__(self, patience=15):
        """
        初始化早停器
        
        Args:
            patience: 早停耐心值，连续多少轮验证损失没有改善就停止
        """
        self.patience = patience
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        """
        检查是否应该早停
        
        Args:
            validation_loss: 当前验证损失
            
        Returns:
            bool: 是否应该早停
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > self.min_validation_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False