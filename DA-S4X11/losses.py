# reference.
# 1. https://github.com/thuml/Xlearn/blob/master/pytorch/src/loss.py
# 2. https://github.com/MaterialsInformaticsDemo/DAN/blob/main/code/MK_MMD.py
# 3. https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/alignment/dan.py
import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.optim as optim
import pandas as pd
from torchsummary import summary
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import init
import torch.nn.functional as F
from sklearn.model_selection import ParameterGrid
import time


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """Generate Gaussian kernel matrix given input 'source' and 'target'
    """
    # The simplest version: K = \beta * k where \beta=1
    n_samples_source = source.size(0)
    n_samples_target = target.size(0)
    total = torch.cat([source, target], dim=0)

    # Calculate the L2 distance matrix efficiently using matrix operations.
    total_xx = torch.sum(total * total, dim=1, keepdim=True)
    L2_distance = total_xx - 2.0 * torch.matmul(total, total.t()) + total_xx.t()
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        # Set the average value of distance matrix as the bandwidth
        bandwidth = torch.sum(L2_distance.data) / (n_samples_source * n_samples_target - n_samples_source)

    bandwidth /= kernel_mul ** (kernel_num // 2) #
    bandwidth_list = [bandwidth * (kernel_mul ** (1 * i)) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val) # final gaussian kernel matrix



def MK_MMD(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=1.68): # fix_sigma from None to 1.68
    """Compute the Multi-kernel MMD given 'source' and 'target'
    """
    n_s, n_t = source.size(0), target.size(0)

    kernels = gaussian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    # Efficiently calculate loss components using vectorized operations.
    SS = torch.triu(kernels[:n_s, :n_s], diagonal=1).sum() / float(n_s * n_s)
    TT = torch.triu(kernels[-n_t:, -n_t:], diagonal=1).sum() / float(n_t * n_t)
    ST = -kernels[:n_s, -n_t:].sum() / float(n_s * n_t)
    TS = -kernels[-n_t:, :n_s].sum() / float(n_s * n_t)

    loss = torch.abs(SS + TT - ST - TS)

    return loss

def MMD(Xs, Xt):
    """ Compute the MMD distance given source domain(Xs) and target domain(Xt)
    """
    ns, nt = Xs.size(0), Xt.size(0)

    # linear kernel version
    mmd_s = (Xs @ Xs.t()).sum() / (ns * ns)
    mmd_t = (Xt @ Xt.t()).sum() / (nt * nt)
    mmd_st = (Xs @ Xt.t()).sum() * 2.0 / (ns * nt)
    loss = mmd_s - mmd_st + mmd_t

    return torch.abs(loss)


def JMMD_Linear(source_list, target_list, kernel_muls=[2.0, 2.0, 2.0], kernel_nums=[5, 5, 1], fix_sigma_list=[None, None, 1.68]):
    """ Compute the Joint MMD(Linear version) given source_list and target_list(contains the outputs of multiple layers)
    """
    batch_size = int(source_list[0].size()[0])
    layer_num = len(source_list)
    joint_kernels = None
    for i in range(layer_num):
        source = source_list[i]
        target = target_list[i]
        kernel_mul = kernel_muls[i]
        kernel_num = kernel_nums[i]
        fix_sigma = fix_sigma_list[i]
        kernels = gaussian_kernel(source, target,
            kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        if joint_kernels is not None:
            joint_kernels = joint_kernels * kernels
        else:
            joint_kernels = kernels

    s1 = torch.arange(batch_size)
    s2 = (s1 + 1) % batch_size
    t1 = s1 + batch_size
    t2 = s2 + batch_size

    # Compute the loss in a vectorized way
    loss = torch.sum(joint_kernels[s1, s2] + joint_kernels[t1, t2] - joint_kernels[s1, t2] - joint_kernels[s2, t1])
    return torch.abs(loss) / float(batch_size) # loss might be negative



def _primal_kernel(Xs, Xt):
    Z = torch.cat((Xs.T, Xt.T), 1)  # Xs / Xt: batch_size * k
    return Z


def _linear_kernel(Xs, Xt):
    Z = torch.cat((Xs, Xt), 0)  # Xs / Xt: batch_size * k
    K = torch.mm(Z, Z.T)
    return K


def _rbf_kernel(Xs, Xt, sigma):
    if Xs.ndim == 1:
        Xs = Xs.unsqueeze(1)
    if Xt.ndim == 1:
        Xt = Xt.unsqueeze(1)
    Z = torch.cat((Xs, Xt), 0)
    ZZT = torch.mm(Z, Z.T)
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.T
    K = torch.exp(-exponent / (2 * sigma ** 2))
    return K

# functions to compute the marginal MMD with rbf kernel
def rbf_mmd(Xs, Xt, sigma):
    device = Xs.device

    K = _rbf_kernel(Xs, Xt, sigma)
    m = Xs.size(0)  # assume Xs, Xt are same shape
    e = torch.cat((1 / m * torch.ones(m, 1), -1 / m * torch.ones(m, 1)), 0).to(device)
    M = e * e.T
    tmp = torch.mm(torch.mm(K, M), K.T)
    loss = torch.trace(tmp).to(device)
    return loss

# functions to compute rbf kernel JMMD
def rbf_jmmd(Xs, Ys, Xt, Yt0, sigma):
    device = Xs.device

    K = _rbf_kernel(Xs, Xt, sigma)
    n = K.size(0)
    m = Xs.size(0)  # assume Xs, Xt are same shape
    e = torch.cat((1 / m * torch.ones(m, 1), -1 / m * torch.ones(m, 1)), 0).to(device)
    C = len(torch.unique(Ys))
    M = e * e.T * C
    for c in torch.unique(Ys):
        e = torch.zeros(n, 1, device=device)
        e[:m][Ys == c] = 1 / len(Ys[Ys == c])
        if len(Yt0[Yt0 == c]) == 0:
            e[m:][Yt0 == c] = 0
        else:
            e[m:][Yt0 == c] = -1 / len(Yt0[Yt0 == c])
        M = M + e * e.T
    M = M / torch.norm(M, p='fro')  # can reduce the training loss only for jmmd
    tmp = torch.mm(torch.mm(K, M), K.T)
    loss = torch.trace(tmp).to(device)
    return loss


def rbf_jpmmd(Xs, Ys, Xt, Yt0, sigma):
    device = Xs.device

    K = _rbf_kernel(Xs, Xt, sigma)
    n = K.size(0)
    m = Xs.size(0)  # assume Xs, Xt are same shape
    M = 0
    for c in torch.unique(Ys):
        e = torch.zeros(n, 1, device=device)
        e[:m] = 1 / len(Ys)
        if len(Yt0[Yt0 == c]) == 0:
            e[m:] = 0
        else:
            e[m:] = -1 / len(Yt0)
        M = M + e * e.T
    tmp = torch.mm(torch.mm(K, M), K.T)
    loss = torch.trace(tmp).to(device)
    return loss


# functions to compute rbf kernel DJP-MMD
def rbf_djpmmd(Xs, Ys, Xt, Yt0, sigma):
    # Assuming _rbf_kernel is already optimized and running on the correct device
    K = _rbf_kernel(Xs, Xt, sigma)
    m, C = Xs.size(0), 2  # Assuming number of classes C is fixed at 2

    # Ensure all tensors start on the same device, ideally on the GPU if available
    device = Xs.device
    Ns = torch.zeros(m, C, device=device).scatter_(1, Ys.unsqueeze(1), 1) / m
    Nt = torch.zeros(m, C, device=device)
    if len(torch.unique(Yt0)) == 1:
        Nt = torch.zeros(m, C, device=device).scatter_(1, Yt0.unsqueeze(1), 1) / m

    Rmin_1 = torch.cat((torch.mm(Ns, Ns.T), torch.mm(-Ns, Nt.T)), 0)
    Rmin_2 = torch.cat((torch.mm(-Nt, Ns.T), torch.mm(Nt, Nt.T)), 0)
    Rmin = torch.cat((Rmin_1, Rmin_2), 1)

    # For discriminability
    Ms = torch.empty(m, (C - 1) * C).to(device)
    Mt = torch.empty(m, (C - 1) * C).to(device)
    for i in range(0, C):
        idx = torch.arange((C - 1) * i, (C - 1) * (i + 1))
        Ms[:, idx] = Ns[:, i].repeat(C - 1, 1).T
        tmp = torch.arange(0, C)
        Mt[:, idx] = Nt[:, tmp[tmp != i]]
    Rmax_1 = torch.cat((torch.mm(Ms, Ms.T), torch.mm(-Ms, Mt.T)), 0)
    Rmax_2 = torch.cat((torch.mm(-Mt, Ms.T), torch.mm(Mt, Mt.T)), 0)
    Rmax = torch.cat((Rmax_1, Rmax_2), 1)

    M = Rmin - 0.1 * Rmax
    # M = Rmin + Rmax
    # Operate in the same device as K and M to avoid device transfers
    tmp = torch.mm(torch.mm(K, M), K.T).to(device)
    loss = torch.trace(tmp)

    return loss


def compute_sigma(H):
    dists = torch.pdist(H)
    sigma = dists.median() / 2

    return sigma.detach()

def GaussianMatrix(X, Y, sigma, if_use_cdist=False, median_sigma = False):
    """ Compute the gaussian kernel matrix given X and Y
    """
    if not if_use_cdist:
        size1 = X.size()
        size2 = Y.size()
        G = (X*X).sum(-1)
        H = (Y*Y).sum(-1)
        Q = G.unsqueeze(-1).repeat(1,size2[0])
        R = H.unsqueeze(-1).T.repeat(size1[0],1)
        H = Q + R - 2 * X @ (Y.T)
    else:
        H = torch.cdist(X, Y, p=2) ** 2

    if sigma > 0:
        H = torch.exp(-H / 2 / sigma ** 2)
    else:
        if median_sigma:
            sigma = compute_sigma(H)
            H = torch.exp(-H / 2 / sigma / 2)
        else:
            sigma = H.mean().detach()
            H = torch.exp(-H / sigma)

    return H

def CS(x1, x2, sigma = 10, if_use_cdist=False, median_sigma=False):
    """ Compute the CS divergence given source domain(x1) and target domain(x2)
    """
    K1 = GaussianMatrix(x1, x1, sigma, if_use_cdist, median_sigma)
    K2 = GaussianMatrix(x2, x2, sigma, if_use_cdist, median_sigma)

    K12 = GaussianMatrix(x1, x2, sigma, if_use_cdist, median_sigma)

    dim1 = K1.shape[0]
    self_term1 = K1.sum() / (dim1**2)

    dim2 = K2.shape[0]
    self_term2 = K2.sum() / (dim2**2)

    cross_term = K12.sum() / (dim1*dim2)

    cs =  -2 * torch.log(cross_term + 1e-10) + torch.log(self_term1 + 1e-10) + torch.log(self_term2 + 1e-10)

    return cs


def CCS(x1, x2, y1, y2, sigma=1, if_use_cdist=False, median_sigma=False):
    """ Compute the CS+CCS divergence given source domain(x1, y1) and target domain(x2, y2) where y2 is the pseudo-labels
    """
    # Input dimension: N x d

    K1 = _rbf_kernel(x1, x1, sigma)
    K2 = _rbf_kernel(x2, x2, sigma)
    L1 = _rbf_kernel(y1, y1, sigma)
    L2 = _rbf_kernel(y2, y2, sigma)

    K12 = _rbf_kernel(x1, x2, sigma)
    L12 = _rbf_kernel(y1, y2, sigma)

    K21 = _rbf_kernel(x2, x1, sigma)
    L21 = _rbf_kernel(y2, y1, sigma)

    H1 = K1 * L1
    self_term1 = (H1.sum(-1) / ((K1.sum(-1)) ** 2)).sum(0)
    assert not torch.isnan(self_term1).any(), "self_term1 contains NaN"

    H2 = K2 * L2
    self_term2 = (H2.sum(-1) / ((K2.sum(-1)) ** 2)).sum(0)
    assert not torch.isnan(self_term2).any(), 'self_term2 contains NaN'

    H3 = K12 * L12
    cross_term1 = (H3.sum(-1) / ((K1.sum(-1)) * (K12.sum(-1)))).sum(0)

    assert not torch.isnan(cross_term1).any(), 'cross_term1 contains NaN'

    H4 = K21 * L21
    cross_term2 = (H4.sum(-1) / ((K2.sum(-1))*(K21.sum(-1)))).sum(0)
    assert not torch.isnan(cross_term1).any(), 'cross_term2 contains NaN'

    ccs = -torch.log(cross_term1 + 1e-10) - torch.log(cross_term2 + 1e-10) + torch.log(self_term1 + 1e-10) + torch.log(self_term2 + 1e-10)

    return ccs