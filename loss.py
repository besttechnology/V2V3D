from torch.fft import fft2
from torch.nn.functional import l1_loss,relu
from utils import calBacknoise
import torch

def posloss(xguess):

    return torch.sum(relu(-xguess))

def fftloss(x,y):
    conv_x = fft2(x)
    conv_y = fft2(y)
    loss = l1_loss(conv_x,conv_y)
    return loss

def deCrosstalk_loss(centerview,xguess,psf_energy_mean):
    if xguess.ndim == 4: xguess = torch.squeeze(xguess,0)
    xguess_backnoise = calBacknoise(centerview)/psf_energy_mean
    loss = torch.mean(relu(xguess_backnoise-xguess))
    return loss

def zloss(xguess):
    if xguess.ndim ==3: zloss = torch.mean(torch.abs(2*xguess[1:-1,...]-xguess[:-2,:,:]-xguess[2:,:,:]))
    else: zloss = torch.mean(torch.abs(2*xguess[:,1:-1,...]-xguess[:,:-2,:,:]-xguess[:,2:,:,:]))
    return zloss


def gauss_scale_loss(scales, s_max):
    """尺度超过 s_max 的超量平方和。scales: (N, 3)"""
    over = relu(scales - s_max)
    return torch.mean(over ** 2)


def gauss_sparse_loss(densities):
    """L1 稀疏正则。densities: (N, 1) 或 (N,)"""
    return torch.mean(torch.abs(densities))


def gauss_aniso_loss(scales, kappa=5.0, eps=1e-6):
    """各向异性正则：惩罚最大/最小 scale 比超过 kappa。scales: (N, 3)"""
    s_max = scales.max(dim=-1).values
    s_min = scales.min(dim=-1).values.clamp_min(eps)
    ratio = s_max / s_min
    return torch.mean(relu(ratio - kappa) ** 2)


def gauss_total_reg(gp_list, lambda_s=0.01, lambda_sp=1e-3, lambda_a=1e-4,
                    s_max=2.0, kappa=5.0):
    """对一组分支的 Gaussian 参数同时施加 3 项正则。

    gp_list: list of dict(positions, densities, scales, rotations)
    """
    total = 0
    for gp in gp_list:
        total = total + lambda_s * gauss_scale_loss(gp['scales'], s_max)
        total = total + lambda_sp * gauss_sparse_loss(gp['densities'])
        if lambda_a > 0:
            total = total + lambda_a * gauss_aniso_loss(gp['scales'], kappa)
    return total