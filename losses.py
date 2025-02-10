import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reconstruction_loss(x, x_recon):
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    return recon_loss

def perturbation_loss_input_triplet(x, x_recon, x_anom, min_margin, max_anom_dist):

    dist_recon = F.mse_loss(x, x_recon, reduction='none').mean(dim=(1, 2))  # Per sample
    dist_anom = F.mse_loss(x, x_anom, reduction='none').mean(dim=(1, 2))  # Per sample
    
    perturb_loss = torch.clamp(dist_recon - dist_anom + min_margin, min=0.0)
    anom_constraint = torch.clamp(dist_anom - max_anom_dist, min=0.0)
    
    total_pert = perturb_loss.sum() + anom_constraint.sum()

    return total_pert

def perturbation_zero_reg_loss(x, x_anom):

    all_zero_dimensions = (x.sum(dim=(0, 1)) == 0).nonzero(as_tuple=True)[0]

    x_zero = x[:, :, all_zero_dimensions]
    x_anom_zero = x_anom[:, :, all_zero_dimensions]
    dist = F.mse_loss(x_anom_zero, x_zero)

    zero_perturbation_loss = 1.0 / (dist.mean() + 1.0)

    return zero_perturbation_loss

def kl_en_loss(logvar, mu, sigma_prior):

    sigma_prior_I = torch.tensor(sigma_prior, device=mu.device)
    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - torch.exp(logvar) / (sigma_prior_I ** 2) - 2 * torch.log(sigma_prior_I)
    )
    return kl_loss