import os
import numpy as np
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR, StepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TCVAE(nn.Module):
    def __init__(self, input_dim, seq_len, latent_dim, hidden_dim, batch_size):
        super(TCVAE, self).__init__()
        
        self.input_dim = input_dim  # Number of features in the time series
        self.seq_len = seq_len  # Length of the time series
        self.latent_dim = latent_dim  # Dimensionality of the latent space
        self.hidden_dim = hidden_dim  # Dimensionality of hidden layers
        self.dropout = nn.Dropout(p=0.1)
        k_size = 3
        strd = 1
        padd = 1

        # Encoder: 1D Conv layers followed by fully connected layer for mean and log variance
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size = k_size, stride= strd, padding=padd),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 4, kernel_size= k_size, stride= strd, padding=padd),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim * 2, out_channels=hidden_dim * 4, kernel_size= k_size, stride= strd, padding=padd),
            nn.ReLU()
        )
        
        # Calculate the output size of the last Conv layer
        conv_output_size = self._get_conv_output_size(seq_len, batch_size)
        
        # Fully connected layers to get mean and log variance
        self.fc_mu = nn.Linear(hidden_dim * 4 * conv_output_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 4 * conv_output_size, latent_dim)
        
        # Fully connected decoder to match the input size for ConvTranspose layers
        self.fc_decoder = nn.Linear(latent_dim, hidden_dim * 4 * conv_output_size)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=hidden_dim * 4, out_channels=hidden_dim * 2, kernel_size= k_size, stride= strd, padding=padd),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=hidden_dim * 8, out_channels=hidden_dim, kernel_size= k_size, stride= strd, padding=padd),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=hidden_dim, out_channels=input_dim, kernel_size= k_size, stride= strd, padding=padd),
            nn.ReLU()  # Sigmoid to ensure output is in range (0, 1)
        )
        
        self.perturb_scale = nn.Parameter(torch.rand(latent_dim) * 2.0 - 1.0)
    
    def _get_conv_output_size(self, seq_len, batch_size):
        # Simulate passing an input through the conv layers to get the output size
        with torch.no_grad():
            x = torch.zeros(batch_size, self.input_dim, seq_len)  # input shape is (batch_size, input_dim, seq_len)
            x = self.encoder(x)
            return x.size(-1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def reparameterize_pert(self, mu, logvar, perturb_scale_):
        logvar = (perturb_scale_ * logvar)
        std = torch.exp(0.5 * logvar) 
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape the input to (batch_size, input_dim, seq_len) for Conv1d
        x = x.permute(0, 2, 1)  # Changes (batch_size, seq_len, input_dim) to (batch_size, input_dim, seq_len)

        # Encoding
        x = self.encoder(x)  # Shape: (batch_size, hidden_dim*4, conv_output_size)
        x = x.view(batch_size, -1)  # Flatten for fully connected layers
        
        # Latent space
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        z = self.dropout(z)
        
        # Decoding
        x = self.fc_decoder(z)
        x = x.view(batch_size, self.hidden_dim * 4, -1)  # Reshape back for transposed conv layers
        x_recon = self.decoder(x)  # Reconstructed output
        
        # Reshape back to (batch_size, seq_len, input_dim)
        x_recon = x_recon.permute(0, 2, 1)  # Changes back to (batch_size, seq_len, input_dim)
        
        
        # ==== Perturbation =====
        z_pert = self.reparameterize_pert(mu, logvar, self.perturb_scale)
        
        # Decoding
        x_pert = self.fc_decoder(z_pert)
        x_pert = x_pert.view(batch_size, self.hidden_dim * 4, -1)  # Reshape back for transposed conv layers
        x_anom = self.decoder(x_pert)  # Reconstructed output
        
        # Reshape back to (batch_size, seq_len, input_dim)
        x_anom = x_anom.permute(0, 2, 1)
        
        return x_recon, mu, logvar, z, x_anom, z_pert


