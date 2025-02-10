import os
from models import TCVAE
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from sklearn.preprocessing import RobustScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def add_noise_to_zeros(data, noise_scale=1e-3):
    noise = (data == 0).astype(float) * np.random.randn(*data.shape) * noise_scale
    return data + noise

def create_windows_min_max(data, window_size, step_size):
    data = np.nan_to_num(data)

    min_column = np.amin(data, axis=0)
    max_column = np.amax(data, axis=0)
    dmin, dmax = min_column, max_column    
    epsilon = 1e-20
    range_column = (dmax - dmin) + epsilon
    data = (data - dmin) / range_column
    
    num_windows = (data.shape[0] - window_size) // step_size + 1
    windows = np.zeros((num_windows, window_size, data.shape[1]))

    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        windows[i] = data[start:end]

    return windows

def create_windows_min_max_2(data, window_size=200, step_size=1, tmin=None, tmax=None, is_label=False):
    data = np.nan_to_num(data)

    if is_label:
        dmin, dmax = 0, 0
        num_windows = (data.shape[0] - window_size) // step_size + 1
        windows = np.zeros(num_windows)

        for i in range(num_windows):
            start = i * step_size
            end = start + window_size
            windows[i] = max(data[start:end])
    else:
        if tmin is None:
            min_column = np.amin(data, axis=0)
            max_column = np.amax(data, axis=0)
            dmin, dmax = min_column, max_column
        else:
            dmin, dmax = tmin, tmax
        epsilon = 1e-20
        range_column = (dmax - dmin) + epsilon
        data = (data - dmin) / range_column

        # print('shape --> ', data.shape)
        
        num_windows = (data.shape[0] - window_size) // step_size + 1
        windows = np.zeros((num_windows, window_size, data.shape[1]))

        for i in range(num_windows):
            start = i * step_size
            end = start + window_size
            windows[i] = data[start:end]

    return windows, dmin, dmax

def create_windows_robust(data, window_size, step_size):
    scaler = RobustScaler()
    data = np.nan_to_num(data)
    data = scaler.fit_transform(data)
    
    num_windows = (data.shape[0] - window_size) // step_size + 1
    windows = np.zeros((num_windows, window_size, data.shape[1]))

    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        windows[i] = data[start:end]

    return windows

def create_windows_log_transform(data, window_size, step_size):
    data = np.nan_to_num(data)
    data = np.log1p(data)  # Log transform

    num_windows = (data.shape[0] - window_size) // step_size + 1
    windows = np.zeros((num_windows, window_size, data.shape[1]))

    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        windows[i] = data[start:end]

    return windows

def create_windows_local_normalization(data, window_size, step_size):
    data = np.nan_to_num(data)
    num_windows = (data.shape[0] - window_size) // step_size + 1
    windows = np.zeros((num_windows, window_size, data.shape[1]))

    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        window = data[start:end]
        min_val = np.min(window, axis=0)
        max_val = np.max(window, axis=0)
        range_val = (max_val - min_val) + 1e-20
        windows[i] = (window - min_val) / range_val

    return windows


def create_windows_mean_std(data, window_size, step_size):
    data = np.nan_to_num(data)
    std = np.std(data, axis=0)
    mean = np.mean(data, axis=0)
    std = np.where(std == 0, 1e-20, std)
    data = (data - mean) / std
    
    num_windows = (data.shape[0] - window_size) // step_size + 1
    windows = np.zeros((num_windows, window_size, data.shape[1]))

    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        windows[i] = data[start:end]

    return windows

def saveBestGenModel(entity, epochs, batch_size, learning_rate, normal_windows, hid_size, latent_dim, loss_function, savedir):
    
    dataloader = DataLoader(TensorDataset(torch.tensor(normal_windows, dtype=torch.float32).to(device)), 
                            batch_size=batch_size, shuffle=True)

    seq_len = normal_windows.shape[1]
    input_dim = normal_windows.shape[2]

    print(f'Entity: {entity} \n ==================')
    model = TCVAE(input_dim=input_dim, seq_len=seq_len, latent_dim=latent_dim, hidden_dim=hid_size, batch_size=batch_size).to(device)
    best_model = TCVAE(input_dim=input_dim, seq_len=seq_len, latent_dim=latent_dim, hidden_dim=hid_size, batch_size=batch_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = StepLR(optimizer, step_size=100 ,gamma=0.95)

    # Training loop
    losses = []
    center = []
    radius_threshold = 0
    robust_radius = 0
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        aggregated_losses = None

        for i, batch in enumerate(dataloader):
            x = batch[0].to(device)
            x_hat, mu, log_var, latent_vectors, x_anom, z_pert = model(x)

            loss_dict = loss_function(x, x_hat, mu, log_var, latent_vectors, z_pert, x_anom)
            loss = loss_dict['total_loss']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if aggregated_losses is None:
                aggregated_losses = {key: 0.0 for key in loss_dict.keys()}

            for key, value in loss_dict.items():
                aggregated_losses[key] += value.item()

        if aggregated_losses["total_loss"] < best_loss:
            best_loss = aggregated_losses["total_loss"]
            best_model = model

        if (epoch % 100) == 0:
            loss_output = '\t'.join([f'{key}: {value:.6f}' for key, value in aggregated_losses.items()])
            print(f'Epoch [{epoch + 1}/{epochs}] \n {loss_output}')
            
        losses.append(aggregated_losses["total_loss"])
        
        scheduler.step()
    
    print('Done!')
    print('Best loss:', best_loss)

    save_dir = f'gen_models/{savedir}'
    model_filename = f'gen_anom_{entity}.pth'
    save_path = os.path.join(save_dir, model_filename)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_save_dict = {
        'model_state_dict': best_model.state_dict(),
        'radius_threshold': radius_threshold,
        'robust_radius': robust_radius,
        'center' : center
    }
    torch.save(model_save_dict, save_path)
    print('radius threshold:', radius_threshold)
    print('robust radius:', robust_radius)

    return losses