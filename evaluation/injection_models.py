import sys
import os

current_dir = os.getcwd()
parent_of_parent = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_of_parent)

import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models import TCVAE
from train_utils import *
from losses import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====================== GenIAS ======================

def genias_patching(windows_sampled, generated_abnormal_windows, threshold):
    """
    Apply the patched_by_threshold logic for each window and dimension in a batch using GPU acceleration.

    Args:
        windows_sampled (torch.Tensor): Original windows, shape (batch_size, window_length, dims).
        generated_abnormal_windows (torch.Tensor): Generated abnormal windows, shape (batch_size, window_length, dims).
        threshold (float): Threshold for deviations.

    Returns:
        torch.Tensor: Patched windows, shape (batch_size, window_length, dims).
        list: List of deviation indices for each window and dimension.
    """
    # Ensure tensors are on the same device
    # device = windows_sampled.device
    generated_abnormal_windows = generated_abnormal_windows.to(device)

    # Vectorized processing for amplitude and differences on GPU
    amp = torch.max(windows_sampled, dim=1, keepdim=True).values - torch.min(windows_sampled, dim=1, keepdim=True).values
    diff = (windows_sampled - generated_abnormal_windows).abs() ** 1.0

    epsilon = 1e-10
    # Identify deviations beyond the threshold
    deviations = diff > (threshold * amp) #(epsilon + np.random.rand()*threshold * amp)
    patched_windows = torch.where(deviations, generated_abnormal_windows, windows_sampled)

    # Collect deviation indices per window and dimension
    deviation_indices_batch = []
    for b in range(windows_sampled.size(0)):
        deviation_indices_window = []
        for d in range(windows_sampled.size(2)):
            deviation_indices = torch.nonzero(deviations[b, :, d], as_tuple=False).squeeze().cpu().numpy()
            deviation_indices_window.append(deviation_indices)
        deviation_indices_batch.append(deviation_indices_window)

    return patched_windows, deviation_indices_batch

def genias_injection(x, fname, savedir, scale, tau):

    # print(type(x), x.shape)

    x_copy = x.clone()

    # Define the input_size, feature_size, hidden_size, and latent_size
    seq_len = x.shape[1]
    input_dim = x.shape[2]
    if (input_dim == 1):
        hid_size = 100
        latent_dim = 50
    else:
        hid_size = 200
        latent_dim = 100
    batch_size_init = 50

    # Load the saved model
    model = TCVAE(input_dim=input_dim, seq_len=seq_len, latent_dim=latent_dim, hidden_dim=hid_size, batch_size=batch_size_init)
    model.load_state_dict(torch.load(f"../gen_models/{savedir}/gen_anom_{fname}.pth")['model_state_dict'])

    # Use the loaded model
    model.eval()

    batch_size = x.shape[0]
    
    with torch.no_grad():
        model = model.to(device)
        # x = torch.tensor(x.clone().detach(), dtype=torch.float32).to(device)
        # Reshape the input to (batch_size, input_dim, seq_len) for Conv1d
        x = x.permute(0, 2, 1)  # Changes (batch_size, seq_len, input_dim) to (batch_size, input_dim, seq_len)

        # Encoding
        x = model.encoder(x)  # Shape: (batch_size, hidden_dim*4, conv_output_size)
        x = x.view(batch_size, -1)  # Flatten for fully connected layers
        
        # Latent space
        mu = model.fc_mu(x)
        logvar = model.fc_logvar(x)
        z = model.reparameterize(mu, logvar)
        z = model.dropout(z)
        
        # Decoding
        x = model.fc_decoder(z)
        x = x.view(batch_size, model.hidden_dim * 4, -1)  # Reshape back for transposed conv layers
        x_recon = model.decoder(x)  # Reconstructed output
        
        # Reshape back to (batch_size, seq_len, input_dim)
        x_recon = x_recon.permute(0, 2, 1)  # Changes back to (batch_size, seq_len, input_dim)
        
        
        # ==== Perturbation =====
        z_pert = model.reparameterize_pert(mu, logvar, model.perturb_scale*scale)
        
        # Decoding
        x_pert = model.fc_decoder(z_pert)
        x_pert = x_pert.view(batch_size, model.hidden_dim * 4, -1)  # Reshape back for transposed conv layers
        x_anom = model.decoder(x_pert)  # Reconstructed output
        
        # Reshape back to (batch_size, seq_len, input_dim)
        x_anom = x_anom.permute(0, 2, 1)  # Changes back to

        # print(x_anom.shape, x_copy.shape)

    x_anom, _ = genias_patching(x_copy, x_anom, tau)
    
    return x_anom, x_recon, z, z_pert

# ====================== COUTA ======================

def COUTA_create_neg(batch_seqs, max_cut_ratio=0.5, seed=0, return_mul_label=False, ss_type='FULL'):
        """
        Creates a batch of negative samples.

        Args:
        
            batch_seqs (numpy.ndarray): 
                The batch of sequences.
                
            max_cut_ratio (float, optional): 
                The maximum cut ratio. Default is 0.5.
                
            seed (int, optional): 
                The seed for the random number generator. Default is 0.
                
            return_mul_label (bool, optional): 
                Whether to return multiple labels. Default is False.
                
            ss_type (str, optional): 
                The type of the subsequence. Default is 'FULL'.

        Returns:
        
            tuple: 
                The batch of negative samples and their labels.
        """
        
        rng = np.random.RandomState(seed=seed)

        batch_size, l, dim = batch_seqs.shape
        cut_start = l - rng.randint(1, int(max_cut_ratio * l), size=batch_size)
        n_cut_dim = rng.randint(1, dim+1, size=batch_size)
        cut_dim = [rng.randint(dim, size=n_cut_dim[i]) for i in range(batch_size)]

        if type(batch_seqs) == np.ndarray:
            batch_neg = batch_seqs.copy()
            neg_labels = np.zeros(batch_size, dtype=int)
        else:
            batch_neg = batch_seqs.clone()
            neg_labels = torch.LongTensor(batch_size)

        if ss_type != 'FULL':
            pool = rng.randint(1e+6, size=int(1e+4))
            if ss_type == 'collective':
                pool = [a % 6 == 0 or a % 6 == 1 for a in pool]
            elif ss_type == 'contextual':
                pool = [a % 6 == 2 or a % 6 == 3 for a in pool]
            elif ss_type == 'point':
                pool = [a % 6 == 4 or a % 6 == 5 for a in pool]
            flags = rng.choice(pool, size=batch_size, replace=False)
        else:
            flags = rng.randint(1e+5, size=batch_size)

        n_types = 6
        for ii in range(batch_size):
            flag = flags[ii]

            # collective anomalies
            if flag % n_types == 0:
                batch_neg[ii, cut_start[ii]:, cut_dim[ii]] = 0
                neg_labels[ii] = 1

            elif flag % n_types == 1:
                batch_neg[ii, cut_start[ii]:, cut_dim[ii]] = 1
                neg_labels[ii] = 1

            # contextual anomalies
            elif flag % n_types == 2:
                mean = torch.mean(batch_neg[ii, -10:, cut_dim[ii]], dim=0)
                batch_neg[ii, -1, cut_dim[ii]] = mean + 0.5
                neg_labels[ii] = 2

            elif flag % n_types == 3:
                mean = torch.mean(batch_neg[ii, -10:, cut_dim[ii]], dim=0)
                batch_neg[ii, -1, cut_dim[ii]] = mean - 0.5
                neg_labels[ii] = 2

            # point anomalies
            elif flag % n_types == 4:
                batch_neg[ii, -1, cut_dim[ii]] = 2
                neg_labels[ii] = 3

            elif flag % n_types == 5:
                batch_neg[ii, -1, cut_dim[ii]] = -2
                neg_labels[ii] = 3

        if return_mul_label:
            return batch_neg, neg_labels
        else:
            neg_labels = torch.ones(batch_size).long()
            return batch_neg, neg_labels

# ====================== NCAD ======================

def coe_batch(x: torch.Tensor, y: torch.Tensor, coe_rate: float, suspect_window_length: int,
                random_start_end: bool = True):
    """
    Generates a batch of data with contextual outlier exposure (COE) augmentations.

    Args:
    
        x (torch.Tensor): 
            Input batch of data with dimensions (batch, ts channels, time).
        
        y (torch.Tensor): 
            Target labels for the batch.
        
        coe_rate (float): 
            The proportion of the batch to augment with COE.
        
        suspect_window_length (int): 
            The length of the window considered as suspect for anomaly.
        
        random_start_end (bool, optional): 
            Whether to permute a random subset within the suspect segment. Defaults to True.

    Returns:
    
        tuple: 
            A tuple containing the augmented data and corresponding labels.
            
    """

    if coe_rate == 0:
        raise ValueError(f"coe_rate must be > 0.")
    batch_size = x.shape[0]
    ts_channels = x.shape[1]
    oe_size = int(batch_size * coe_rate)

    # Select indices
    idx_1 = torch.arange(oe_size)
    idx_2 = torch.arange(oe_size)
    while torch.any(idx_1 == idx_2):
        idx_1 = torch.randint(low=0, high=batch_size, size=(oe_size,)).type_as(x).long()
        idx_2 = torch.randint(low=0, high=batch_size, size=(oe_size,)).type_as(x).long()

    if ts_channels > 3:
        numb_dim_to_swap = np.random.randint(low=3, high=ts_channels, size=(oe_size))
        # print(numb_dim_to_swap)
    else:
        numb_dim_to_swap = np.ones(oe_size) * ts_channels

    x_oe = x[idx_1].clone()  # .detach()
    oe_time_start_end = np.random.randint(
        low=x.shape[-1] - suspect_window_length, high=x.shape[-1] + 1, size=(oe_size, 2)
    )
    oe_time_start_end.sort(axis=1)
    # for start, end in oe_time_start_end:
    for i in range(len(idx_2)):
        # obtain the dimensons to swap
        numb_dim_to_swap_here = int(numb_dim_to_swap[i])
        dims_to_swap_here = np.random.choice(
            range(ts_channels), size=numb_dim_to_swap_here, replace=False
        )

        # obtain start and end of swap
        start, end = oe_time_start_end[i]

        # swap
        x_oe[i, dims_to_swap_here, start:end] = x[idx_2[i], dims_to_swap_here, start:end]

    # Label as positive anomalies
    y_oe = torch.ones(oe_size).type_as(y)

    return x_oe, y_oe


def mixup_batch(x: torch.Tensor, y: torch.Tensor, mixup_rate: float):
    """
    Generates a batch of data with mixup augmentations.

    Args:
    
        x (torch.Tensor): 
            Input batch of data with dimensions (batch, ts channels, time).
            
        y (torch.Tensor): 
            Target labels for the batch.
        
        mixup_rate (float): 
            The proportion of the batch to augment with mixup.

    Returns:
    
        tuple: 
            A tuple containing the mixup-augmented data and corresponding labels.
            
    """

    if mixup_rate == 0:
        raise ValueError(f"mixup_rate must be > 0.")
    batch_size = x.shape[0]
    mixup_size = int(batch_size * mixup_rate)  #

    # Select indices
    idx_1 = torch.arange(mixup_size)
    idx_2 = torch.arange(mixup_size)
    while torch.any(idx_1 == idx_2):
        idx_1 = torch.randint(low=0, high=batch_size, size=(mixup_size,)).type_as(x).long()
        idx_2 = torch.randint(low=0, high=batch_size, size=(mixup_size,)).type_as(x).long()

    # sample mixing weights:
    beta_param = float(0.05)
    beta_distr = torch.distributions.beta.Beta(
        torch.tensor([beta_param]), torch.tensor([beta_param])
    )
    weights = torch.from_numpy(np.random.beta(beta_param, beta_param, (mixup_size,))).type_as(x)
    oppose_weights = 1.0 - weights

    # Create contamination
    x_mix_1 = x[idx_1].clone()
    x_mix_2 = x[idx_1].clone()
    x_mixup = (
        x_mix_1 * weights[:, None, None] + x_mix_2 * oppose_weights[:, None, None]
    )  # .detach()

    # ydevice = y.device
    y = y.to(device)
    idx_1 = idx_1.to(device)
    idx_2 = idx_2.to(device)
    weights = weights.to(device)
    oppose_weights = oppose_weights.to(device)

    # Label as positive anomalies
    y_mixup = y[idx_1].clone() * weights + y[idx_2].clone() * oppose_weights

    return x_mixup, y_mixup

def NCAD_inject(batch_x, coe_rate, mixup_rate, suspect_win_len):
    x0 = batch_x.to(device)
    y0 = torch.zeros(len(batch_x)).to(device)
    # print('len: ',  x0.shape, y0.shape)
    if coe_rate > 0:
        x_oe, y_oe = coe_batch(
            x=x0.transpose(2, 1),
            y=y0,
            coe_rate=coe_rate,
            suspect_window_length=suspect_win_len,
            random_start_end=True,
        )
        # Add COE to training batch
        x0 = x_oe.transpose(2, 1) #torch.cat((x0, x_oe.transpose(2, 1)), dim=0)
        y0 = y_oe #torch.cat((y0, y_oe), dim=0)
    
    # print('len oe: ', x0.shape, y0.shape)

    if mixup_rate > 0.0:
        x_mixup, y_mixup = mixup_batch(
            x=x0.transpose(2, 1),
            y=y0,
            mixup_rate=mixup_rate,
        )
        # Add Mixup to training batch
        x0 = x_mixup.transpose(2, 1) #torch.cat((x0, x_mixup.transpose(2, 1)), dim=0)
        y0 = y_mixup #torch.cat((y0, y_mixup), dim=0)

    # print('len mixup: ',  x0.shape, y0.shape)

    # print('len final: ', len(x0), len(y0))

    # print(y0.sum())

    return x0, y0
    
# ====================== CutAddPaste ======================

def cut_add_paste_outlier(train_x: torch.Tensor, configs: dict):
    """
    Inject outliers into a batch of time series data using the CutAddPaste method.
    
    Args:
        train_x (torch.Tensor): A 3D tensor of shape (batch_size, sequence_length, num_features),
                                expected to be on CUDA.
        configs (dict): Configuration dictionary with the following keys:
            - cut_rate (float): Minimum cut segment length.
            - dim (int): Number of features to modify in each segment.
            - trend_rate (float): Scaling factor for the added trend.
    
    Returns:
        torch.Tensor: Modified tensor with injected anomalies.
    """
    
    batch_size, seq_len, num_features = train_x.shape

    for i in range(batch_size):
        # Determine the cut segment length
        radius = max(int(configs['cut_rate']), int(torch.rand(1, device=train_x.device) * seq_len))

        # Select a random time series from the batch
        cut_random = torch.randint(0, batch_size, (1,), device=train_x.device).item()
        cut_data = train_x[cut_random].clone()  # Clone to avoid memory aliasing

        # Randomly select a segment to cut
        from_position = torch.randint(0, seq_len - radius, (1,), device=train_x.device).item()
        cut_data = cut_data[from_position:from_position + radius].clone()  # Clone again to prevent aliasing

        # Modify the selected segment by adding a slope (trend)
        if num_features > 1:
            elements = torch.arange(num_features, device=train_x.device)
            dim_size = torch.randint(1, configs['dim'] + 1, (1,), device=train_x.device).item() if configs['dim'] > 1 else 1
            selected_elements = elements[torch.randperm(num_features, device=train_x.device)[:dim_size]]  # Random feature selection

            for item in selected_elements:
                factor = torch.rand(1, device=train_x.device) * configs['trend_rate']
                slope = torch.randint(0, 2, (1,), device=train_x.device).float() * 2 - 1  # Randomly choose -1 or 1
                trend = slope * factor * torch.arange(radius, device=train_x.device).float()
                cut_data[:, item] += trend  # Add trend to selected features
        else:
            factor = torch.rand(1, device=train_x.device) * configs['trend_rate']
            slope = torch.randint(0, 2, (1,), device=train_x.device).float() * 2 - 1
            trend = slope * factor * torch.arange(radius, device=train_x.device).float()
            cut_data[:, 0] += trend  # Modify the only feature

        # Select a position in the current sample to paste the modified segment
        to_position = torch.randint(0, seq_len - radius, (1,), device=train_x.device).item()
        train_x[i, to_position:to_position + radius] = cut_data.clone()  # Clone before assignment

    return train_x

# ====================== CARLA ======================

def inject_frequency_anomaly(window, 
                             subsequence_length: int = None,
                             compression_factor: int = None,
                             scale_factor: float = None,
                             trend_factor: float = None,
                             shapelet_factor: bool = False,
                             trend_end: bool = False,
                             start_index: int = None):
    """
    Injects an anomaly into a multivariate or univariate time series window.

    :param window: The time series window (T, D) for multivariate or (T,) for univariate.
    :return: Modified window with the anomaly injected.
    """

    # Clone the input tensor to avoid modifying the original data
    window = window.clone()
    T = window.shape[0]  # Time steps

    # Determine the number of features
    D = window.shape[1] if window.ndim > 1 else 1

    # Set the subsequence_length if not provided
    if subsequence_length is None:
        min_len = max(1, int(T * 0.1))
        max_len = max(2, int(T * 0.9))
        subsequence_length = random.randint(min_len, max_len)

    # Set the compression_factor if not provided
    if compression_factor is None:
        compression_factor = random.randint(2, 5)

    # Set the scale_factor if not provided
    if scale_factor is None:
        scale_factor = torch.rand(D, device=window.device) * 1.9 + 0.1  # (0.1, 2.0)

    # Select random start index for the subsequence
    if start_index is None:
        start_index = random.randint(0, T - subsequence_length)
    end_index = min(start_index + subsequence_length, T)

    if trend_end:
        end_index = T  # Extend trend anomaly to end of the window

    # Extract the subsequence
    anomalous_subsequence = window[start_index:end_index]

    # Tile and downsample the subsequence (compression)
    anomalous_subsequence = anomalous_subsequence.repeat(compression_factor, 1)
    anomalous_subsequence = anomalous_subsequence[::compression_factor]

    # Apply scaling
    anomalous_subsequence *= scale_factor

    # Add trend
    if trend_factor is None:
        trend_factor = torch.normal(mean=1.0, std=0.5, size=(1,), device=window.device).item()  # Fix here
    coef = 1 if torch.rand(1).item() < 0.5 else -1
    anomalous_subsequence += coef * trend_factor

    # Shapelet-based anomaly
    if shapelet_factor:
        anomalous_subsequence = window[start_index] + (torch.rand(len(anomalous_subsequence), D, device=window.device) * 0.1)

    # Inject anomaly back into the window
    window[start_index:end_index] = anomalous_subsequence

    return window.squeeze()  # Ensure it returns correct shape


def carla_anomaly_injection(X):
    """
    Injects anomalies into a batch of time series (both multivariate and univariate).

    :param X: 
        - Multivariate: (batch_size, T, D) torch tensor on CUDA.
        - Univariate: (batch_size, T) torch tensor on CUDA.
    :return: Anomalous batch with same shape as input.
    """

    batch_size = X.shape[0]
    T = X.shape[1]
    is_multivariate = (X.shape[2] > 1)  # True if (batch_size, T, D), False if (batch_size, T)
    D = X.shape[2] if is_multivariate else 1  # Number of features

    anomalous_batch = X.clone()

    # Define possible anomaly types
    anomaly_types = ["seasonal", "trend", "global", "contextual", "shapelet"]

    for b in range(batch_size):
        window = anomalous_batch[b]  # Extract single time series

        # Choose a random anomaly type for this window
        anomaly_type = random.choice(anomaly_types)

        # Define random subsequence length
        min_len = max(1, int(T * 0.1))
        max_len = max(2, int(T * 0.9))
        subsequence_length = random.randint(min_len, max_len)
        start_index = random.randint(0, T - subsequence_length)

        # Select a subset of affected features for multivariate cases
        if is_multivariate:
            num_dims = random.randint(max(1, D // 10), max(2, D // 2))  # Affects ~10-50% of features
            feature_indices = random.sample(range(D), num_dims)
        else:
            feature_indices = [0]  # Single dimension for univariate

        # Apply the selected anomaly type only to chosen features
        for i in feature_indices:
            temp_win = window[:, i].view(T, 1) # if is_multivariate else window.view(T, 1)

            if anomaly_type == "seasonal":
                window[:, i] = inject_frequency_anomaly(temp_win, 
                                                        scale_factor=1, 
                                                        trend_factor=0, 
                                                        subsequence_length=subsequence_length, 
                                                        start_index=start_index)

            elif anomaly_type == "trend":
                window[:, i] = inject_frequency_anomaly(temp_win, 
                                                        compression_factor=1, 
                                                        scale_factor=1, 
                                                        trend_end=True, 
                                                        subsequence_length=subsequence_length, 
                                                        start_index=start_index)

            elif anomaly_type == "global":
                window[:, i] = inject_frequency_anomaly(temp_win, 
                                                        subsequence_length=2, 
                                                        compression_factor=1, 
                                                        scale_factor=8, 
                                                        trend_factor=0, 
                                                        start_index=start_index)

            elif anomaly_type == "contextual":
                window[:, i] = inject_frequency_anomaly(temp_win, 
                                                        subsequence_length=4, 
                                                        compression_factor=1, 
                                                        scale_factor=3, 
                                                        trend_factor=0, 
                                                        start_index=start_index)

            elif anomaly_type == "shapelet":
                window[:, i] = inject_frequency_anomaly(temp_win, 
                                                        compression_factor=1, 
                                                        scale_factor=1, 
                                                        trend_factor=0, 
                                                        shapelet_factor=True, 
                                                        subsequence_length=subsequence_length, 
                                                        start_index=start_index)

        anomalous_batch[b] = window  # Update batch

    return anomalous_batch
