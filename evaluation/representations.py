import sys
import os

current_dir = os.getcwd()
parent_of_parent = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_of_parent)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models import TCVAE
from train_utils import *
from losses import *
from ts_loader import load_data
from deepsvdd_adv import *
from injection_models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_representation(injector='genias', injector_params=None, dataset='MSL', entity='M-6', window_size=200, step_size=1, with_training=False, save=False):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data, test_data, test_labels = load_data(dataset, entity)
    train_windows, dmin, dmax = create_windows_min_max_2(train_data, window_size, step_size)
    test_windows, _, _ = create_windows_min_max_2(test_data, window_size, step_size)
    test_labels, _, _ = create_windows_min_max_2(test_labels, window_size, step_size, 0, 0, is_label=True)

    train_data = torch.tensor(train_windows).to(device).float()
    test_data = torch.tensor(test_windows).to(device).float()

    save_dir = f'generated_anoms/{dataset}/{entity}'
    model_filename = 'deepsvdd_model.pth'
    params_filename = 'deepsvdd_params.pth'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model_path = os.path.join(save_dir, model_filename)
    params_path = os.path.join(save_dir, params_filename)


    if( (not os.path.exists(model_path)) or with_training):
        num_windows, window_length, ts_dims = train_data.shape

        hidden_dim = 128
        latent_dim = 8

        batch_size = 64
        train_loader = prepare_data(train_data, batch_size)

        model = DeepSVDD(ts_dims=ts_dims, hidden_dim=hidden_dim, latent_dim=latent_dim, window_length=window_length).to(device)
        model.calculate_center(train_loader)  # Calculate the initial center
        train_deepsvdd(model, train_loader, num_epochs=100, lr=1e-3)

        train_representations, train_scores = anomaly_score(model, train_data.to(device))
        test_representations, test_scores = anomaly_score(model, test_data.to(device))

        data_to_save = {
            "train_representations": train_representations,
            "train_scores": train_scores,
            "test_representations": test_representations,
            "test_scores": test_scores,
            "test_labels": test_labels
        }
        train_test_rep_path = os.path.join(save_dir, 'train_test_rep.pth')
        torch.save(data_to_save, train_test_rep_path)

        torch.save(model.state_dict(), model_path)
        torch.save({'hidden_dim': hidden_dim, 'latent_dim': latent_dim, 'window_length': window_length, 'ts_dims': ts_dims}, params_path)
    else:
        # if not os.path.exists(model_path):
        #     raise FileNotFoundError(f"\nNo pretrained model found for {dataset}-{entity}\n+++ Hint: Try the model with with_training=True +++")
        params = torch.load(params_path)
        hidden_dim = params['hidden_dim']
        latent_dim = params['latent_dim']
        window_length = params['window_length']
        ts_dims = params['ts_dims']

        model = DeepSVDD(ts_dims=ts_dims, hidden_dim=hidden_dim, latent_dim=latent_dim, window_length=window_length).to(device)
        model.load_state_dict(torch.load(model_path))

    if(injector == 'genias'):
        scale = injector_params['scale']
        tau = injector_params['tau']
        gen_model = injector_params['gen_model']
        injected_windows, _, _, _ = genias_injection(train_data, entity, savedir=gen_model, scale=scale, tau=tau)
    elif(injector == 'couta'):
        injected_windows, _ = COUTA_create_neg(batch_seqs=train_data, max_cut_ratio=0.5, seed=0, return_mul_label=False, ss_type='FULL')
    elif(injector == 'ncad'):
        coe_rate = injector_params['coe_rate']
        mixup_rate = injector_params['mixup_rate']
        suspect_win_len = injector_params['suspect_win_len']
        injected_windows, _ = NCAD_inject(batch_x=train_data, coe_rate=coe_rate, mixup_rate=mixup_rate, suspect_win_len=suspect_win_len)
    elif(injector == 'cutaddpaste'):
        injected_windows = cut_add_paste_outlier(train_x=train_data, configs=injector_params)
    elif(injector == 'carla'):
        injected_windows = carla_anomaly_injection(X=train_data)

    if not torch.is_tensor(injected_windows):
        injected_windows = torch.tensor(injected_windows)
    injected_windows = injected_windows.to(device).float()

    injected_representations, injected_scores = anomaly_score(model, injected_windows.to(device))
    if save:
        data_to_save = {
            "injected_representations": injected_representations,
            "injected_scores": injected_scores,
        }
        injected_rep_path = os.path.join(save_dir, f'{injector}_rep.pth')
        torch.save(data_to_save, injected_rep_path)

    return injected_representations, injected_scores


    
