import argparse
import numpy as np
import pandas as pd
import torch
from train_utils import *
from losses import *

def vae_loss(x, x_recon, mu, logvar, latent_vectors, z_pert, x_anom):
    recon_loss = reconstruction_loss(x, x_recon)
    perturb_loss = perturbation_loss_input_triplet(x, x_recon, x_anom)
    perturb_zero_loss = perturbation_zero_reg_loss(x, x_anom)
    sigma_prior = 0.5
    kl_loss = kl_en_loss(logvar, mu, sigma_prior)
    
    # Loss weights
    alpha = 10.0  # Weight for Recon loss
    beta = 1.0    # Weight for Perturb loss
    gamma = 0.1   # Weight for Perturb Zero loss
    zetta = 1.0   # Weight for KL_en loss    
    
    total_loss = (
        alpha * recon_loss + beta * perturb_loss + gamma * perturb_zero_loss + zetta * kl_loss
    )
    
    return {
        "total_loss": total_loss,
        "recon_loss": recon_loss,
        "perturb_loss": perturb_loss,
        "perturb_zero_loss": perturb_zero_loss,
        "kl_loss": kl_loss,
    }

def main():
    parser = argparse.ArgumentParser(description="VAE Training Script")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--window_size", type=int, default=200, help="Window size for time series segmentation")
    parser.add_argument("--step_size", type=int, default=1, help="Step size for windowing")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for training")
    parser.add_argument("--hid_size", type=int, default=200, help="Hidden layer size")
    parser.add_argument("--latent_dim", type=int, default=100, help="Latent dimension size")
    parser.add_argument("--dataset", type=str, default="msl", choices=["msl", "other"], help="Dataset selection")
    parser.add_argument("--savedir", type=str, default="msl_kl12", help="Directory to save models")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.dataset == "msl":
        with open("../../datasets/MSL_SMAP/labeled_anomalies.csv", 'r') as file:
            csv_reader = pd.read_csv(file, delimiter=',')
        data_info = csv_reader[csv_reader['spacecraft'] == 'MSL']
        file_list = np.array(data_info['chan_id'])
        
        for filename in file_list:
            time_series = np.load(f"../../datasets/MSL_SMAP/train/{filename}.npy")
            windows = create_windows_min_max(time_series, args.window_size, args.step_size)
            print(windows.shape)
            saveBestGenModel(filename, args.epochs, args.batch_size, args.learning_rate, windows, args.hid_size, args.latent_dim, vae_loss, args.savedir)
    else:
        print("- other datasets -")

if __name__ == "__main__":
    main()
