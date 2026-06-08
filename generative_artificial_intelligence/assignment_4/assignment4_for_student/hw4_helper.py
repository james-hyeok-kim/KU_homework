import numpy as np
import matplotlib.pyplot as plt
from utils import (
    generate_pinwheel_data,
    generate_gmm_3d_data,
    save_training_plot,
    plot_energy_landscape,
    plot_score_field,
    plot_scatter_samples,
    plot_energy_landscape_3d,
    plot_score_field_3d,
    plot_scatter_samples_3d
)

def load_data(dset_type, n_train=4000, n_val=1000, seed=42):
    """
    Loads training and validation datasets.
    dset_type = 1: 2D Pinwheel (Whirlpool)
    dset_type = 2: 3D GMM
    """
    if dset_type == 1:
        train_data = generate_pinwheel_data(n_train, seed=seed)
        val_data = generate_pinwheel_data(n_val, seed=seed + 1)
    elif dset_type == 2:
        train_data = generate_gmm_3d_data(n_train, seed=seed)
        val_data = generate_gmm_3d_data(n_val, seed=seed + 1)
    else:
        raise ValueError(f"Invalid dset_type: {dset_type}")
    return train_data, val_data

def visualize_data(dset_type, data):
    if dset_type == 1:
        plt.figure(figsize=(6, 5))
        plt.scatter(data[:, 0], data[:, 1], s=5, color="blue", alpha=0.5)
        plt.xlim(-4.0, 4.0)
        plt.ylim(-4.0, 4.0)
        plt.title("Pinwheel Data Scatter Plot (Dataset 1)")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()
    elif dset_type == 2:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=5, color="blue", alpha=0.5)
        ax.set_xlim(-4.0, 4.0)
        ax.set_ylim(-4.0, 4.0)
        ax.set_zlim(-4.0, 4.0)
        ax.set_title("3D GMM Data Scatter Plot (Dataset 2)")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
        plt.show()

def q1_save_results(dset_type, train_data, val_data, train_ebm_fn, use_langevin=True):
    name = "Pinwheel (2D)" if dset_type == 1 else "GMM (3D)"
    mcmc_name = "Langevin" if use_langevin else "MH"
    print(f"Training EBM with {mcmc_name} CD on Dataset {dset_type} ({name})...")
    model, train_losses, val_losses, samples = train_ebm_fn(train_data, val_data, dset_type, use_langevin)
    
    print(f"Final Train Loss: {train_losses[-1]:.4f}")
    print(f"Final Val Loss: {val_losses[-1]:.4f}")
    
    save_training_plot(
        train_losses,
        val_losses,
        f"Q1: EBM CD-k ({mcmc_name}) Training Curve (Dataset {dset_type})",
        f"results/q1_dset{dset_type}_{mcmc_name.lower()}_train_plot.png"
    )
    
    if dset_type == 1:
        plot_energy_landscape(
            model,
            f"results/q1_dset{dset_type}_{mcmc_name.lower()}_energy_landscape.png",
            f"Q1: EBM ({mcmc_name}) Learned Energy Landscape (Dataset {dset_type})"
        )
        plot_scatter_samples(
            samples,
            true_data=val_data,
            fname=f"results/q1_dset{dset_type}_{mcmc_name.lower()}_samples.png",
            title=f"Q1: EBM {mcmc_name} MCMC Samples (Dataset {dset_type})"
        )
    else:
        plot_energy_landscape_3d(
            model,
            f"results/q1_dset{dset_type}_{mcmc_name.lower()}_energy_landscape.png",
            f"Q1: EBM ({mcmc_name}) Learned Energy Landscape (Dataset {dset_type})"
        )
        plot_scatter_samples_3d(
            samples,
            true_data=val_data,
            fname=f"results/q1_dset{dset_type}_{mcmc_name.lower()}_samples.png",
            title=f"Q1: EBM {mcmc_name} MCMC Samples (Dataset {dset_type})"
        )
    
    print(f"Q1 ({mcmc_name}, Dataset {dset_type}) results saved successfully!")
    return model

def q2_a_save_results(dset_type, train_data, val_data, train_esm_fn):
    name = "Pinwheel (2D)" if dset_type == 1 else "GMM (3D)"
    print(f"Training Score Model with Explicit Score Matching on Dataset {dset_type} ({name})...")
    model, train_losses, val_losses, langevin_samples = train_esm_fn(train_data, val_data, dset_type)
    
    print(f"Final Train Loss: {train_losses[-1]:.4f}")
    print(f"Final Val Loss: {val_losses[-1]:.4f}")
    
    save_training_plot(
        train_losses,
        val_losses,
        f"Q2(a): ESM Training Curve (Dataset {dset_type})",
        f"results/q2_a_dset{dset_type}_train_plot.png"
    )
    
    if dset_type == 1:
        plot_score_field(
            model,
            true_data=val_data,
            fname=f"results/q2_a_dset{dset_type}_score_field.png",
            title=f"Q2(a): ESM Learned Score Field (Dataset {dset_type})"
        )
        plot_scatter_samples(
            langevin_samples,
            true_data=val_data,
            fname=f"results/q2_a_dset{dset_type}_samples.png",
            title=f"Q2(a): ESM Langevin MCMC Samples (Dataset {dset_type})"
        )
    else:
        plot_score_field_3d(
            model,
            true_data=val_data,
            fname=f"results/q2_a_dset{dset_type}_score_field.png",
            title=f"Q2(a): ESM Learned Score Field (Dataset {dset_type})"
        )
        plot_scatter_samples_3d(
            langevin_samples,
            true_data=val_data,
            fname=f"results/q2_a_dset{dset_type}_samples.png",
            title=f"Q2(a): ESM Langevin MCMC Samples (Dataset {dset_type})"
        )
        
    print(f"Q2(a) (Dataset {dset_type}) results saved successfully!")
    return model

def q2_b_save_results(dset_type, train_data, val_data, train_dsm_fn, sigma):
    name = "Pinwheel (2D)" if dset_type == 1 else "GMM (3D)"
    print(f"Training Score Model with Denoising Score Matching (sigma={sigma}) on Dataset {dset_type} ({name})...")
    model, train_losses, val_losses, langevin_samples = train_dsm_fn(train_data, val_data, sigma, dset_type)
    
    print(f"Final Train Loss: {train_losses[-1]:.4f}")
    print(f"Final Val Loss: {val_losses[-1]:.4f}")
    
    save_training_plot(
        train_losses,
        val_losses,
        f"Q2(b) sigma={sigma} DSM Training Curve (Dataset {dset_type})",
        f"results/q2_b_dset{dset_type}_sigma_{sigma}_train_plot.png"
    )
    
    if dset_type == 1:
        plot_score_field(
            model,
            true_data=val_data,
            fname=f"results/q2_b_dset{dset_type}_sigma_{sigma}_score_field.png",
            title=f"Q2(b): DSM Score Field (sigma={sigma}, Dataset {dset_type})"
        )
        plot_scatter_samples(
            langevin_samples,
            true_data=val_data,
            fname=f"results/q2_b_dset{dset_type}_sigma_{sigma}_samples.png",
            title=f"Q2(b): DSM Langevin Samples (sigma={sigma}, Dataset {dset_type})"
        )
    else:
        plot_score_field_3d(
            model,
            true_data=val_data,
            fname=f"results/q2_b_dset{dset_type}_sigma_{sigma}_score_field.png",
            title=f"Q2(b): DSM Score Field (sigma={sigma}, Dataset {dset_type})"
        )
        plot_scatter_samples_3d(
            langevin_samples,
            true_data=val_data,
            fname=f"results/q2_b_dset{dset_type}_sigma_{sigma}_samples.png",
            title=f"Q2(b): DSM Langevin Samples (sigma={sigma}, Dataset {dset_type})"
        )
        
    print(f"Q2(b) sigma={sigma} (Dataset {dset_type}) results saved successfully!")
    return model
