import os
from os.path import dirname, exists, join
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_util import device

def generate_pinwheel_data(n_samples=5000, seed=0):
    """
    Generates a 2D pinwheel dataset with 5 spiral arms (whirlpool shape).
    """
    rng = np.random.RandomState(seed)
    radial_std = 0.3
    tangential_std = 0.03
    num_classes = 5
    num_per_class = n_samples // num_classes
    rate = 0.85
    
    rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)
    
    features = rng.randn(n_samples, 2) * np.array([radial_std, tangential_std])
    features[:, 0] += 1.0
    
    labels = np.repeat(np.arange(num_classes), num_per_class)
    if len(labels) < n_samples:
        labels = np.concatenate([labels, rng.choice(num_classes, n_samples - len(labels))])
        
    angles = rads[labels] + features[:, 0] * rate
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))
    
    feats = np.einsum('ti,tij->tj', features, rotations)
    feats = feats * 2.5  # Scale to fit [-4, 4]
    return feats.astype(np.float32)

def generate_gmm_3d_data(n_samples=5000, seed=0):
    """
    Generates a 3D GMM dataset with 4 components arranged in a tetrahedron.
    """
    rng = np.random.RandomState(seed)
    means = np.array([
        [-2.0, -2.0, -2.0],
        [2.0, 2.0, -2.0],
        [-2.0, 2.0, 2.0],
        [2.0, -2.0, 2.0]
    ])
    stds = np.array([0.5, 0.5, 0.5, 0.5])
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    
    components = rng.choice(len(weights), size=n_samples, p=weights)
    samples = []
    for comp in components:
        mean = means[comp]
        std = stds[comp]
        sample = mean + std * rng.randn(3)
        samples.append(sample)
    return np.array(samples, dtype=np.float32)

def savefig(fname: str, show_figure: bool = True) -> None:
    if fname is not None:
        dir_name = dirname(fname)
        if dir_name and not exists(dir_name):
            os.makedirs(dir_name)
        plt.tight_layout()
        plt.savefig(fname, dpi=100)
    if show_figure:
        plt.show()
    plt.close()

def save_training_plot(train_losses, val_losses, title, fname):
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title(title)
    plt.xlabel("Iteration / Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    savefig(fname)

# --- 2D Plotting Helpers ---

def plot_energy_landscape(ebm_model, fname=None, title="Learned Energy Landscape"):
    ebm_model.eval()
    x = np.linspace(-4.0, 4.0, 100)
    y = np.linspace(-4.0, 4.0, 100)
    X, Y = np.meshgrid(x, y)
    grid = np.stack([X.ravel(), Y.ravel()], axis=1)
    
    with torch.no_grad():
        grid_tensor = torch.from_numpy(grid).float().to(device)
        energies = ebm_model(grid_tensor)
        energies = energies.cpu().numpy().reshape(X.shape)
        
    plt.figure(figsize=(6, 5))
    plt.contourf(X, Y, energies, levels=50, cmap="viridis")
    plt.colorbar(label="Energy E(x)")
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    savefig(fname)

def plot_score_field(score_model, true_data=None, fname=None, title="Learned Score Field"):
    score_model.eval()
    x = np.linspace(-4.0, 4.0, 20)
    y = np.linspace(-4.0, 4.0, 20)
    X, Y = np.meshgrid(x, y)
    grid = np.stack([X.ravel(), Y.ravel()], axis=1)
    
    with torch.no_grad():
        grid_tensor = torch.from_numpy(grid).float().to(device)
        scores = score_model(grid_tensor)
        scores = scores.cpu().numpy()
    
    U = scores[:, 0].reshape(X.shape)
    V = scores[:, 1].reshape(X.shape)
    
    norms = np.sqrt(U**2 + V**2) + 1e-8
    U_norm = U / norms
    V_norm = V / norms
    
    plt.figure(figsize=(6, 5))
    if true_data is not None:
        plt.scatter(true_data[:, 0], true_data[:, 1], s=2, color="blue", alpha=0.15, label="True Data")
    plt.quiver(X, Y, U_norm, V_norm, color="red", width=0.003, scale=30)
    plt.xlim(-4.0, 4.0)
    plt.ylim(-4.0, 4.0)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(loc="upper right")
    savefig(fname)

def plot_scatter_samples(samples, true_data=None, fname=None, title="Generated MCMC Samples"):
    plt.figure(figsize=(6, 5))
    if true_data is not None:
        plt.scatter(true_data[:, 0], true_data[:, 1], s=2, color="blue", alpha=0.15, label="True Data")
    plt.scatter(samples[:, 0], samples[:, 1], s=5, color="green", alpha=0.6, label="MCMC Samples")
    plt.xlim(-4.0, 4.0)
    plt.ylim(-4.0, 4.0)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(loc="upper right")
    savefig(fname)

# --- 3D Plotting Helpers ---

def plot_energy_landscape_3d(ebm_model, fname=None, title="Learned Energy Landscape (3D)"):
    ebm_model.eval()
    x = np.linspace(-4.0, 4.0, 20)
    y = np.linspace(-4.0, 4.0, 20)
    z = np.linspace(-4.0, 4.0, 20)
    X, Y, Z = np.meshgrid(x, y, z)
    grid = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    
    with torch.no_grad():
        grid_tensor = torch.from_numpy(grid).float().to(device)
        energies = ebm_model(grid_tensor)
        energies = energies.cpu().numpy().squeeze()
        
    threshold = np.percentile(energies, 15)
    mask = energies < threshold
    
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection='3d')
    sc = ax.scatter(grid[mask, 0], grid[mask, 1], grid[mask, 2], c=energies[mask], cmap="viridis", s=15, alpha=0.6)
    fig.colorbar(sc, ax=ax, label="Energy E(x)")
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(-4.0, 4.0)
    ax.set_zlim(-4.0, 4.0)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    savefig(fname)

def plot_score_field_3d(score_model, true_data=None, fname=None, title="Learned Score Field (3D)"):
    score_model.eval()
    x = np.linspace(-3.5, 3.5, 8)
    y = np.linspace(-3.5, 3.5, 8)
    z = np.linspace(-3.5, 3.5, 8)
    X, Y, Z = np.meshgrid(x, y, z)
    grid = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    
    with torch.no_grad():
        grid_tensor = torch.from_numpy(grid).float().to(device)
        scores = score_model(grid_tensor)
        scores = scores.cpu().numpy()
        
    U = scores[:, 0]
    V = scores[:, 1]
    W = scores[:, 2]
    
    norms = np.sqrt(U**2 + V**2 + W**2) + 1e-8
    U_norm = U / norms
    V_norm = V / norms
    W_norm = W / norms
    
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection='3d')
    if true_data is not None:
        ax.scatter(true_data[:, 0], true_data[:, 1], true_data[:, 2], s=2, color="blue", alpha=0.08, label="True Data")
    ax.quiver(X.ravel(), Y.ravel(), Z.ravel(), U_norm, V_norm, W_norm, color="red", length=0.6, pivot="middle")
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(-4.0, 4.0)
    ax.set_zlim(-4.0, 4.0)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.legend(loc="upper right")
    savefig(fname)

def plot_scatter_samples_3d(samples, true_data=None, fname=None, title="Generated MCMC Samples (3D)"):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection='3d')
    if true_data is not None:
        ax.scatter(true_data[:, 0], true_data[:, 1], true_data[:, 2], s=2, color="blue", alpha=0.08, label="True Data")
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=5, color="green", alpha=0.5, label="MCMC Samples")
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(-4.0, 4.0)
    ax.set_zlim(-4.0, 4.0)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.legend(loc="upper right")
    savefig(fname)
