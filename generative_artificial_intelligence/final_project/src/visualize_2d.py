"""
2D two-moons visualization comparing four generative paradigms:
  NF (Normalizing Flow), FM (Flow Matching), DDPM, DDIM

Trains lightweight 2D models from scratch and visualizes:
  - Transport paths from noise to data (fig8)
  - Intermediate distributions at 6 time snapshots (fig9)

Saves to experiments/results/fig8_transport_paths.png
              experiments/results/fig9_twomoons_4way.png

Usage:
  python visualize_2d.py
  python visualize_2d.py --steps 10000 --n_points 2000
"""

import os, sys, argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

OUT_FIG8 = "/home/jovyan/workspace/KU_homework/generative_artificial_intelligence/final_project/experiments/results/fig8_transport_paths.png"
OUT_FIG9 = "/home/jovyan/workspace/KU_homework/generative_artificial_intelligence/final_project/experiments/results/fig9_twomoons_4way.png"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps_nf", type=int, default=8000)
    p.add_argument("--steps_fm", type=int, default=10000)
    p.add_argument("--steps_ddpm", type=int, default=10000)
    p.add_argument("--n_points", type=int, default=1500)
    p.add_argument("--n_paths", type=int, default=300)
    return p.parse_args()


# ── Data ──────────────────────────────────────────────────────────────────────
def get_data(n):
    X, _ = make_moons(n_samples=n, noise=0.07)
    X = (X - X.mean(0)) / X.std(0)
    return torch.FloatTensor(X)


# ── Shared MLP ────────────────────────────────────────────────────────────────
def mlp(in_d, out_d, hidden=128):
    return nn.Sequential(
        nn.Linear(in_d, hidden), nn.SiLU(),
        nn.Linear(hidden, hidden), nn.SiLU(),
        nn.Linear(hidden, hidden), nn.SiLU(),
        nn.Linear(hidden, out_d),
    )


# ── Normalizing Flow (RealNVP-style, 2D) ─────────────────────────────────────
class CouplingLayer(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask
        self.net = mlp(2, 2, 64)

    def forward(self, x, reverse=False):
        x0 = x * self.mask
        st = self.net(x0)
        s, t = st[:, :1], st[:, 1:]
        s = torch.tanh(s) * 2
        if not reverse:
            y = x * self.mask + (1 - self.mask) * (x * torch.exp(s) + t)
            return y, s.sum(1)
        else:
            y = x * self.mask + (1 - self.mask) * ((x - t) * torch.exp(-s))
            return y


class NF2D(nn.Module):
    def __init__(self, n_flows=8):
        super().__init__()
        masks = [torch.FloatTensor([1, 0]), torch.FloatTensor([0, 1])]
        self.flows = nn.ModuleList([CouplingLayer(masks[i % 2]) for i in range(n_flows)])

    def forward(self, x):
        log_det = 0
        for f in self.flows:
            x, ld = f(x)
            log_det = log_det + ld
        log_pz = -0.5 * (x ** 2).sum(1) - np.log(2 * np.pi)
        return x, log_det, log_pz

    @torch.no_grad()
    def sample(self, n):
        z = torch.randn(n, 2)
        for f in reversed(self.flows):
            z = f(z, reverse=True)
        return z

    @torch.no_grad()
    def transport_path(self, z, n_steps=20):
        """Track a point z through all flow layers."""
        path = [z.clone()]
        for f in reversed(self.flows):
            z = f(z, reverse=True)
            path.append(z.clone())
        return path


def train_nf(data, steps, device):
    model = NF2D(n_flows=10).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(data), (256,))
        x = data[idx].to(device)
        _, log_det, log_pz = model(x)
        loss = -(log_pz + log_det).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 2000 == 0:
            print(f"  step {step}: loss={loss.item():.4f}")
    return model


# ── Flow Matching (simple vector field) ───────────────────────────────────────
class VectorField(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(3, 2, 128)  # (x, t) -> v

    def forward(self, x, t):
        t_emb = t.view(-1, 1).expand(-1, 1)
        return self.net(torch.cat([x, t_emb], dim=1))


def train_fm(data, steps, device):
    model = VectorField().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(data), (256,))
        x1 = data[idx].to(device)
        x0 = torch.randn_like(x1)
        t  = torch.rand(len(x1), device=device)
        xt = (1 - t.view(-1, 1)) * x0 + t.view(-1, 1) * x1
        v_target = x1 - x0
        v_pred = model(xt, t)
        loss = ((v_pred - v_target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 2000 == 0:
            print(f"  step {step}: loss={loss.item():.4f}")
    return model


@torch.no_grad()
def sample_fm(model, n, n_steps=50, device="cpu"):
    x = torch.randn(n, 2, device=device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((n,), i * dt, device=device)
        x = x + model(x, t) * dt
    return x


@torch.no_grad()
def fm_path(model, z, n_steps=20, device="cpu"):
    path = [z.clone()]
    x = z.clone()
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((len(x),), i * dt, device=device)
        x = x + model(x, t) * dt
        path.append(x.clone())
    return path


# ── DDPM (simple score network) ───────────────────────────────────────────────
class ScoreNet(nn.Module):
    def __init__(self, T=1000):
        super().__init__()
        self.T = T
        self.net = mlp(3, 2, 128)
        betas = torch.linspace(1e-4, 0.02, T)
        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, 0)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bar", alpha_bar)

    def forward(self, x_t, t):
        t_norm = (t.float() / self.T).view(-1, 1)
        return self.net(torch.cat([x_t, t_norm], dim=1))

    def q_sample(self, x0, t):
        ab = self.alpha_bar[t].view(-1, 1)
        noise = torch.randn_like(x0)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise, noise

    @torch.no_grad()
    def ddpm_sample(self, n, device):
        x = torch.randn(n, 2, device=device)
        path = [x.clone()]
        for t in reversed(range(self.T)):
            t_batch = torch.full((n,), t, device=device, dtype=torch.long)
            eps = self(x, t_batch)
            ab = self.alpha_bar[t]
            ab_prev = self.alpha_bar[t - 1] if t > 0 else torch.tensor(1.0)
            beta = self.betas[t]
            x0_pred = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
            mean = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps
            if t > 0:
                x = mean + beta.sqrt() * torch.randn_like(x)
            else:
                x = mean
            if t % (self.T // 6) == 0 or t == 0:
                path.append(x.clone())
        return x, path

    @torch.no_grad()
    def ddim_sample(self, n, n_steps, device):
        x = torch.randn(n, 2, device=device)
        step_size = self.T // n_steps
        timesteps = list(range(self.T - 1, -1, -step_size))
        path = [x.clone()]
        for i, t in enumerate(timesteps):
            t_batch = torch.full((n,), t, device=device, dtype=torch.long)
            eps = self(x, t_batch)
            ab = self.alpha_bar[t]
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            ab_prev = self.alpha_bar[t_prev] if t_prev > 0 else torch.tensor(1.0)
            x0_pred = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
            x = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps
            if i % (len(timesteps) // 5) == 0 or i == len(timesteps) - 1:
                path.append(x.clone())
        return x, path


def train_ddpm(data, steps, device):
    model = ScoreNet(T=300).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(data), (256,))
        x0 = data[idx].to(device)
        t  = torch.randint(0, model.T, (len(x0),), device=device)
        x_t, noise = model.q_sample(x0, t)
        loss = ((model(x_t, t) - noise) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 2000 == 0:
            print(f"  step {step}: loss={loss.item():.4f}")
    return model


# ── Plot helpers ──────────────────────────────────────────────────────────────
COLORS = {"NF": "#2EA05B", "FM": "#2E5FA3", "DDPM": "#E84A2F", "DDIM": "#F5A623"}
BG = "#0D1117"


def plot_transport_paths(paths_dict, data, out_path):
    """paths_dict: {name: list of (n_paths, 2) tensors along trajectory}"""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor=BG)
    data_np = data.numpy()

    for ax, (name, path_list) in zip(axes, paths_dict.items()):
        ax.set_facecolor(BG)
        # background: data distribution
        ax.scatter(data_np[:, 0], data_np[:, 1], c="#FFFFFF", s=2, alpha=0.15, zorder=1)
        # transport paths
        col = COLORS[name]
        for pi in range(len(path_list[0])):
            traj_x = [step[pi, 0].item() for step in path_list]
            traj_y = [step[pi, 1].item() for step in path_list]
            ax.plot(traj_x, traj_y, c=col, alpha=0.35, lw=0.8, zorder=2)
        # start (noise) and end (sample) points
        ax.scatter(path_list[0][:, 0], path_list[0][:, 1],
                   c="#FFFFFF", s=8, alpha=0.6, zorder=3)
        ax.scatter(path_list[-1][:, 0], path_list[-1][:, 1],
                   c=col, s=12, alpha=0.9, zorder=4)
        ax.set_title(name, color=col, fontsize=16, fontweight="bold", pad=8)
        ax.set_xlim(-3, 3); ax.set_ylim(-2.5, 2.5)
        ax.tick_params(colors="gray"); ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    fig.suptitle("Transport Path Comparison on 2D Two-Moons",
                 color="white", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


def plot_intermediate(snapshots_dict, data, out_path, n_snapshots=6):
    """snapshots_dict: {name: list of (N,2) tensors at snapshots}"""
    n_models = len(snapshots_dict)
    fig, axes = plt.subplots(n_models, n_snapshots,
                             figsize=(n_snapshots * 3, n_models * 3),
                             facecolor=BG)
    data_np = data.numpy()

    for ri, (name, snaps) in enumerate(snapshots_dict.items()):
        col = COLORS[name]
        snaps = snaps[:n_snapshots]
        for ci, snap in enumerate(snaps):
            ax = axes[ri][ci]
            ax.set_facecolor(BG)
            ax.scatter(data_np[:, 0], data_np[:, 1], c="#FFFFFF", s=3, alpha=0.1)
            s = snap.numpy() if isinstance(snap, torch.Tensor) else snap
            # color by position index to track identity
            cmap_val = np.linspace(0, 1, len(s))
            ax.scatter(s[:, 0], s[:, 1], c=cmap_val, cmap="plasma",
                       s=10, alpha=0.7, vmin=0, vmax=1)
            ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3, 3); ax.set_aspect("equal")
            ax.axis("off")
            frac = ci / (n_snapshots - 1)
            if ri == 0:
                ax.set_title(f"t={frac:.1f}", color="white", fontsize=11)
        axes[ri][0].set_ylabel(name, color=col, fontsize=14, fontweight="bold",
                               rotation=0, labelpad=50, va="center")

    fig.suptitle("Intermediate Distributions (6 Snapshots per Model)\nColor = point identity tracked from noise to data",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


def main():
    args = parse_args()
    device = torch.device("cpu")  # 2D models: CPU is fine
    data = get_data(args.n_points)

    print("Training NF...")
    nf = train_nf(data, args.steps_nf, device)
    print("Training FM...")
    fm = train_fm(data, args.steps_fm, device)
    print("Training DDPM...")
    ddpm = train_ddpm(data, args.steps_ddpm, device)

    n_paths = args.n_paths
    noise = torch.randn(n_paths, 2)

    # ── Transport paths ───────────────────────────────────────────────────────
    print("Computing transport paths...")
    # NF: reverse flows
    nf_path = nf.transport_path(noise.clone())

    # FM: ODE integration
    fm_path_list = fm_path(fm, noise.clone(), n_steps=20, device=device)

    # DDPM: stochastic reverse (subsample timesteps for path)
    with torch.no_grad():
        x = noise.clone()
        ddpm_path_list = [x.clone()]
        step_size = ddpm.T // 20
        for t in reversed(range(0, ddpm.T, step_size)):
            t_b = torch.full((n_paths,), t, dtype=torch.long)
            eps = ddpm(x, t_b)
            ab = ddpm.alpha_bar[t]
            ab_prev = ddpm.alpha_bar[max(t - step_size, 0)]
            x0_pred = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
            noise_term = (1 - ab_prev).sqrt() * eps
            x = ab_prev.sqrt() * x0_pred + noise_term
            if t > 0:
                x = x + ddpm.betas[t].sqrt() * torch.randn_like(x) * 0.3
            ddpm_path_list.append(x.clone())

    # DDIM: deterministic reverse
    with torch.no_grad():
        x = noise.clone()
        ddim_path_list = [x.clone()]
        n_steps = 20
        step_size = ddpm.T // n_steps
        timesteps = list(range(ddpm.T - 1, -1, -step_size))
        for i, t in enumerate(timesteps):
            t_b = torch.full((n_paths,), t, dtype=torch.long)
            eps = ddpm(x, t_b)
            ab = ddpm.alpha_bar[t]
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            ab_prev = ddpm.alpha_bar[t_prev] if t_prev > 0 else torch.tensor(1.0)
            x0_pred = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
            x = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps
            ddim_path_list.append(x.clone())

    paths_dict = {
        "NF":   nf_path,
        "FM":   fm_path_list,
        "DDPM": ddpm_path_list,
        "DDIM": ddim_path_list,
    }
    os.makedirs(os.path.dirname(OUT_FIG8), exist_ok=True)
    plot_transport_paths(paths_dict, data, OUT_FIG8)

    # ── Intermediate distributions ────────────────────────────────────────────
    print("Computing intermediate distributions...")
    n_large = 1000
    noise_large = torch.randn(n_large, 2)

    # NF snapshots
    nf_snaps = nf.transport_path(noise_large.clone())
    nf_snaps = [nf_snaps[i] for i in np.linspace(0, len(nf_snaps)-1, 6, dtype=int)]

    # FM snapshots
    fm_snaps = fm_path(fm, noise_large.clone(), n_steps=50, device=device)
    fm_snaps = [fm_snaps[i] for i in np.linspace(0, len(fm_snaps)-1, 6, dtype=int)]

    # DDPM snapshots
    _, ddpm_snaps = ddpm.ddpm_sample(n_large, device)
    ddpm_snaps = list(reversed(ddpm_snaps))[:6]

    # DDIM snapshots
    _, ddim_snaps = ddpm.ddim_sample(n_large, n_steps=50, device=device)
    ddim_snaps = list(reversed(ddim_snaps))[:6]

    snaps_dict = {
        "NF":   nf_snaps,
        "FM":   fm_snaps,
        "DDPM": ddpm_snaps,
        "DDIM": ddim_snaps,
    }
    plot_intermediate(snaps_dict, data, OUT_FIG9)
    print("Done.")


if __name__ == "__main__":
    main()
