"""
GLOW: Generative Flow with Invertible 1x1 Convolutions
Kingma & Dhariwal, NeurIPS 2018

Architecture reconstructed from checkpoint (glow_ffhq64_005000.pt):
  4 blocks x 32 flows, hidden=512
  Multi-scale with split+prior+squeeze between blocks:
    Block 0: 32x32x12  -> 32 flows -> split(6+6)  -> prior -> squeeze -> 16x16x24
    Block 1: 16x16x24  -> 32 flows -> split(12+12) -> prior -> squeeze -> 8x8x48
    Block 2: 8x8x48    -> 32 flows -> split(24+24) -> prior -> squeeze -> 4x4x96
    Block 3: 4x4x96    -> 32 flows -> full prior (no split)
  Total latent: 6*32*32 + 12*16*16 + 24*8*8 + 96*4*4 = 12288 = 64*64*3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Primitive layers
# ---------------------------------------------------------------------------

class ActNorm(nn.Module):
    """Data-dependent actnorm. Keys: .loc, .scale, .initialized"""

    def __init__(self, in_channels):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, x):
        with torch.no_grad():
            flat = x.permute(1, 0, 2, 3).contiguous().view(x.shape[1], -1)
            mean = flat.mean(1).view(1, -1, 1, 1)
            std = flat.std(1).view(1, -1, 1, 1)
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))
            self.initialized.fill_(1)

    def forward(self, x, reverse=False):
        if not self.initialized:
            self.initialize(x)
        _, _, h, w = x.shape
        log_det = h * w * torch.sum(torch.log(torch.abs(self.scale)))
        if reverse:
            return (x - self.loc) / self.scale, -log_det
        return self.scale * (x + self.loc), log_det


class InvConv2d(nn.Module):
    """Invertible 1x1 conv with LU decomposition. Keys: w_l, w_s, w_u, w_p, u_mask, l_mask, s_sign, l_eye"""

    def __init__(self, in_channels):
        super().__init__()
        q, _ = torch.linalg.qr(torch.randn(in_channels, in_channels))
        p, l, u = torch.linalg.lu(q)
        s = torch.diag(u)
        u_triu = torch.triu(u, 1)
        s_sign = torch.sign(s)
        log_s = torch.log(torch.abs(s))

        self.register_buffer("w_p", p)
        self.register_buffer("u_mask", torch.triu(torch.ones(in_channels, in_channels), 1))
        self.register_buffer("l_mask", torch.tril(torch.ones(in_channels, in_channels), -1))
        self.register_buffer("s_sign", s_sign)
        self.register_buffer("l_eye", torch.eye(in_channels))
        self.w_l = nn.Parameter(l)
        self.w_s = nn.Parameter(log_s)
        self.w_u = nn.Parameter(u_triu)

    def calc_weight(self):
        # clamp w_s to prevent exp → 0 (singular matrix)
        w_s = self.w_s.clamp(-10, 10)
        L = self.w_l * self.l_mask + self.l_eye
        U = self.w_u * self.u_mask + torch.diag(self.s_sign * torch.exp(w_s))
        return (self.w_p @ L @ U).unsqueeze(2).unsqueeze(3)

    def forward(self, x, reverse=False):
        _, _, h, w = x.shape
        w_s = self.w_s.clamp(-10, 10)
        log_det = h * w * torch.sum(w_s)
        if reverse:
            # Invert via LU structure (numerically stable, avoids singular-matrix error)
            # W = P @ L @ U  →  W_inv = U_inv @ L_inv @ P^T
            dtype = torch.float64
            dev = x.device
            I = torch.eye(w_s.shape[0], dtype=dtype, device=dev)
            L = (self.w_l * self.l_mask + self.l_eye).to(dtype)
            U = (self.w_u * self.u_mask +
                 torch.diag(self.s_sign * torch.exp(w_s))).to(dtype)
            L_inv = torch.linalg.solve_triangular(L, I, upper=False)
            U_inv = torch.linalg.solve_triangular(U, I, upper=True)
            W_inv = (U_inv @ L_inv @ self.w_p.to(dtype).T).to(x.dtype)
            return F.conv2d(x, W_inv.unsqueeze(2).unsqueeze(3)), -log_det
        W = self.calc_weight()
        return F.conv2d(x, W), log_det


class ZeroConv2d(nn.Module):
    """Zero-init conv with learnable scale. Keys: .conv.weight, .conv.bias, .scale"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, x):
        x = F.pad(x, [1, 1, 1, 1], value=1)
        return self.conv(x) * torch.exp(self.scale * 3)


class AffineCoupling(nn.Module):
    """Affine coupling layer. net: Sequential(conv3x3, relu, conv1x1, relu, zeroconv3x3)"""

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        half = in_channels // 2
        self.net = nn.Sequential(
            nn.Conv2d(half, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(hidden_channels, in_channels),
        )

    def forward(self, x, reverse=False):
        xa, xb = x.chunk(2, dim=1)
        out = self.net(xb)
        log_s, t = out.chunk(2, dim=1)
        # clamp: allows negative log_s; tighter range (-3,3) for numerical stability
        log_s = log_s.clamp(-3, 3)
        if reverse:
            xa = (xa - t) * torch.exp(-log_s)
            log_det = -log_s.sum(dim=[1, 2, 3])
        else:
            xa = torch.exp(log_s) * xa + t
            log_det = log_s.sum(dim=[1, 2, 3])
        return torch.cat([xa, xb], dim=1), log_det


# ---------------------------------------------------------------------------
# Flow step and block
# ---------------------------------------------------------------------------

class FlowStep(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.actnorm = ActNorm(in_channels)
        self.invconv = InvConv2d(in_channels)
        self.coupling = AffineCoupling(in_channels, hidden_channels)

    def forward(self, x, reverse=False):
        if reverse:
            x, ld3 = self.coupling(x, reverse=True)
            x, ld2 = self.invconv(x, reverse=True)
            x, ld1 = self.actnorm(x, reverse=True)
        else:
            x, ld1 = self.actnorm(x)
            x, ld2 = self.invconv(x)
            x, ld3 = self.coupling(x)
        return x, ld1 + ld2 + ld3


class GlowBlock(nn.Module):
    """K flow steps + ZeroConv prior for the split/output distribution."""

    def __init__(self, in_channels, n_flows, hidden_channels, is_last=False):
        super().__init__()
        self.flows = nn.ModuleList(
            [FlowStep(in_channels, hidden_channels) for _ in range(n_flows)]
        )
        # prior: ZeroConv maps "continue" half → (mean, logstd) of "split" half
        # For last block: maps all channels → (mean, logstd) of all channels
        prior_in = in_channels if is_last else in_channels // 2
        prior_out = in_channels * 2 if is_last else in_channels
        self.prior = ZeroConv2d(prior_in, prior_out)
        self.is_last = is_last

    def forward(self, x, reverse=False):
        if not reverse:
            return self._forward(x)
        return self._reverse(x)

    def _forward(self, x):
        log_det = 0
        for flow in self.flows:
            x, ld = flow(x)
            log_det = log_det + ld

        if self.is_last:
            # compute log p(z) under learned prior
            prior_out = self.prior(torch.zeros_like(x))
            mean, log_std = prior_out.chunk(2, dim=1)
            log_pz = -0.5 * ((x - mean) ** 2 * torch.exp(-2 * log_std)
                              + 2 * log_std + np.log(2 * np.pi))
            log_pz = log_pz.sum(dim=[1, 2, 3])
            return x, log_det, log_pz, None
        else:
            z, x_cont = x.chunk(2, dim=1)
            prior_out = self.prior(x_cont)
            mean, log_std = prior_out.chunk(2, dim=1)
            log_pz = -0.5 * ((z - mean) ** 2 * torch.exp(-2 * log_std)
                              + 2 * log_std + np.log(2 * np.pi))
            log_pz = log_pz.sum(dim=[1, 2, 3])
            return x_cont, log_det, log_pz, z

    def _reverse(self, inputs):
        """inputs: (x_cont, z_split) or (z_last,) for last block"""
        if self.is_last:
            x = inputs
            prior_out = self.prior(torch.zeros_like(x))
        else:
            x_cont, z_split = inputs
            prior_out = self.prior(x_cont)
            mean, log_std = prior_out.chunk(2, dim=1)
            x = torch.cat([z_split, x_cont], dim=1)

        log_det = 0
        for flow in reversed(self.flows):
            x, ld = flow(x, reverse=True)
            log_det = log_det + ld
        return x, log_det


# ---------------------------------------------------------------------------
# squeeze / unsqueeze
# ---------------------------------------------------------------------------

def squeeze(x):
    B, C, H, W = x.shape
    x = x.view(B, C, H // 2, 2, W // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(B, C * 4, H // 2, W // 2)


def unsqueeze(x):
    B, C, H, W = x.shape
    x = x.view(B, C // 4, 2, 2, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    return x.view(B, C // 4, H * 2, W * 2)


# ---------------------------------------------------------------------------
# Full Glow model
# ---------------------------------------------------------------------------

class Glow(nn.Module):
    """
    Multi-scale Glow for FFHQ-64x64.
    Input flow: 64x64x3 -> squeeze -> 32x32x12
    Per-block channel doubling via split+squeeze:
      block 0: 12ch, block 1: 24ch, block 2: 48ch, block 3: 96ch
    """

    def __init__(self, in_channels=3, n_blocks=4, n_flows=32, hidden_channels=512):
        super().__init__()
        C = in_channels * 4  # 12 after first squeeze
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            is_last = (i == n_blocks - 1)
            self.blocks.append(GlowBlock(C, n_flows, hidden_channels, is_last=is_last))
            if not is_last:
                C = (C // 2) * 4  # split halves C, then squeeze quadruples

    def forward(self, x):
        """Returns (z_list, total_log_det, total_log_pz)"""
        x = squeeze(x)
        log_det = 0
        log_pz = 0
        z_list = []

        for i, block in enumerate(self.blocks):
            if block.is_last:
                z, ld, lpz, _ = block(x)
                log_det = log_det + ld
                log_pz = log_pz + lpz
                z_list.append(z)
            else:
                x_cont, ld, lpz, z_split = block(x)
                log_det = log_det + ld
                log_pz = log_pz + lpz
                z_list.append(z_split)
                x = squeeze(x_cont)

        return z_list, log_det, log_pz

    def nll_loss(self, x):
        """NLL in bits/dim."""
        _, log_det, log_pz = self.forward(x)
        nll = -(log_pz + log_det)
        n_pixels = x[0].numel() if isinstance(x, (list, tuple)) else x[0:1].numel()
        n_pixels = x.shape[1] * x.shape[2] * x.shape[3]
        return (nll / (n_pixels * np.log(2))).mean()

    @torch.no_grad()
    def sample(self, n, temperature=0.7, device="cuda"):
        """Sample by reversing each block from last to first."""
        # Sample z for last block: 4x4x96
        z = torch.randn(n, 96, 4, 4, device=device) * temperature
        x, _ = self.blocks[-1]._reverse(z)

        # Reverse intermediate blocks (from second-to-last to first)
        for i in range(len(self.blocks) - 2, -1, -1):
            x = unsqueeze(x)
            # Sample z_split from prior(x)
            block = self.blocks[i]
            prior_out = block.prior(x)
            mean, log_std = prior_out.chunk(2, dim=1)
            z_split = mean + torch.randn_like(mean) * torch.exp(log_std) * temperature
            x, _ = block._reverse((x, z_split))

        x = unsqueeze(x)
        return x.clamp(-0.5, 0.5)
