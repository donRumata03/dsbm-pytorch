import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint


__all__ = [
    "set_seed",
    "TransportTargets",
    "GaussianTransportConfig",
    "GaussianSampler",
    "DriftNet",
    "ODEFunc",
    "sde_sample",
    "ode_flow",
    "pairwise_sq_dists",
    "mmd_rbf_torch",
    "compute_stats",
    "plot_transport_stats",
    "create_drift_net",
    "build_drift_pair",
]


def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TransportTargets:
    mean: float = -0.1
    variance: float = 1.0
    covariance: float = 0.62


@dataclass
class GaussianTransportConfig:
    dim: int = 5
    a: float = 0.1
    sigma: float = 1.0
    K: int = 64
    time_samples: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    targets: TransportTargets = field(default_factory=TransportTargets)


class GaussianSampler:
    """Independent sampling from N(+a, I_d) and N(-a, I_d)."""

    def __init__(self, dim: int, a: float, device: str):
        self.dim = dim
        self.a = a
        self.device = device

    def sample_x0(self, n: int) -> torch.Tensor:
        return torch.randn(n, self.dim, device=self.device) + self.a

    def sample_x1(self, n: int) -> torch.Tensor:
        return torch.randn(n, self.dim, device=self.device) - self.a


class DriftNet(nn.Module):
    def __init__(self, dim: int, width: int = 128, depth: int = 4):
        super().__init__()
        freqs = 2 * torch.pi * torch.arange(1, 9, dtype=torch.float32)
        self.register_buffer("freqs", freqs)

        def time_embed(t: torch.Tensor) -> torch.Tensor:
            angles = t @ self.freqs.view(1, -1)
            return torch.cat([t, torch.sin(angles), torch.cos(angles)], dim=1)

        self.time_embed = time_embed

        layers = []
        in_dim = dim + 1 + 16
        for _ in range(depth - 1):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.SiLU())
            in_dim = width
        layers.append(nn.Linear(width, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self.time_embed(t)], dim=1))


class ODEFunc(nn.Module):
    def __init__(self, drift: DriftNet, dim: int):
        super().__init__()
        self.drift = drift
        self.dim = dim

    def forward(self, t: torch.Tensor, y_flat: torch.Tensor) -> torch.Tensor:
        x = y_flat.view(-1, self.dim)
        t_in = torch.full((x.size(0), 1), float(t), device=x.device, dtype=x.dtype)
        return self.drift(x, t_in).view(-1)


def sde_sample(drift: DriftNet, x0: torch.Tensor, K: int, sigma: float) -> torch.Tensor:
    """Euler-Maruyama sampler from t=0 to t=1 with K steps."""
    dt = 1.0 / K
    x = x0.clone()
    for step in range(K):
        t_now = torch.full((x.size(0), 1), step * dt, device=x.device, dtype=x.dtype)
        drift_val = drift(x, t_now)
        noise = torch.randn_like(x) * math.sqrt(dt) * sigma
        x = x + drift_val * dt + noise
    return x


def ode_flow(
    odefunc: ODEFunc,
    x0: torch.Tensor,
    K: int,
    method: str = "rk4",
) -> torch.Tensor:
    with torch.no_grad():
        time_grid = torch.tensor([0.0, 1.0], device=x0.device, dtype=x0.dtype)
        x1 = odeint(odefunc, x0.view(-1), time_grid, method=method, options={"step_size": 1.0 / K})
    return x1[-1].view_as(x0)


def pairwise_sq_dists(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    X2 = (X * X).sum(dim=1, keepdim=True)
    Y2 = (Y * Y).sum(dim=1, keepdim=True).transpose(0, 1)
    return X2 + Y2 - 2.0 * X @ Y.transpose(0, 1)


def mmd_rbf_torch(X: torch.Tensor, Y: torch.Tensor, sigmas: Tuple[float, ...]) -> torch.Tensor:
    N, M = X.size(0), Y.size(0)
    if N < 2 or M < 2:
        raise ValueError("Need at least two samples per batch to compute unbiased MMD")

    XX = pairwise_sq_dists(X, X)
    YY = pairwise_sq_dists(Y, Y)
    XY = pairwise_sq_dists(X, Y)

    mmd2 = 0.0
    for sigma in sigmas:
        scale = 2.0 * sigma * sigma
        Kxx = torch.exp(-XX / scale)
        Kyy = torch.exp(-YY / scale)
        Kxy = torch.exp(-XY / scale)
        mmd2 = mmd2 + (Kxx.sum() - Kxx.diag().sum()) / (N * (N - 1))
        mmd2 = mmd2 + (Kyy.sum() - Kyy.diag().sum()) / (M * (M - 1))
        mmd2 = mmd2 - 2.0 * Kxy.mean()

    return torch.clamp(mmd2, min=0.0)


def compute_stats(x0: np.ndarray, x1hat: np.ndarray) -> Tuple[float, float, float]:
    mean_val = float(x1hat.mean())
    var_val = float(x1hat.var(ddof=1))
    x0_centered = x0 - x0.mean(axis=0, keepdims=True)
    x1_centered = x1hat - x1hat.mean(axis=0, keepdims=True)
    cov = (x0_centered.T @ x1_centered) / (x0.shape[0] - 1)
    cov_scalar = float(np.trace(cov) / cov.shape[0])
    return mean_val, var_val, cov_scalar


def plot_transport_stats(
    means: Sequence[float],
    variances: Sequence[float],
    covariances: Sequence[float],
    targets: Optional[TransportTargets] = None,
    epochs: Optional[Sequence[int]] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 3),
):
    if epochs is None:
        epochs = range(1, len(means) + 1)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].plot(epochs, means, label="mean[x1_hat]")
    if targets is not None:
        axes[0].axhline(targets.mean, linestyle=":", color="k", label=f"target {targets.mean}")
    axes[0].set_title("Mean")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(epochs, variances, label="var[x1_hat]")
    if targets is not None:
        axes[1].axhline(targets.variance, linestyle=":", color="k", label=f"target {targets.variance}")
    axes[1].set_title("Variance")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(epochs, covariances, label="cov[x0,x1_hat]")
    if targets is not None:
        axes[2].axhline(targets.covariance, linestyle=":", color="k", label=f"target {targets.covariance}")
    axes[2].set_title("Covariance diag-mean")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    if show:
        plt.show()
    return fig, axes


def create_drift_net(dim: int, hidden: int = 128, depth: int = 4) -> DriftNet:
    return DriftNet(dim=dim, width=hidden, depth=depth)


def build_drift_pair(
    config: GaussianTransportConfig,
    hidden: int = 128,
    depth: int = 4,
) -> Tuple[DriftNet, DriftNet]:
    fwd = create_drift_net(config.dim, hidden, depth).to(config.device)
    bwd = create_drift_net(config.dim, hidden, depth).to(config.device)
    return fwd, bwd
