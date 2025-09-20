import math, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional

# ===============================================================
#  Schrödinger Bridge via IPF-style alternating drifts (σ=1)
#  - two NeuralODE drifts (forward and backward)
#  - no trajectory supervision, only samples from x0 ~ N(+a,I), x1 ~ N(-a,I)
#  - learns by (i) distribution matching at the end-time via MMD
#              + (ii) small ODE-vs-SDE consistency loss per batch
#  - SDE sampler is Euler–Maruyama with σ = 1
#  - At each outer iteration we alternate (IPF style) updates:
#       * train forward drift with samples from μ0 to match μ1
#       * train backward drift with samples from μ1 to match μ0
#  - We track mean/var and diag-mean covariance Cov(x0, x1_hat)
#    and plot their convergence toward target values.
# ===============================================================

# ------------------------- Utils -------------------------------

def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class SBConfig:
    dim: int = 5
    a: float = 0.1
    sigma: float = 1.0        # diffusion strength in SDE (dx = f dt + sigma dW)
    K: int = 64               # time steps for SDE (and ODE step_size = 1/K)
    batch_size: int = 2048
    lr: float = 2e-3
    hidden: int = 128
    depth: int = 4
    epochs: int = 30          # outer IPF iterations
    fwd_steps: int = 1        # inner updates per outer iteration
    bwd_steps: int = 1
    mmd_sigmas: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    ode_consistency_w: float = 0.1  # weight for ||x1_ode - x1_sde||^2
    cycle_w: float = 0.0            # optional cycle x0->x1_hat->x0_cyc loss
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class GaussianSampler:
    """Independent sampling from μ0 = N(+a, I_d) and μ1 = N(−a, I_d)."""
    def __init__(self, dim: int, a: float, device: str):
        self.dim, self.a, self.device = dim, a, device

    def sample_x0(self, n: int) -> torch.Tensor:
        return torch.randn(n, self.dim, device=self.device) + self.a

    def sample_x1(self, n: int) -> torch.Tensor:
        return torch.randn(n, self.dim, device=self.device) - self.a


# --------------------- Model definitions -----------------------

class DriftNet(nn.Module):
    def __init__(self, d: int, width: int = 128, depth: int = 4):
        super().__init__()
        self.freqs = 2 * torch.pi * torch.arange(1, 9)
        def time_embed(t: torch.Tensor):  # (N,1)
            ang = t @ self.freqs.to(t.device).view(1, -1)
            return torch.cat([t, torch.sin(ang), torch.cos(ang)], 1)
        self.time_embed = time_embed

        layers: List[nn.Module] = []
        in_dim = d + 1 + 16  # x + [t, sin, cos]
        for _ in range(depth - 1):
            layers += [nn.Linear(in_dim, width), nn.SiLU()]
            in_dim = width
        layers += [nn.Linear(width, d)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self.time_embed(t)], 1))


class ODEFunc(nn.Module):
    def __init__(self, drift: DriftNet, dim: int):
        super().__init__(); self.drift, self.dim = drift, dim
    def forward(self, t: torch.Tensor, y_flat: torch.Tensor) -> torch.Tensor:
        x = y_flat.view(-1, self.dim)
        t_in = torch.full((x.size(0), 1), float(t), device=x.device, dtype=x.dtype)
        return self.drift(x, t_in).view(-1)


# ----------------------- Samplers ------------------------------

def sde_sample(drift: DriftNet, x0: torch.Tensor, K: int, sigma: float) -> torch.Tensor:
    """Euler–Maruyama sampler from t=0 to t=1 with K steps."""
    dt = 1.0 / K
    x = x0.clone()
    for i in range(K):
        t_now = torch.full((x.size(0), 1), i * dt, device=x.device, dtype=x.dtype)
        f = drift(x, t_now)
        noise = torch.randn_like(x) * math.sqrt(dt) * sigma
        x = x + f * dt + noise
    return x


def ode_flow(odefunc: ODEFunc, x0: torch.Tensor, K: int) -> torch.Tensor:
    with torch.no_grad():
        x1 = odeint(odefunc, x0.view(-1), torch.tensor([0., 1.], device=x0.device),
                    method='rk4', options=dict(step_size=1.0 / K))[-1].view_as(x0)
    return x1


# ------------------------ Losses --------------------------------

def pairwise_sq_dists(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    # X: (N,d), Y:(M,d)
    X2 = (X*X).sum(dim=1, keepdim=True)  # (N,1)
    Y2 = (Y*Y).sum(dim=1, keepdim=True).T  # (1,M)
    return X2 + Y2 - 2.0 * X @ Y.T


def mmd_rbf_torch(X: torch.Tensor, Y: torch.Tensor, sigmas: Tuple[float, ...]) -> torch.Tensor:
    # Unbiased MMD^2 with RBF mixture
    N, M = X.size(0), Y.size(0)
    XX = pairwise_sq_dists(X, X)
    YY = pairwise_sq_dists(Y, Y)
    XY = pairwise_sq_dists(X, Y)
    mmd2 = 0.0
    for s in sigmas:
        s2 = 2.0 * (s * s)
        Kxx = torch.exp(-XX / s2)
        Kyy = torch.exp(-YY / s2)
        Kxy = torch.exp(-XY / s2)
        # remove diagonal for unbiased estimate
        mmd2 = mmd2 + (Kxx.sum() - Kxx.diag().sum()) / (N*(N-1)) \
               + (Kyy.sum() - Kyy.diag().sum()) / (M*(M-1)) \
               - 2.0 * Kxy.mean()
    return torch.clamp(mmd2, min=0.0)


# ------------------------ Metrics -------------------------------

def compute_stats(x0: np.ndarray, x1hat: np.ndarray) -> Tuple[float, float, float]:
    m = float(x1hat.mean())
    v = float(x1hat.var(ddof=1))
    X0c = x0 - x0.mean(0, keepdims=True)
    X1c = x1hat - x1hat.mean(0, keepdims=True)
    Cov = (X0c.T @ X1c) / (x0.shape[0] - 1)
    cov_scalar = float(np.trace(Cov) / Cov.shape[0])
    return m, v, cov_scalar


# ------------------------ Trainer -------------------------------

class SBIPFTrainer:
    def __init__(self, cfg: SBConfig):
        self.cfg = cfg
        self.device = cfg.device
        self.sampler = GaussianSampler(cfg.dim, cfg.a, self.device)

        self.fwd = DriftNet(cfg.dim, cfg.hidden, cfg.depth).to(self.device)
        self.bwd = DriftNet(cfg.dim, cfg.hidden, cfg.depth).to(self.device)
        self.ode_f = ODEFunc(self.fwd, cfg.dim).to(self.device)
        self.ode_g = ODEFunc(self.bwd, cfg.dim).to(self.device)

        self.opt_f = optim.AdamW(self.fwd.parameters(), lr=cfg.lr, betas=(0.9, 0.99))
        self.opt_g = optim.AdamW(self.bwd.parameters(), lr=cfg.lr, betas=(0.9, 0.99))
        self.mse = nn.MSELoss()

        # Tracking
        self.means: List[float] = []
        self.vars: List[float] = []
        self.covs: List[float] = []

    # ----- single update steps -----
    def update_forward(self):
        cfg = self.cfg
        self.fwd.train()
        x0 = self.sampler.sample_x0(cfg.batch_size)
        with torch.no_grad():
            x1_tgt = self.sampler.sample_x1(cfg.batch_size)
        # SDE sample using forward drift
        x1_sde = sde_sample(self.fwd, x0, cfg.K, cfg.sigma)
        # ODE flow endpoint (no grad; used for consistency target)
        x1_ode = ode_flow(self.ode_f, x0, cfg.K)
        # Distribution matching loss (MMD)
        mmd = mmd_rbf_torch(x1_sde, x1_tgt, cfg.mmd_sigmas)
        loss = mmd
        # ODE vs SDE endpoint consistency (encourages drift to explain expected motion)
        if cfg.ode_consistency_w > 0:
            loss = loss + cfg.ode_consistency_w * self.mse(x1_sde, x1_ode)
        # Optional cycle consistency through backward drift
        if cfg.cycle_w > 0:
            with torch.no_grad():
                x0_cyc = sde_sample(self.bwd, x1_sde.detach(), cfg.K, cfg.sigma)  # g acts forward-in-time from 1→0 via time labels inside net
            loss = loss + cfg.cycle_w * self.mse(x0_cyc, x0)

        self.opt_f.zero_grad(); loss.backward(); self.opt_f.step()
        return float(loss.item()), float(mmd.item())

    def update_backward(self):
        cfg = self.cfg
        self.bwd.train()
        x1 = self.sampler.sample_x1(cfg.batch_size)
        with torch.no_grad():
            x0_tgt = self.sampler.sample_x0(cfg.batch_size)
        # SDE sample using backward drift (interpreted as 1→0 dynamics)
        x0_sde = sde_sample(self.bwd, x1, cfg.K, cfg.sigma)
        x0_ode = ode_flow(self.ode_g, x1, cfg.K)
        mmd = mmd_rbf_torch(x0_sde, x0_tgt, cfg.mmd_sigmas)
        loss = mmd
        if cfg.ode_consistency_w > 0:
            loss = loss + cfg.ode_consistency_w * self.mse(x0_sde, x0_ode)
        if cfg.cycle_w > 0:
            with torch.no_grad():
                x1_cyc = sde_sample(self.fwd, x0_sde.detach(), cfg.K, cfg.sigma)
            loss = loss + cfg.cycle_w * self.mse(x1_cyc, x1)

        self.opt_g.zero_grad(); loss.backward(); self.opt_g.step()
        return float(loss.item()), float(mmd.item())

    # ----- evaluation -----
    @torch.no_grad()
    def eval_forward_stats(self, N: int = 32768) -> Tuple[float, float, float]:
        cfg = self.cfg
        self.fwd.eval()
        x0 = self.sampler.sample_x0(N)
        x1hat = sde_sample(self.fwd, x0, cfg.K, cfg.sigma)
        m, v, c = compute_stats(x0.cpu().numpy(), x1hat.cpu().numpy())
        return m, v, c

    # ----- training loop -----
    def train(self):
        cfg = self.cfg
        print(f"Device: {self.device} | dim={cfg.dim} | σ={cfg.sigma} | K={cfg.K}")
        for epoch in range(1, cfg.epochs + 1):
            # IPF-style alternating updates
            f_losses, g_losses = [], []
            for _ in range(cfg.fwd_steps):
                l, m = self.update_forward()
                f_losses.append((l, m))
            for _ in range(cfg.bwd_steps):
                l, m = self.update_backward()
                g_losses.append((l, m))

            # Stats via forward sampling (x0 → x1_hat)
            m, v, c = self.eval_forward_stats(N=32768)
            self.means.append(m); self.vars.append(v); self.covs.append(c)

            f_loss = np.mean([x[0] for x in f_losses]) if f_losses else float('nan')
            g_loss = np.mean([x[0] for x in g_losses]) if g_losses else float('nan')
            print(f"E{epoch:03d}  Lf={f_loss:.4f}  Lg={g_loss:.4f}  | mean={m:.3f}→-0.1  var={v:.3f}→1  cov={c:.3f}→0.62")

    # ----- plotting -----
    def plot_convergence(self):
        epochs = np.arange(1, len(self.means) + 1)
        fig, axs = plt.subplots(1, 3, figsize=(12, 3))
        axs[0].plot(epochs, self.means, label='mean[x1_hat]')
        axs[0].axhline(-0.1, linestyle=':', color='k', label='target -0.1')
        axs[0].set_title('Mean (raw)'); axs[0].set_xlabel('Epoch'); axs[0].grid(True); axs[0].legend()

        axs[1].plot(epochs, self.vars, label='var[x1_hat]')
        axs[1].axhline(1.0, linestyle=':', color='k', label='target 1')
        axs[1].set_title('Variance (raw)'); axs[1].set_xlabel('Epoch'); axs[1].grid(True); axs[1].legend()

        axs[2].plot(epochs, self.covs, label='diag-mean Cov(x0,x1_hat)')
        axs[2].axhline(0.62, linestyle=':', color='k', label='target ≈0.62')
        axs[2].set_title('Covariance diag-mean (raw)'); axs[2].set_xlabel('Epoch'); axs[2].grid(True); axs[2].legend()

        plt.tight_layout(); plt.show()


# ------------------------ Main ---------------------------------
if __name__ == '__main__':
    set_seed(0)
    cfg = SBConfig(
        dim=5,
        a=0.1,
        sigma=1.0,
        K=64,
        batch_size=2048,
        lr=2e-3,
        hidden=128,
        depth=4,
        epochs=300,
        fwd_steps=1,
        bwd_steps=1,
        mmd_sigmas=(0.5, 1.0, 2.0, 4.0),
        ode_consistency_w=0.1,
        cycle_w=0.0,
    )

    trainer = SBIPFTrainer(cfg)
    trainer.train()
    trainer.plot_convergence()

    # Final report with a larger fresh sample
    with torch.no_grad():
        x0 = trainer.sampler.sample_x0(65536)
        x1_hat = sde_sample(trainer.fwd, x0, cfg.K, cfg.sigma)
    m, v, c = compute_stats(x0.cpu().numpy(), x1_hat.cpu().numpy())
    print("\nFinal stats (forward SDE):")
    print(f"  mean[x1_hat] = {m:.4f} (target -0.1)")
    print(f"  var[x1_hat]  = {v:.4f} (target 1.0)")
    print(f"  cov_diagmean = {c:.4f} (target ≈0.62)")
