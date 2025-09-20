import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import List, Tuple

from gaussian_transport import (
    GaussianSampler,
    GaussianTransportConfig,
    ODEFunc,
    build_drift_pair,
    compute_stats,
    mmd_rbf_torch,
    ode_flow,
    plot_transport_stats,
    sde_sample,
    set_seed,
)


# ===============================================================
#  Schrodinger Bridge via IPF-style alternating drifts (sigma=1)
#  - two NeuralODE drifts (forward and backward)
#  - no trajectory supervision, only samples from x0 ~ N(+a,I), x1 ~ N(-a,I)
#  - learns by (i) distribution matching at the end-time via MMD
#              + (ii) small ODE-vs-SDE consistency loss per batch
#  - SDE sampler is Euler-Maruyama with sigma = 1
#  - At each outer iteration we alternate (IPF style) updates:
#       * train forward drift with samples from p0 to match p1
#       * train backward drift with samples from p1 to match p0
#  - We track mean/var and diag-mean covariance Cov(x0, x1_hat)
#    and plot their convergence toward target values.
# ===============================================================


@dataclass
class SBConfig(GaussianTransportConfig):
    batch_size: int = 2048
    lr: float = 2e-3
    hidden: int = 128
    depth: int = 4
    epochs: int = 30
    fwd_steps: int = 1
    bwd_steps: int = 1
    mmd_sigmas: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    ode_consistency_w: float = 0.1
    cycle_w: float = 0.0


class SBIPFTrainer:
    def __init__(self, cfg: SBConfig):
        self.cfg = cfg
        self.device = cfg.device
        self.sampler = GaussianSampler(cfg.dim, cfg.a, self.device)

        self.fwd, self.bwd = build_drift_pair(cfg, cfg.hidden, cfg.depth)
        self.ode_f = ODEFunc(self.fwd, cfg.dim).to(self.device)
        self.ode_g = ODEFunc(self.bwd, cfg.dim).to(self.device)

        self.opt_f = optim.AdamW(self.fwd.parameters(), lr=cfg.lr, betas=(0.9, 0.99))
        self.opt_g = optim.AdamW(self.bwd.parameters(), lr=cfg.lr, betas=(0.9, 0.99))
        self.mse = nn.MSELoss()

        self.means: List[float] = []
        self.vars: List[float] = []
        self.covs: List[float] = []

    def update_forward(self) -> Tuple[float, float]:
        cfg = self.cfg
        self.fwd.train()
        x0 = self.sampler.sample_x0(cfg.batch_size)
        with torch.no_grad():
            x1_tgt = self.sampler.sample_x1(cfg.batch_size)

        x1_sde = sde_sample(self.fwd, x0, cfg.K, cfg.sigma)
        x1_ode = ode_flow(self.ode_f, x0, cfg.K)

        mmd = mmd_rbf_torch(x1_sde, x1_tgt, cfg.mmd_sigmas)
        loss = mmd
        if cfg.ode_consistency_w > 0:
            loss = loss + cfg.ode_consistency_w * self.mse(x1_sde, x1_ode)
        if cfg.cycle_w > 0:
            with torch.no_grad():
                x0_cyc = sde_sample(self.bwd, x1_sde.detach(), cfg.K, cfg.sigma)
            loss = loss + cfg.cycle_w * self.mse(x0_cyc, x0)

        self.opt_f.zero_grad()
        loss.backward()
        self.opt_f.step()
        return float(loss.item()), float(mmd.item())

    def update_backward(self) -> Tuple[float, float]:
        cfg = self.cfg
        self.bwd.train()
        x1 = self.sampler.sample_x1(cfg.batch_size)
        with torch.no_grad():
            x0_tgt = self.sampler.sample_x0(cfg.batch_size)

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

        self.opt_g.zero_grad()
        loss.backward()
        self.opt_g.step()
        return float(loss.item()), float(mmd.item())

    @torch.no_grad()
    def eval_forward_stats(self, n_samples: int = 32768) -> Tuple[float, float, float]:
        cfg = self.cfg
        self.fwd.eval()
        x0 = self.sampler.sample_x0(n_samples)
        x1hat = sde_sample(self.fwd, x0, cfg.K, cfg.sigma)
        mean_val, var_val, cov_val = compute_stats(x0.cpu().numpy(), x1hat.cpu().numpy())
        return mean_val, var_val, cov_val

    def train(self) -> None:
        cfg = self.cfg
        print(f"Device: {self.device} | dim={cfg.dim} | sigma={cfg.sigma} | K={cfg.K}")
        for epoch in range(1, cfg.epochs + 1):
            f_losses, g_losses = [], []
            for _ in range(cfg.fwd_steps):
                loss_val, mmd_val = self.update_forward()
                f_losses.append((loss_val, mmd_val))
            for _ in range(cfg.bwd_steps):
                loss_val, mmd_val = self.update_backward()
                g_losses.append((loss_val, mmd_val))

            mean_val, var_val, cov_val = self.eval_forward_stats(n_samples=32768)
            self.means.append(mean_val)
            self.vars.append(var_val)
            self.covs.append(cov_val)

            f_loss = np.mean([item[0] for item in f_losses]) if f_losses else float("nan")
            g_loss = np.mean([item[0] for item in g_losses]) if g_losses else float("nan")
            print(
                f"E{epoch:03d}  Lf={f_loss:.4f}  Lg={g_loss:.4f}  | "
                f"mean={mean_val:.3f}->{cfg.targets.mean}  "
                f"var={var_val:.3f}->{cfg.targets.variance}  "
                f"cov={cov_val:.3f}->{cfg.targets.covariance}"
            )

    def plot_convergence(self) -> None:
        if not self.means:
            print("No training history to plot.")
            return
        epochs = np.arange(1, len(self.means) + 1)
        plot_transport_stats(
            means=self.means,
            variances=self.vars,
            covariances=self.covs,
            targets=self.cfg.targets,
            epochs=epochs,
        )


if __name__ == "__main__":
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

    with torch.no_grad():
        x0 = trainer.sampler.sample_x0(65536)
        x1_hat = sde_sample(trainer.fwd, x0, cfg.K, cfg.sigma)
    mean_val, var_val, cov_val = compute_stats(x0.cpu().numpy(), x1_hat.cpu().numpy())
    print("\nFinal stats (forward SDE):")
    print(f"  mean[x1_hat] = {mean_val:.4f} (target {cfg.targets.mean})")
    print(f"  var[x1_hat]  = {var_val:.4f} (target {cfg.targets.variance})")
    print(f"  cov_diagmean = {cov_val:.4f} (target {cfg.targets.covariance})")
