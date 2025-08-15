# Train a neural ODE to shoot from x0 to x1.  With the toggle
#    match_known_traj = True
# the loss also forces the model to go through one *genuine* interior
# point of the recorded trajectory instead of an artificially chosen
# linear-interpolation point.

import math, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchdiffeq import odeint_adjoint as odeint
from pathlib import Path

# ------------------------------------------------------------------
# HyperParameters
# ------------------------------------------------------------------

# file = "experiments/gaussian/dim=5,inner_iters=10000,model_name=dsb/1/traj.npy"
# file = r"C:\dev\aim\dsbm-pytorch\experiments\gaussian\dim=5,inner_iters=2000,model_name=dsbm\1\traj.npy"
file = r"C:\dev\aim\dsbm-pytorch\traj-maria.npy"
# file = "sb_trajectories.npy"
epochs = 20
batch_size = 1024
hidden = 128
lr = 1e-3
energy_reg = 5e-4  # λ * ∫‖f‖²dt   (set 0 to disable)
energy_reg = 5e-1  # λ * ∫‖f‖²dt   (set 0 to disable)
energy_reg = 0  # λ * ∫‖f‖²dt   (set 0 to disable)
match_known_traj = True
# ------------------------------------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt_path = Path('shoot_only_best.pt')

# ---------------------- load trajectories -----------------------
traj = np.load(file)  # (B, K+1, d)
B, Kp1, d = traj.shape
K = Kp1 - 1
print(f"Loaded {B} trajectories, each of length {Kp1} and dimension {d}")

x0 = traj[:, 0, :]
x1 = traj[:, -1, :]

# -------- feature normalisation (z-score) ----------------
mu = x0.mean(0, keepdims=True)
print("Raw x1 mean:", x1.mean())
std = x0.std(0, keepdims=True) + 1e-8
traj_n = (traj - mu) / std  # complete normalised trajectory
x0_n = traj_n[:, 0, :]
x1_n = traj_n[:, -1, :]


# dataset of (x0, x1, traj)
class TrajDataset(torch.utils.data.Dataset):
    def __init__(self, x0, x1, traj_full):
        self.x0, self.x1, self.traj = x0, x1, traj_full

    def __len__(self):
        return len(self.x0)

    def __getitem__(self, i):
        return self.x0[i], self.x1[i], self.traj[i]


dataset = TrajDataset(torch.tensor(x0_n, dtype=torch.float32),
                      torch.tensor(x1_n, dtype=torch.float32),
                      torch.tensor(traj_n, dtype=torch.float32))

val_len = int(0.1 * B)
train_set, val_set = random_split(dataset, [B - val_len, val_len])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size * 2, shuffle=False)

# --------------------- baselines (un-normalized space) -----------------------

# Exact copy of x0
mse_copy = np.mean((x1 - x0) ** 2)
print(f'Baseline MSE (x̂₁ = x₀)         : {mse_copy:.6f}')

# print("Real x1 mean:", x1.mean())

# Linear transform
X_aug = np.hstack([x0, np.ones((B, 1))])
coeff, *_ = np.linalg.lstsq(X_aug, x1, rcond=None)
A, b = coeff[:-1].T, coeff[-1]
mse_aff = np.mean(((x0 @ A.T + b) - x1) ** 2)
print(f'Baseline MSE (affine)          : {mse_aff:.6f}')


# --------------------------- NeuralODE drift model ------------------------------
class DriftNet(nn.Module):
    def __init__(self, d, width=128, depth=4):
        super().__init__()
        # Fourier time-embedding
        self.freqs = 2 * torch.pi * torch.arange(1, 9)

        def time_embed(t):  # t:(N,1)
            ang = t @ self.freqs.to(t.device).view(1, -1)
            return torch.cat([t, torch.sin(ang), torch.cos(ang)], 1)

        self.time_embed = time_embed
        layers, in_dim = [], d + 1 + 16  # 1(t) + 16 Fourier
        for _ in range(depth - 1):
            layers += [nn.Linear(in_dim, width), nn.SiLU()]
            in_dim = width
        layers += [nn.Linear(width, d)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):  # x:(N,d), t:(N,1)
        return self.net(torch.cat([x, self.time_embed(t)], 1))


fθ = DriftNet(d, hidden).to(device)


class ODEFunc(nn.Module):
    def __init__(self, drift, dim): super().__init__(); self.drift, self.dim = drift, dim

    def forward(self, t, y_flat):
        x = y_flat.view(-1, self.dim)
        t_in = torch.full((x.size(0), 1), float(t), device=x.device, dtype=x.dtype)
        return self.drift(x, t_in).view(-1)


odefunc = ODEFunc(fθ, d).to(device)

# -------------------- optimiser & loss --------------------------
opt = optim.AdamW(fθ.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.)
mse = nn.MSELoss()
best = 1e9
val = None

# ---------------------- training loop ---------------------------
for epoch in range(1, epochs + 1):
    fθ.train()
    running = 0.

    for xb, x1b, trajb in train_loader:
        xb, x1b, trajb = xb.to(device), x1b.to(device), trajb.to(device)

        if match_known_traj:
            k = torch.randint(1, K, ()).item()  # integer 1..K-1
            tau = k / K
            x_tau_true = trajb[:, k, :]
        else:
            tau = torch.rand(1).item()
            x_tau_true = xb + tau * (x1b - xb)

        t_eval = torch.tensor([0., tau, 1.], device=device)

        sol = odeint(odefunc, xb.view(-1), t_eval, method='rk4',
                     adjoint_params=tuple(fθ.parameters()))
        x_tau_pred, x1_pred = sol[1].view(-1, d), sol[2].view(-1, d)

        loss_main = mse(x1_pred, x1b) + 0.2 * mse(x_tau_pred, x_tau_true)

        # kinetic-energy regulariser
        if energy_reg > 0 and val is not None and val < 0.5:
            T = len(t_eval)
            states = sol.view(T, -1, d)  # (T,B,d)
            ts = t_eval.view(T, 1, 1)
            drifts = fθ(states.reshape(-1, d),
                        ts.expand(-1, states.size(1), 1).reshape(-1, 1)
                        ).view(T, -1, d)
            energy = drifts.pow(2).mean() * energy_reg
            loss = loss_main + energy
        else:
            loss = loss_main

        opt.zero_grad()
        loss.backward()
        opt.step()
        running += loss_main.item() * xb.size(0)

    # -------------------- validation -------------------------------
    fθ.eval()
    val = 0.
    with torch.no_grad():
        for xb, x1b, _ in val_loader:
            xb, x1b = xb.to(device), x1b.to(device)
            x1_pred = odeint(odefunc, xb.view(-1),
                             torch.tensor([0., 1.], device=device),
                             method='rk4', options=dict(step_size=1. / K)
                             )[-1].view(-1, d)
            val += mse(x1_pred, x1b).item() * xb.size(0)
    val /= len(val_set)

    print(f'E{epoch:03d} train {running / len(train_set):.3e} | val {val:.3e}')

    if val < best:
        best = val
        torch.save({'state': fθ.state_dict(), 'mu': mu, 'std': std}, ckpt_path)
        print(f'  ↳ saved (best={best:.3e})')

print('done, best val =', best)
print(odefunc)

# -------------------- 5. report (raw space) ------------------------
sigma = (std if isinstance(std, np.ndarray) else std.cpu().numpy())
sigma2 = (sigma ** 2).mean()  # mean variance per dim
mse_raw = best * sigma2
rmse_dim = math.sqrt(mse_raw / d)

print(f'Model MSE  (raw space)         : {mse_raw:.3f}')
print(f'Model RMSE per dimension       : {rmse_dim:.3f}')


# ———————————————————
# --------- distribution statistics (raw space) ---------
import numpy as np
import torch

# load best weights for evaluation
ckpt = torch.load(ckpt_path, map_location=device)
fθ.load_state_dict(ckpt['state'])
fθ.eval()

# integrate from x0 (normalized) to get predicted x1, then de-normalize to raw space
with torch.no_grad():
    x0n_all = torch.tensor(x0_n, dtype=torch.float32, device=device)
    x1n_pred = odeint(
        odefunc,
        x0n_all.view(-1),
        torch.tensor([0., 1.], device=device),
        method='rk4',
        options=dict(step_size=1. / K)
    )[-1].view(-1, d)

x1_pred = x1n_pred.cpu().numpy() * std + mu  # back to raw

def compute_stats(X0, X1):
    mean_x1 = X1.mean(axis=0)
    var_x1 = X1.var(axis=0, ddof=1)
    X0c = X0 - X0.mean(axis=0, keepdims=True)
    X1c = X1 - X1.mean(axis=0, keepdims=True)
    cov_x0x1 = (X0c.T @ X1c) / (X0.shape[0] - 1)
    return mean_x1, var_x1, cov_x0x1

m_true, v_true, cov_true = compute_stats(x0, x1)
m_pred, v_pred, cov_pred = compute_stats(x0, x1_pred)

np.set_printoptions(precision=6, suppress=True)
print("Initial (true) distribution:")
print("  mean[x1] =", m_true)
print("  var[x1]  =", v_true)
print("  Cov[x0, x1] =\n", cov_true)

print("Resultant (NeuralODE) distribution:")
print("  mean[x1_hat] =", m_pred)
print("  var[x1_hat]  =", v_pred)
print("  Cov[x0, x1_hat] =\n", cov_pred)

# ————————————————————————

import matplotlib.pyplot as plt
import matplotlib.cm as cm


def straightness_ratio(traj):
    # traj: (K+1, d)
    segs = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    arc_len = np.sum(segs)
    chord = np.linalg.norm(traj[0] - traj[-1])
    return chord / arc_len if arc_len > 0 else 0.0


if __name__ == '__main__':
    # --- Visualization section ---
    fθ.eval()

    num_plot = min(8, len(val_set))
    indices = np.random.choice(len(val_set), num_plot, replace=False)
    fig, axes = plt.subplots(1, num_plot, figsize=(3 * num_plot, 3))

    for i, idx in enumerate(indices):
        x0, x1, traj = val_set[idx]
        x0 = x0.numpy()
        x1 = x1.numpy()
        gt_traj = traj.numpy()  # (K+1, d)
        gt_traj_raw = gt_traj * std + mu

        t_eval = torch.linspace(0, 1, K + 1)
        with torch.no_grad():
            pred_traj = odeint(
                odefunc,
                torch.tensor(x0.reshape(-1), device=device),
                t_eval.to(device),
                method='rk4'
            ).cpu().numpy().reshape(K + 1, d)
        pred_traj_raw = pred_traj * std + mu

        # Only plot first two coordinates
        colors = cm.plasma(np.linspace(0, 1, K + 1))
        ax = axes[i] if num_plot > 1 else axes

        # GT
        ax.scatter(gt_traj_raw[:, 0], gt_traj_raw[:, 1], c=colors, s=18, label="GT", marker='o', alpha=0.8)
        ax.plot(gt_traj_raw[:, 0], gt_traj_raw[:, 1], color='C0', lw=1, alpha=0.5)
        # NeuralODE
        ax.scatter(pred_traj_raw[:, 0], pred_traj_raw[:, 1], c=colors, s=18, label="NeuralODE", marker='x', alpha=0.8)
        ax.plot(pred_traj_raw[:, 0], pred_traj_raw[:, 1], color='C1', lw=1, alpha=0.5)
        # Chord
        ax.plot([gt_traj_raw[0, 0], gt_traj_raw[-1, 0]], [gt_traj_raw[0, 1], gt_traj_raw[-1, 1]], ':', color='gray',
                lw=1)

        # Straightness
        gt_str = straightness_ratio(gt_traj_raw[:, :2])
        pred_str = straightness_ratio(pred_traj_raw[:, :2])
        ax.set_title(f"GT S={gt_str:.2f}\nODE S={pred_str:.2f}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_aspect('equal')
        ax.grid(True)
        if i == 0:
            # Show colorbar for time (only once)
            sm = plt.cm.ScalarMappable(cmap=cm.plasma, norm=plt.Normalize(vmin=0, vmax=1))
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Time")

    plt.tight_layout()
    plt.suptitle("Trajectories in (x1, x2): GT (o), NeuralODE (x), color=time", y=1.05)
    plt.show()

    # --- (Optional) print average straightness ---
    # Compute on full val set:
    gt_straightnesses = []
    pred_straightnesses = []
    for xb, x1b, trajb in val_loader:
        xb = xb.cpu().numpy()
        trajb = trajb.cpu().numpy()
        for i in range(min(xb.shape[0], 100)):
            gt_traj = trajb[i] * std + mu
            gt_straightnesses.append(straightness_ratio(gt_traj))
            # Predict with ODE
            with torch.no_grad():
                pred_traj = odeint(
                    odefunc,
                    torch.tensor((xb[i]).reshape(-1), device=device),
                    torch.linspace(0, 1, K + 1).to(device),
                    method='rk4'
                ).cpu().numpy().reshape(K + 1, d)
            pred_traj_raw = pred_traj * std + mu
            pred_straightnesses.append(straightness_ratio(pred_traj_raw))
        print(f"Processed batch of size {xb.shape[0]}")
        break

    print(f"Mean straightness: GT = {np.mean(gt_straightnesses):.3f}, NeuralODE = {np.mean(pred_straightnesses):.3f}")

    # ======================================================================
    # 6.  EXTRA VALIDATION METRICS (paste after the previous __main__ block)
    # ======================================================================
    import scipy.linalg as sla


    def gaussian_w2(mean1, cov1, mean2, cov2):
        """
        Closed-form W₂ distance between two Gaussians 𝒩(m1,C1) and 𝒩(m2,C2)
        Eq. (2.6) in: Dowson & Landau 1982; or §2.2 in Peyré & Cuturi “Computational OT”.
        Returns the *scalar* W₂ (not the square).
        """
        mean_term = np.sum((mean1 - mean2) ** 2)
        # √C1
        sqrtC1 = sla.sqrtm(cov1)
        # Guard against tiny imaginary parts coming from numeric sqrtm
        sqrtC1 = sqrtC1.real
        prod = sqrtC1 @ cov2 @ sqrtC1
        cov_term = np.trace(cov1 + cov2 - 2 * sla.sqrtm(prod).real)
        w2_sq = mean_term + cov_term
        return float(np.sqrt(max(w2_sq, 0.)))  # clip small <0 because of round-off


    def mmd_rbf(X, Y, sigmas=(0.5, 1.0, 2.0, 4.0)):
        """
        Unbiased estimator of MMD² with mixture of RBF kernels.
        X, Y : (N,d) numpy arrays
        """
        XX = torch.from_numpy(X)
        YY = torch.from_numpy(Y)
        N, M = XX.size(0), YY.size(0)
        # pairwise ‖·‖²
        XX_sq = (XX.unsqueeze(1) - XX.unsqueeze(0)).pow(2).sum(-1)
        YY_sq = (YY.unsqueeze(1) - YY.unsqueeze(0)).pow(2).sum(-1)
        XY_sq = (XX.unsqueeze(1) - YY.unsqueeze(0)).pow(2).sum(-1)

        k = 0.
        for s in sigmas:
            k += torch.exp(-XX_sq / (2 * s * s)).sum() / (N * (N - 1))
            k += torch.exp(-YY_sq / (2 * s * s)).sum() / (M * (M - 1))
            k -= 2 * torch.exp(-XY_sq / (2 * s * s)).mean()
        return float(np.sqrt(max(k.item(), 0.)))


    def traj_energy(traj):  # traj : (T,d)
        vel = np.diff(traj, axis=0)  # Δx
        return (vel ** 2).sum() / vel.shape[0]  # mean ‖v‖²


    # ------------------------------------------------------------------
    print("\n────────────────  EXTRA VALIDATION  ────────────────")
    fθ.eval()

    # ---------- generate predicted end-points on the whole validation set
    pred_ends = []
    true_ends = []
    energy_gt = []
    energy_ode = []

    for xb, x1b, trajb in val_loader:
        xb = xb.to(device)
        with torch.no_grad():
            t_eval = torch.tensor([0., 1.], device=device)
            x1_pred = odeint(odefunc, xb.view(-1), t_eval, method='rk4')[-1]
        # de-normalise to raw space
        x1_pred_raw = (x1_pred.view(-1, d).cpu().numpy()) * std + mu
        x1_true_raw = (x1b.view(-1, d).cpu().numpy()) * std + mu

        pred_ends.append(x1_pred_raw)
        true_ends.append(x1_true_raw)

        # trajectory energies ------------------------------------------------
        trajb_raw = trajb.cpu().numpy() * std + mu
        for i in range(trajb_raw.shape[0]):
            energy_gt.append(traj_energy(trajb_raw[i]))
        # predicted trajectory (coarse  N_eval = K+1 points)
        N_eval = K + 1
        t_line = torch.linspace(0, 1, N_eval, device=device)
        with torch.no_grad():
            pred_traj = odeint(odefunc, xb.view(-1), t_line, method='rk4') \
                .cpu().numpy().reshape(N_eval, -1, d).transpose(1, 0, 2)  # (B_eval,T,d)
        for i in range(pred_traj.shape[0]):
            energy_ode.append(traj_energy(pred_traj[i] * std + mu))

    pred_ends = np.concatenate(pred_ends, axis=0)
    true_ends = np.concatenate(true_ends, axis=0)

    # ---------------- MMD ---------------------------------------------------
    mmd_val = mmd_rbf(pred_ends, true_ends)

    # ---------------- Gaussian W₂ ------------------------------------------
    μ_hat = pred_ends.mean(0, keepdims=True).ravel()
    Σ_hat = np.cov(pred_ends.T)
    μ_tgt = (true_ends.mean(0, keepdims=True)).ravel()  # should be 'a·1'
    Σ_tgt = np.cov(true_ends.T)  # ≈I
    w2_val = gaussian_w2(μ_hat, Σ_hat, μ_tgt, Σ_tgt)

    # -------------- optional : multivariate Sinkhorn ------------------------
    # import ot
    # reg    = 0.05     # entropic regularisation ε
    # a_hist = np.ones(len(pred_ends)) / len(pred_ends)
    # b_hist = np.ones(len(true_ends)) / len(true_ends)
    # sink_val = ot.sinkhorn2(a_hist, b_hist, pred_ends, true_ends, reg)[0]
    # print(f"Sinkhorn²={sink_val:.4f}")

    # -------------- kinetic energies ----------------------------------------
    print(f"MMD (RBF, mixture σ)            : {mmd_val:.5f}")
    print(f"W₂(P̂, 𝒩) (closed-form, raw)     : {w2_val:.5f}")
    print(f"Mean energy  GT trajectories    : {np.mean(energy_gt):.4f}")
    print(f"Mean energy  NeuralODE paths    : {np.mean(energy_ode):.4f}")
    print("──────────────────────────────────────────────────────")
