# sb_neural_ode_shoot.py
#
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
# hyper-parameters
# file = "experiments/gaussian/dim=5,inner_iters=10000,model_name=dsb/1/traj.npy"
file = "sb_trajectories.npy"
epochs = 100
batch_size = 1024
hidden = 128
lr = 1e-3
energy_reg = 5e-4  # λ * ∫‖f‖²dt   (set 0 to disable)
match_known_traj = False
# ------------------------------------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt_path = Path('shoot_only_best.pt')

# ---------------------- 1. load trajectories -----------------------
traj = np.load(file)  # (B, K+1, d)
B, Kp1, d = traj.shape
K = Kp1 - 1
print(f"Loaded {B} trajectories, each of length {Kp1} and dimension {d}")

x0 = traj[:, 0, :]
x1 = traj[:, -1, :]

# -------- optional feature normalisation (z-score) ----------------
mu = x0.mean(0, keepdims=True)
std = x0.std(0, keepdims=True) + 1e-8
traj_n = (traj - mu) / std  # complete normalised trajectory
x0_n = traj_n[:, 0, :]
x1_n = traj_n[:, -1, :]


# dataset that also keeps the *whole* trajectory (needed when
# match_known_traj == True)
class TrajDataset(torch.utils.data.Dataset):
    def __init__(self, x0, x1, traj_full):
        self.x0, self.x1, self.traj = x0, x1, traj_full

    def __len__(self):  return len(self.x0)

    def __getitem__(self, i):
        return self.x0[i], self.x1[i], self.traj[i]


dataset = TrajDataset(torch.tensor(x0_n, dtype=torch.float32),
                      torch.tensor(x1_n, dtype=torch.float32),
                      torch.tensor(traj_n, dtype=torch.float32))

val_len = int(0.1 * B)
train_set, val_set = random_split(dataset, [B - val_len, val_len])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size * 2, shuffle=False)

# --------------------- baselines (raw space) -----------------------
mse_copy = np.mean((x1 - x0) ** 2)
print(f'Baseline MSE (x̂₁ = x₀)         : {mse_copy:.6f}')

X_aug = np.hstack([x0, np.ones((B, 1))])
coeff, *_ = np.linalg.lstsq(X_aug, x1, rcond=None)
A, b = coeff[:-1].T, coeff[-1]
mse_aff = np.mean(((x0 @ A.T + b) - x1) ** 2)
print(f'Baseline MSE (affine)          : {mse_aff:.6f}')


# --------------------------- 2. model ------------------------------
class DriftNet(nn.Module):
    def __init__(self, d, width=256, depth=4):
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
last = fθ.net[-1]  # for optional init


class ODEFunc(nn.Module):
    def __init__(self, drift, dim): super().__init__(); self.drift, self.dim = drift, dim

    def forward(self, t, y_flat):
        x = y_flat.view(-1, self.dim)
        t_in = torch.full((x.size(0), 1), float(t), device=x.device, dtype=x.dtype)
        return self.drift(x, t_in).view(-1)


odefunc = ODEFunc(fθ, d).to(device)

# -------------------- 3. optimiser & loss --------------------------
opt = optim.AdamW(fθ.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.)
mse = nn.MSELoss()
best = 1e9
val = None

# ---------------------- 4. training loop ---------------------------
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

        # optional kinetic-energy regulariser
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

        opt.zero_grad();
        loss.backward();
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

# -------------------- 5. report (raw space) ------------------------
sigma = (std if isinstance(std, np.ndarray) else std.cpu().numpy())
sigma2 = (sigma ** 2).mean()  # mean variance per dim
mse_raw = best * sigma2
rmse_dim = math.sqrt(mse_raw / d)

print(f'Model MSE  (raw space)         : {mse_raw:.3f}')
print(f'Model RMSE per dimension       : {rmse_dim:.3f}')
