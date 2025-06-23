# sb_neural_ode_shoot.py
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchdiffeq import odeint_adjoint as odeint
from pathlib import Path

# ------------ hyper-parameters ------------------
file = 'sb_trajectories.npy'
epochs = 15
batch_size = 1024
hidden = 128
lr = 2e-3
energy_reg = 1e-3  # λ * ∫‖f‖²dt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt_path = Path('shoot_only_best.pt')
# -------------------------------------------------

# 1.  load (x0,x1) pairs
traj = np.load(file)  # (B,K+1,d)
B, Kp1, d = traj.shape
K = Kp1 - 1
x0 = traj[:, 0, :]
x1 = traj[:, -1, :]

# (optional) feature normalisation
mu = x0.mean(0, keepdims=True)
std = x0.std(0, keepdims=True) + 1e-8
x0_n = (x0 - mu) / std
x1_n = (x1 - mu) / std

dataset = TensorDataset(torch.tensor(x0_n, dtype=torch.float32),
                        torch.tensor(x1_n, dtype=torch.float32))
val_len = int(0.1 * B)
train_set, val_set = random_split(dataset, [B - val_len, val_len])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size * 2, shuffle=False)


# 2.  model
class DriftNet(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x, t):  # x:(N,d)  t:(N,1)
        return self.net(torch.cat([x, t], 1))


fθ = DriftNet(d, hidden).to(device)


class ODEFunc(nn.Module):
    def __init__(self, drift, dim): super().__init__(); self.drift, self.dim = drift, dim

    def forward(self, t, y_flat):
        x = y_flat.view(-1, self.dim)
        t_in = torch.full((x.size(0), 1), float(t), device=x.device, dtype=x.dtype)
        return self.drift(x, t_in).view(-1)


odefunc = ODEFunc(fθ, d).to(device)

# 3.  optimiser
opt = optim.AdamW(fθ.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.)
mse = nn.MSELoss()
best = 1e9

# 4.  training loop
for epoch in range(1, epochs + 1):
    fθ.train();
    running = 0.
    for xb, x1b in train_loader:
        xb, x1b = xb.to(device), x1b.to(device)

        # number of grid points = Kp1  (= 21 for your data)
        t_eval = torch.linspace(0., 1., steps=Kp1, device=device)

        sol = odeint(odefunc,  # shape (Kp1, batch*d)
                     xb.view(-1),
                     t_eval,
                     method='rk4', options=dict(step_size=1. / K),
                     adjoint_params=tuple(fθ.parameters()))
        x1_pred = sol[-1].view(-1, d)

        loss_main = mse(x1_pred, x1b)

        if energy_reg > 0:
            # reshape into (time, batch, dim)
            states = sol.view(Kp1, -1, d)
            ts = t_eval.view(-1, 1, 1)  # (time,1,1)

            # compute drift on every snapshot
            drifts = fθ(states.reshape(-1, d),
                        ts.expand(-1, states.size(1), 1).reshape(-1, 1)
                        ).view(Kp1, -1, d)
            energy = drifts.pow(2).mean() * energy_reg  # ∫‖f‖² dt (≈ mean)
            loss = loss_main + energy
        else:
            loss = loss_main

        opt.zero_grad()
        loss.backward()
        opt.step()
        running += loss_main.item() * xb.size(0)

    # validation
    fθ.eval();
    val = 0.
    with torch.no_grad():
        for xb, x1b in val_loader:
            xb, x1b = xb.to(device), x1b.to(device)
            x1_pred = odeint(odefunc, xb.view(-1),
                             torch.tensor([0., 1.], device=device),
                             method='rk4', options=dict(step_size=1. / K))[-1].view(-1, d)
            val += mse(x1_pred, x1b).item() * xb.size(0)
    val /= len(val_set)

    print(f'E{epoch:03d} train {running / len(train_set):.3e} | val {val:.3e}')

    if val < best:
        best = val
        torch.save({'state': fθ.state_dict(), 'mu': mu, 'std': std}, ckpt_path)
        print('  ↳ saved (best={:.3e})'.format(best))

print('done, best val =', best)

import numpy as np, torch, math

# --------------- baselines ---------------------------------------
x0_np = traj[:, 0, :]
x1_np = traj[:, -1, :]

mse_copy = np.mean((x1_np - x0_np) ** 2)
print(f'Baseline MSE (x̂₁ = x₀)         : {mse_copy:.3f}')

B, d = x0_np.shape
X_aug = np.hstack([x0_np, np.ones((B, 1))])

coeff, *_ = np.linalg.lstsq(X_aug, x1_np, rcond=None)
A = coeff[:-1].T        # (d,d)
b = coeff[-1]           # (d,)
mse_aff = np.mean(((x0_np @ A.T + b) - x1_np) ** 2)
print(f'Baseline MSE (affine)          : {mse_aff:.3f}')

# --------------- NeuralODE error ---------------------------------
ckpt  = torch.load('shoot_only_best.pt', map_location='cpu')
best  = 0.6635          # <- copy from your log or read from ckpt if stored

sigma = ckpt['std']
if isinstance(sigma, torch.Tensor):
    sigma = sigma.cpu().numpy()
sigma2 = (sigma ** 2).mean()          # mean variance per dim

mse_raw  = best * sigma2
rmse_dim = math.sqrt(mse_raw / d)

print(f'Model MSE  (raw space)         : {mse_raw:.3f}')
print(f'Model RMSE per dimension       : {rmse_dim:.3f}')
