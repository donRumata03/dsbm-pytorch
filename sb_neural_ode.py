# sb_neural_ode_shoot.py
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchdiffeq import odeint_adjoint as odeint
from pathlib import Path

# ------------ hyper-parameters ------------------
# file = 'sb_trajectories.npy'
file = "experiments/gaussian/dim=5,inner_iters=10000,model_name=dsb/1/traj.npy"
epochs = 100
batch_size = 1024
hidden = 128
lr = 1e-3
energy_reg = 5e-4  # λ * ∫‖f‖²dt
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt_path = Path('shoot_only_best.pt')
# -------------------------------------------------

# 1.  load (x0,x1) pairs
traj = np.load(file)  # (B,K+1,d)
B, Kp1, d = traj.shape
K = Kp1 - 1
print(f"Loaded {B} trajectories, each of length {Kp1} and dimension {d}")
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

# --------------- baselines ---------------------------------------
x0_np = traj[:, 0, :]
x1_np = traj[:, -1, :]

mse_copy = np.mean((x1_np - x0_np) ** 2)
print(f'Baseline MSE (x̂₁ = x₀)         : {mse_copy:.6f}')

B, d = x0_np.shape
X_aug = np.hstack([x0_np, np.ones((B, 1))])

coeff, *_ = np.linalg.lstsq(X_aug, x1_np, rcond=None)
A = coeff[:-1].T  # (d,d)
b = coeff[-1]  # (d,)
mse_aff = np.mean(((x0_np @ A.T + b) - x1_np) ** 2)
print(f'Baseline MSE (affine)          : {mse_aff:.6f}')


# 2.  model
# class DriftNet(nn.Module):
#     def __init__(self, dim, hidden):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim + 1, hidden), nn.SiLU(),
#             nn.Linear(hidden, hidden), nn.SiLU(),
#             nn.Linear(hidden, dim)
#         )
# 
#     def forward(self, x, t):  # x:(N,d)  t:(N,1)
#         return self.net(torch.cat([x, t], 1))
# 


class DriftNet(nn.Module):
    def __init__(self, d, width=256, depth=4):
        super().__init__()
        layers = []
        dummy = self.time_embed(torch.zeros(1, 1))
        embed_dim = dummy.shape[1]
        in_dim = d + embed_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(in_dim, width), nn.SiLU()]
            in_dim = width
        layers += [nn.Linear(width, d)]
        self.net = nn.Sequential(*layers)

    def time_embed(self, t):  # t:(N,1)
        # 8 Fourier frequencies
        freqs = 2 * torch.pi * torch.arange(1, 9, device=t.device)
        ang = t @ freqs.view(1, -1)  # (N,8)
        return torch.cat([t, torch.sin(ang), torch.cos(ang)], dim=1)

    def forward(self, x, t):
        h = self.time_embed(t)
        return self.net(torch.cat([x, h], 1))


fθ = DriftNet(d, hidden).to(device)
last = fθ.net[-1]  # final Linear
last.weight.data.zero_()
last.bias.data.copy_(torch.tensor(b, dtype=torch.float32))  # b from lstsq


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
val = None

# 4.  training loop
for epoch in range(1, epochs + 1):
    fθ.train();
    running = 0.
    for xb, x1b in train_loader:
        xb, x1b = xb.to(device), x1b.to(device)

        # draw a random tau ∼ U(0,1)  (different for every minibatch call)
        tau = torch.rand(1).item()
        t_eval = torch.tensor([0., tau, 1.], device=device)

        sol = odeint(odefunc, xb.view(-1), t_eval, method='rk4',
                     adjoint_params=tuple(fθ.parameters()))
        x_tau_pred, x1_pred = sol[1].view(-1, d), sol[2].view(-1, d)

        # true x(τ) = linear interpolation (almost exact for your trajectories)
        x_tau_true = xb + tau * (x1b - xb)

        loss_main = mse(x1_pred, x1b) + 0.2 * mse(x_tau_pred, x_tau_true)

        if energy_reg > 0 and val is not None and val < 0.5:
            # reshape into (time, batch, dim)
            T = len(t_eval)  # 3
            states = sol.view(T, -1, d)
            ts = t_eval.view(T, 1, 1)

            # compute drift on every snapshot
            drifts = fθ(states.reshape(-1, d),
                        ts.expand(-1, states.size(1), 1).reshape(-1, 1)
                        ).view(T, -1, d)
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

# --------------- NeuralODE error ---------------------------------
ckpt = torch.load('shoot_only_best.pt', map_location='cpu')
# best = 0.6635  # <- copy from your log or read from ckpt if stored

sigma = ckpt['std']
if isinstance(sigma, torch.Tensor):
    sigma = sigma.cpu().numpy()
sigma2 = (sigma ** 2).mean()  # mean variance per dim

mse_raw = best * sigma2
rmse_dim = math.sqrt(mse_raw / d)

print(f'Model MSE  (raw space)         : {mse_raw:.3f}')
print(f'Model RMSE per dimension       : {rmse_dim:.3f}')

# 0.00002
# 0.000064