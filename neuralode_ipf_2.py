import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# HyperParameters
# ------------------------------------------------------------------
# Problem Definition: Transport N(a, σ²I) -> N(-a, σ²I)
D = 5              # Dimension of the problem
A_VAL = 0.1        # Mean offset
SIGMA = 1.0        # Std dev of distributions & SDE noise variance

# IPF Training for Schrödinger Bridge
IPF_ITERATIONS = 15         # Number of outer IPF iterations
INNER_EPOCHS = 5            # Number of training epochs for f/b in each IPF iter
NUM_TRAIN_SAMPLES = 20000   # Number of samples for training datasets
NUM_EVAL_SAMPLES = 5000     # Number of samples for evaluation/stats

# Model & Optimizer
BATCH_SIZE = 1024
HIDDEN = 128
LR = 1e-3

# SDE Sampler
K = 50  # Number of steps for the SDE sampler

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# --------------------------- NeuralODE drift model ------------------------------
class DriftNet(nn.Module):
    """Neural network for approximating the drift of an SDE."""
    def __init__(self, d, width=128, depth=4):
        super().__init__()
        # Fourier time-embedding
        self.freqs = 2 * torch.pi * torch.arange(1, 9, device=device)

        def time_embed(t):  # t:(N,1)
            ang = t @ self.freqs.view(1, -1)
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

class ODEFunc(nn.Module):
    """Wrapper for DriftNet to be used with torchdiffeq for training."""
    def __init__(self, drift, dim, backward=False):
        super().__init__()
        self.drift = drift
        self.dim = dim
        self.backward = backward

    def forward(self, t, y_flat):
        x = y_flat.view(-1, self.dim)
        # The network always sees time progressing from 0 to 1.
        # For the backward model, the solver integrates from t=1 to t=0,
        # so we feed the network `1-t`.
        t_in_val = 1.0 - t if self.backward else t
        t_in = torch.full((x.size(0), 1), float(t_in_val), device=x.device, dtype=x.dtype)
        return self.drift(x, t_in).view(-1)


# --------------------------- SDE Sampler ------------------------------------
@torch.no_grad()
def euler_maruyama_sampler(model, x_initial, T, K, sigma, forward=True):
    """
    Generates samples using the Euler-Maruyama method for the SDE.
    dx_t = model(x_t, t) dt + sigma dW_t
    """
    model.eval()
    dt = T / K
    x_t = x_initial.clone()

    time_steps = torch.linspace(0, T, K + 1, device=device)
    if not forward:
        time_steps = torch.flip(time_steps, [0])

    for i in range(K):
        t_now = time_steps[i]

        # For the backward model, the network expects time s=1-t
        t_in_val = (T - t_now) if not forward else t_now
        t_in = torch.full((x_t.size(0), 1), t_in_val, device=device, dtype=torch.float32)

        drift = model(x_t, t_in)
        noise = torch.randn_like(x_t) * math.sqrt(abs(dt)) * sigma

        x_t = x_t + drift * dt if forward else x_t - drift * dt
        x_t = x_t + noise

    return x_t


# --------------------------- IPF Trainer ------------------------------------
class IPFTrainer:
    """
    Manages the models, data, and training loop for Iterative Proportional Fitting
    to solve the Schrödinger Bridge problem.
    """
    def __init__(self, d, a_val, sigma, device, **kwargs):
        self.d = d
        self.a = torch.full((d,), a_val, device=device)
        self.sigma = sigma
        self.device = device

        # Unpack kwargs
        self.hidden_dim = kwargs.get('hidden', 128)
        self.lr = kwargs.get('lr', 1e-3)

        # Initialize forward and backward models and their optimizers
        self.f_net = DriftNet(d, self.hidden_dim).to(device)
        self.b_net = DriftNet(d, self.hidden_dim).to(device)
        self.opt_f = optim.AdamW(self.f_net.parameters(), lr=self.lr)
        self.opt_b = optim.AdamW(self.b_net.parameters(), lr=self.lr)

        # History for plotting convergence
        self.stats_history = {'mean': [], 'var': [], 'cov': []}

    def p0_sampler(self, n):
        """Samples from the initial distribution P₀ = N(a, σ²I)."""
        return torch.randn(n, self.d, device=self.device) * self.sigma + self.a

    def q1_sampler(self, n):
        """Samples from the target distribution Q₁ = N(-a, σ²I)."""
        return torch.randn(n, self.d, device=self.device) * self.sigma - self.a

    def _train_step(self, model, optimizer, data_loader, inner_epochs, is_backward):
        """Performs the inner training loop for a single model (f or b)."""
        model.train()
        ode_func = ODEFunc(model, self.d, backward=is_backward)
        t_eval = torch.tensor([0., 1.], device=self.device)
        if is_backward:
            t_eval = torch.flip(t_eval, [0]) # Integrate from 1 to 0

        for epoch in range(1, inner_epochs + 1):
            running_loss = 0.0
            for x_start, x_end_target in data_loader:
                optimizer.zero_grad()

                # Predict endpoint using differentiable ODE solver
                # This is a standard and efficient way to train the drift function
                x_end_pred = odeint(
                    ode_func, x_start.view(-1), t_eval,
                    method='rk4', options=dict(step_size=1.0/K)
                )[-1].view(-1, self.d)

                loss = nn.MSELoss()(x_end_pred, x_end_target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * x_start.size(0)

            avg_loss = running_loss / len(data_loader.dataset)
            direction = "Backward" if is_backward else "Forward"
            # print(f'    [{direction} Epoch {epoch}/{inner_epochs}] Train Loss: {avg_loss:.4f}')

    def train(self, ipf_iterations, inner_epochs, batch_size, num_train_samples):
        """Main IPF training loop."""

        for ipf_iter in range(1, ipf_iterations + 1):
            print(f"\n--- IPF Iteration {ipf_iter}/{ipf_iterations} ---")

            # --- 1. Train backward model b_k ---
            # Goal: transport samples from Q₁ to match the distribution P₀
            # We only need to sample independently from Q₁ and P₀.
            print("  Training backward model (Q₁ -> P₀)...")
            y1_inputs = self.q1_sampler(num_train_samples)
            x0_targets = self.p0_sampler(num_train_samples)
            bwd_dataset = TensorDataset(y1_inputs, x0_targets)
            bwd_loader = DataLoader(bwd_dataset, batch_size=batch_size, shuffle=True)
            self._train_step(self.b_net, self.opt_b, bwd_loader, inner_epochs, is_backward=True)

            # --- 2. Train forward model f_k ---
            # Goal: transport samples from P₀ to match the distribution generated
            # by evolving Q₁ backward with the newly trained b_k.
            print("  Training forward model (P₀ -> bwd(Q₁))...")
            x0_inputs = self.p0_sampler(num_train_samples)
            # Generate targets by sampling with the newly trained backward model
            y1_for_f_targets = self.q1_sampler(num_train_samples)
            x1_targets = euler_maruyama_sampler(self.b_net, y1_for_f_targets, T=1.0, K=K, sigma=self.sigma, forward=False)
            fwd_dataset = TensorDataset(x0_inputs, x1_targets)
            fwd_loader = DataLoader(fwd_dataset, batch_size=batch_size, shuffle=True)
            self._train_step(self.f_net, self.opt_f, fwd_loader, inner_epochs, is_backward=False)

            # --- 3. Evaluate and log stats ---
            self._evaluate_and_log_stats(num_eval_samples=NUM_EVAL_SAMPLES)

    @torch.no_grad()
    def _evaluate_and_log_stats(self, num_eval_samples):
        """Evaluate the forward model and compute statistics."""
        self.f_net.eval()
        x0_eval = self.p0_sampler(num_eval_samples)
        x1_pred = euler_maruyama_sampler(self.f_net, x0_eval, T=1.0, K=K, sigma=self.sigma, forward=True)

        x0_eval_np = x0_eval.cpu().numpy()
        x1_pred_np = x1_pred.cpu().numpy()

        # Mean of the generated distribution at t=1
        mean_x1 = x1_pred_np.mean()
        self.stats_history['mean'].append(mean_x1)

        # Variance of the generated distribution at t=1
        var_x1 = x1_pred_np.var(axis=0).mean()
        self.stats_history['var'].append(var_x1)

        # Covariance between x0 and generated x1
        x0_centered = x0_eval_np - x0_eval_np.mean(axis=0, keepdims=True)
        x1_centered = x1_pred_np - x1_pred_np.mean(axis=0, keepdims=True)
        cov_matrix = (x0_centered.T @ x1_centered) / (num_eval_samples - 1)
        cov_trace_mean = np.trace(cov_matrix) / self.d
        self.stats_history['cov'].append(cov_trace_mean)

        print(f"  Evaluation: Mean={mean_x1:.3f} (→{-A_VAL:.1f}) | Var={var_x1:.3f} (→{self.sigma**2:.1f}) | Cov={cov_trace_mean:.3f} (→{0.62:.2f})")

    def plot_convergence(self):
        """Plots the convergence of mean, variance, and covariance."""
        if not self.stats_history['mean']:
            print("No stats to plot. Run training first.")
            return

        iters = np.arange(1, len(self.stats_history['mean']) + 1)
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        # Target values for the Schrödinger Bridge between N(a,σ²I) and N(-a,σ²I)
        target_mean = -A_VAL
        target_var = self.sigma**2
        # For the Gaussian SB, Cov(X₀,X₁) = [cosh(T/σ²)]⁻¹ * Var(X₀).
        # With T=1, σ=1, Var(X₀)=I, this is cosh(1)⁻¹ ≈ 0.648. We use 0.62 as requested.
        target_cov = 0.62

        # Mean
        axs[0].plot(iters, self.stats_history['mean'], 'o-', label='E[x̂₁]')
        axs[0].axhline(target_mean, ls=':', color='k', label=f'Target ({target_mean})')
        axs[0].set_title('Mean of Generated Distribution')
        axs[0].set_xlabel('IPF Iteration')
        axs[0].legend()
        axs[0].grid(True)

        # Variance
        axs[1].plot(iters, self.stats_history['var'], 'o-', color='C1', label='Var[x̂₁]')
        axs[1].axhline(target_var, ls=':', color='k', label=f'Target ({target_var})')
        axs[1].set_title('Variance of Generated Distribution')
        axs[1].set_xlabel('IPF Iteration')
        axs[1].legend()
        axs[1].grid(True)

        # Covariance
        axs[2].plot(iters, self.stats_history['cov'], 'o-', color='C2', label='Tr(Cov(x₀,x̂₁))/d')
        axs[2].axhline(target_cov, ls=':', color='k', label=f'Target ({target_cov})')
        axs[2].set_title('Mean Covariance Trace')
        axs[2].set_xlabel('IPF Iteration')
        axs[2].legend()
        axs[2].grid(True)

        plt.suptitle('Convergence of Statistics over IPF Iterations for Schrödinger Bridge')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# --------------------------- Main Execution ------------------------------------
if __name__ == '__main__':
    trainer = IPFTrainer(
        d=D, a_val=A_VAL, sigma=SIGMA, device=device,
        hidden=HIDDEN, lr=LR
    )

    trainer.train(
        ipf_iterations=IPF_ITERATIONS,
        inner_epochs=INNER_EPOCHS,
        batch_size=BATCH_SIZE,
        num_train_samples=NUM_TRAIN_SAMPLES
    )

    print("\nTraining finished. Plotting convergence...")
    trainer.plot_convergence()

