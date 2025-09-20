import argparse
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.integrate import solve_ivp

import pysindy as ps
import warnings

warnings.filterwarnings(
    "ignore",
    message="Control variables u were ignored because control variables were not used when the model was fit",
    category=UserWarning,
)

from gaussian_transport import (
    GaussianSampler,
    GaussianTransportConfig,
    compute_stats,
    plot_transport_stats,
    set_seed,
)


TimeFeature = Callable[[np.ndarray], np.ndarray]


def _as_column(t: np.ndarray) -> np.ndarray:
    if t.ndim == 1:
        return t.reshape(-1, 1)
    if t.ndim == 2 and t.shape[1] == 1:
        return t
    return t.reshape(-1, 1)


def evaluate_time_functions(t_values: np.ndarray, functions: Sequence[TimeFeature]) -> np.ndarray:
    cols = [np.asarray(func(t_values)).reshape(-1) for func in functions]
    return np.column_stack(cols)


def build_feature_library(library: str) -> Tuple[Any, Optional[List[TimeFeature]]]:
    if library == "constant":
        constant_only = ps.PolynomialLibrary(degree=0, include_bias=True)
        return constant_only, None

    if library != "rich":
        raise ValueError(f"Unknown library type: {library}")

    def ones(t: np.ndarray) -> np.ndarray:
        col = _as_column(t).reshape(-1)
        return np.ones_like(col)

    def identity(t: np.ndarray) -> np.ndarray:
        return _as_column(t).reshape(-1)

    def squared(t: np.ndarray) -> np.ndarray:
        col = _as_column(t).reshape(-1)
        return col * col

    def one_minus(t: np.ndarray) -> np.ndarray:
        col = 1.0 - _as_column(t).reshape(-1)
        return col

    def one_minus_squared(t: np.ndarray) -> np.ndarray:
        base = 1.0 - _as_column(t).reshape(-1)
        return base * base

    def sin_pi(t: np.ndarray) -> np.ndarray:
        col = _as_column(t).reshape(-1)
        return np.sin(np.pi * col)

    def cos_pi(t: np.ndarray) -> np.ndarray:
        col = _as_column(t).reshape(-1)
        return np.cos(np.pi * col)

    time_functions: List[TimeFeature] = [ones, identity, squared, one_minus, one_minus_squared, sin_pi, cos_pi]
    time_names = [
        lambda _: "1",
        lambda _: "t",
        lambda _: "t^2",
        lambda _: "1-t",
        lambda _: "(1-t)^2",
        lambda _: "sin(pi t)",
        lambda _: "cos(pi t)",
    ]

    time_library = ps.CustomLibrary(time_functions, function_names=time_names)
    state_library = ps.PolynomialLibrary(degree=3, include_bias=True)
    generalized = ps.GeneralizedLibrary([state_library, time_library])
    return generalized, time_functions


@dataclass
class SINDyFMConfig(GaussianTransportConfig):
    n_trajectories: int = 1024
    library: str = "rich"
    optimizer_alpha: float = 1e-3
    optimizer_threshold: float = 1e-4
    optimizer_max_iter: int = 20
    n_validation: int = 65536


def generate_flow_matching_data(cfg: SINDyFMConfig, sampler: GaussianSampler) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0 = sampler.sample_x0(cfg.n_trajectories)
    x1 = sampler.sample_x1(cfg.n_trajectories)
    delta = x1 - x0

    states: List[np.ndarray] = []
    derivatives: List[np.ndarray] = []
    times: List[np.ndarray] = []

    for idx in range(cfg.n_trajectories):
        t_vals = torch.rand(cfg.time_samples, 1, device=sampler.device)
        t_vals, _ = torch.sort(t_vals, dim=0)
        x_traj = x0[idx : idx + 1] + t_vals * delta[idx : idx + 1]
        states.append(x_traj.cpu().numpy())
        derivatives.append(delta[idx : idx + 1].repeat(cfg.time_samples, 1).cpu().numpy())
        times.append(t_vals.cpu().numpy())

    X = np.vstack(states)
    X_dot = np.vstack(derivatives)
    T = np.vstack(times)
    return X, X_dot, T


def simulate_with_sindy(
    model: ps.SINDy,
    x0: np.ndarray,
    cfg: SINDyFMConfig,
    time_functions: Optional[Sequence[TimeFeature]],
) -> np.ndarray:
    t_eval = np.linspace(0.0, 1.0, cfg.K + 1)

    def rhs(t: float, x: np.ndarray) -> np.ndarray:
        x_row = x.reshape(1, -1)
        if time_functions is not None:
            u_t = evaluate_time_functions(np.array([[t]], dtype=float), time_functions)
            deriv = model.predict(x_row, u_t)
        else:
            deriv = model.predict(x_row)
        return deriv.ravel()

    sol = solve_ivp(rhs, (0.0, 1.0), x0, t_eval=t_eval, rtol=1e-6, atol=1e-8)
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")
    return sol.y[:, -1]


def train_sindy(cfg: SINDyFMConfig):
    sampler = GaussianSampler(cfg.dim, cfg.a, cfg.device)
    X, X_dot, T = generate_flow_matching_data(cfg, sampler)

    feature_library, time_functions = build_feature_library(cfg.library)
    optimizer = ps.STLSQ(
        alpha=cfg.optimizer_alpha,
        threshold=cfg.optimizer_threshold,
        max_iter=cfg.optimizer_max_iter,
    )
    model = ps.SINDy(
        optimizer=optimizer,
        feature_library=feature_library,
    )

    if time_functions is not None:
        U = evaluate_time_functions(T, time_functions)
        model.fit(X, 1.0, x_dot=X_dot, u=U)
        residual = model.predict(X, U) - X_dot
        score = model.score(X, 1.0, x_dot=X_dot, u=U)
    else:
        model.fit(X, 1.0, x_dot=X_dot)
        residual = model.predict(X) - X_dot
        score = model.score(X, 1.0, x_dot=X_dot)

    mse = float(np.mean(np.sum(residual * residual, axis=1)))
    return model, sampler, time_functions, mse, score


def validate_sindy(
    model: ps.SINDy,
    sampler: GaussianSampler,
    cfg: SINDyFMConfig,
    time_functions: Optional[Sequence[TimeFeature]],
) -> Tuple[float, float, float]:
    with torch.no_grad():
        x0 = sampler.sample_x0(cfg.n_validation)
    x0_np = x0.cpu().numpy()

    x1_preds = []
    for row in x0_np:
        x1 = simulate_with_sindy(model, row, cfg, time_functions)
        x1_preds.append(x1)
    x1_hat = np.vstack(x1_preds)

    mean_val, var_val, cov_val = compute_stats(x0_np, x1_hat)
    return mean_val, var_val, cov_val


def main():
    parser = argparse.ArgumentParser(description="Flow-matching SINDy experiment for Gaussian transport")
    parser.add_argument("--library", choices=["constant", "rich"], default="rich")
    parser.add_argument("--trajectories", type=int, default=1024)
    parser.add_argument("--time-samples", type=int, default=3)
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--a", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--validate", type=int, default=65536, help="number of samples for validation")
    args = parser.parse_args()

    set_seed(args.seed)

    cfg = SINDyFMConfig(
        dim=args.dim,
        a=args.a,
        sigma=args.sigma,
        K=args.K,
        time_samples=args.time_samples,
        n_trajectories=args.trajectories,
        library=args.library,
        optimizer_alpha=args.alpha,
        optimizer_threshold=args.threshold,
        optimizer_max_iter=args.max_iter,
        n_validation=args.validate,
    )

    model, sampler, time_functions, mse, score = train_sindy(cfg)
    print("Learned dynamics:")
    model.print()
    print(f"\nTraining residual MSE: {mse:.6e}")
    print(f"Training R^2 score: {score:.6f}")

    mean_val, var_val, cov_val = validate_sindy(model, sampler, cfg, time_functions)
    print("\nValidation stats (SINDy ODE integration):")
    print(f"  mean[x1_hat] = {mean_val:.4f} (target {cfg.targets.mean})")
    print(f"  var[x1_hat]  = {var_val:.4f} (target {cfg.targets.variance})")
    print(f"  cov_diagmean = {cov_val:.4f} (target {cfg.targets.covariance})")

    if not args.no_plot:
        plot_transport_stats(
            means=[mean_val],
            variances=[var_val],
            covariances=[cov_val],
            targets=cfg.targets,
            epochs=[1],
        )


if __name__ == "__main__":
    main()
