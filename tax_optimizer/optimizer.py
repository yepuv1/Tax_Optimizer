"""S3 strategy optimizer.

Decision vector x = [spouse_a_roth_pct, spouse_b_roth_pct, conv_bracket_idx]
where the conversion bracket index maps to {0%, 12%, 22%, 24%, 32%}.

Three objective modes are supported:

  * `'terminal'`  — maximize deterministic terminal after-tax NW. The
                    classic v1 objective. Uses a single `simulate(cfg)`
                    call per evaluation, so it's fast.
  * `'cvar'`      — Monte-Carlo CVaR(α) of terminal NW. Maximizes the
                    expected terminal in the worst α fraction of paths.
                    Use this when the market is stochastic.
  * `'p_success'` — Monte-Carlo probability of solvency. Maximizes the
                    fraction of paths that never run out of liquid
                    assets. Use this for retirees whose primary goal
                    is "don't outlive my money".

We use `scipy.optimize.differential_evolution` (population-based,
derivative-free) because the objective is non-smooth at every IRMAA
cliff and at every bracket-index `round()` step.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np
from scipy.optimize import differential_evolution

from .config import Config
from .inputs import Inputs
from .metrics import terminal_after_tax_nw
from .monte_carlo import simulate_paths
from .simulator import simulate

ObjectiveType = Literal["terminal", "cvar", "p_success"]

BRACKET_CHOICES: list[float] = [0.0, 0.12, 0.22, 0.24, 0.32]


def x_to_cfg(x: np.ndarray, base_cfg: Config) -> Config:
    a_roth = float(np.clip(x[0], 0.0, 1.0))
    b_roth = float(np.clip(x[1], 0.0, 1.0))
    idx = int(np.clip(round(x[2]), 0, len(BRACKET_CHOICES) - 1))
    return replace(
        base_cfg,
        spouse_a_roth_401k_pct=a_roth,
        spouse_b_roth_401k_pct=b_roth,
        roth_conversion_target_bracket=BRACKET_CHOICES[idx],
    )


def _terminal_objective(
    x: np.ndarray, base_cfg: Config, inputs: Inputs
) -> float:
    cfg = x_to_cfg(x, base_cfg)
    df = simulate(cfg, inputs)
    liquid = df["pretax_balance"] + df["roth_balance"] + df["taxable_balance"]
    floor = df["spending_need"]
    deficit = float((floor - liquid).clip(lower=0).sum())
    irmaa_total = float(df["irmaa"].sum())
    terminal = terminal_after_tax_nw(df)
    return -(terminal - 1e3 * deficit - 0.5 * irmaa_total)


def _cvar_objective(
    x: np.ndarray, base_cfg: Config, inputs: Inputs, n_paths: int, alpha: float
) -> float:
    cfg = x_to_cfg(x, base_cfg)
    mc = simulate_paths(cfg, inputs, n_paths=n_paths, seed=42, keep_paths=False)
    cvar = mc.cvar_terminal(alpha)
    # Apply same deficit + IRMAA-soft penalty as the terminal mode, but
    # using mean over paths.
    irmaa_pen = 0.5 * float(np.mean(mc.lifetime_irmaas))
    p_success = mc.prob_success()
    deficit_pen = (1 - p_success) * 1e6  # heavy penalty for ruin paths
    return -(cvar - irmaa_pen - deficit_pen)


def _p_success_objective(
    x: np.ndarray, base_cfg: Config, inputs: Inputs, n_paths: int
) -> float:
    cfg = x_to_cfg(x, base_cfg)
    mc = simulate_paths(cfg, inputs, n_paths=n_paths, seed=42, keep_paths=False)
    # Maximize prob_success; tiebreak on median terminal.
    return -(mc.prob_success() * 1e8 + float(np.median(mc.terminals)))


def make_objective(
    base_cfg: Config,
    inputs: Inputs,
    *,
    objective: ObjectiveType = "terminal",
    n_paths: int = 200,
    cvar_alpha: float = 0.10,
):
    if objective == "terminal":
        return lambda x: _terminal_objective(x, base_cfg, inputs)
    if objective == "cvar":
        return lambda x: _cvar_objective(x, base_cfg, inputs, n_paths, cvar_alpha)
    if objective == "p_success":
        return lambda x: _p_success_objective(x, base_cfg, inputs, n_paths)
    raise ValueError(f"Unknown objective {objective!r}")


def optimize_s3(
    base_cfg: Config,
    inputs: Inputs | None = None,
    *,
    objective: ObjectiveType = "terminal",
    n_paths: int = 200,
    cvar_alpha: float = 0.10,
    seed: int = 0,
    maxiter: int = 40,
    popsize: int = 15,
) -> tuple[Config, np.ndarray]:
    """Run the S3 optimizer and return (best_cfg, x_opt).

    For `objective='terminal'`, runtime is ~10s on a 41-year horizon.
    For Monte Carlo objectives, runtime is ~`maxiter * popsize *
    n_paths * 0.05s` per evaluation; with the defaults that's ~1-2 min.
    Drop `n_paths` to 100 or `popsize` to 10 to halve it.
    """
    if inputs is None:
        inputs = Inputs()
    obj_fn = make_objective(
        base_cfg, inputs, objective=objective, n_paths=n_paths, cvar_alpha=cvar_alpha
    )

    # Coarse grid as a sanity reference (cheap; only the deterministic
    # objective uses it as a fallback).
    grid_results = []
    for v in np.linspace(0, 1, 4):
        for b in np.linspace(0, 1, 4):
            for c in range(len(BRACKET_CHOICES)):
                x0 = np.array([v, b, c], dtype=float)
                try:
                    grid_results.append((obj_fn(x0), x0))
                except Exception:
                    pass
    grid_results.sort(key=lambda t: t[0])
    best_grid = grid_results[0] if grid_results else (None, np.array([0.0, 0.0, 0.0]))

    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, len(BRACKET_CHOICES) - 1)]
    res = differential_evolution(
        obj_fn,
        bounds=bounds,
        polish=False,
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-3,
        seed=seed,
        init="sobol",
        workers=1,
    )
    if best_grid[0] is not None and -res.fun <= -best_grid[0]:
        x_opt = best_grid[1]
    else:
        x_opt = res.x
    return x_to_cfg(x_opt, base_cfg), x_opt
