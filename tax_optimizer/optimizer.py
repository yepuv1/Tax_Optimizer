"""S3 strategy optimizer.

The decision vector is **dynamically sized** based on which Tier B/C
levers are enabled in the input scenario:

  * Always present:
      - `spouse_a_roth_401k_pct` (continuous, 0..1)
      - `spouse_b_roth_401k_pct` (continuous, 0..1)
      - `roth_conversion_target_bracket` (discrete, 5-bin index)

  * Added when `inputs.spouse_*_mega_backdoor_enabled=True`  (TC-15):
      - `spouse_*_after_tax_401k_pct` (continuous, 0..1)

  * Added when `cfg.optimize_ss_claim_age=True`  (TC-16):
      - `ss_claim_age_a` (discrete index into {62, 65, 67, 70})
      - `ss_claim_age_b` (discrete index into {62, 65, 67, 70})

`_build_decision_vector_meta(cfg, inputs)` returns the active axis list
with name + bound + apply-fn so the optimizer, the bounds list, and the
test suite all stay in lockstep. Any new axis only needs an entry in
that list.

For Monte-Carlo objectives, `mc_seed` (TC-17) is threaded all the way
to `simulate_paths` so users running CVaR / p_success can pin
reproducibility (or sweep seeds to detect overfitting to a single MC
draw).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Literal

import numpy as np
from scipy.optimize import differential_evolution

from .config import Config
from .inputs import Inputs
from .metrics import terminal_after_tax_nw
from .monte_carlo import simulate_paths
from .simulator import simulate

ObjectiveType = Literal["terminal", "cvar", "p_success"]

BRACKET_CHOICES: list[float] = [0.0, 0.12, 0.22, 0.24, 0.32]
SS_CLAIM_AGE_CHOICES: list[int] = [62, 65, 67, 70]


# ---------------------------------------------------------------------
# Decision-vector metadata
# ---------------------------------------------------------------------


def _build_decision_vector_meta(cfg: Config, inputs: Inputs) -> list[dict]:
    """Return one dict per active decision axis.

    Each dict has:
      * `name`     — short human label (used by tests and report)
      * `bounds`   — (lo, hi) tuple
      * `apply`    — fn (cfg, inputs, x) → (cfg, inputs) that sets
                     the relevant field after coercing `x`.
    """
    meta: list[dict] = []

    # Axis 1+2: per-spouse Roth-401(k) percentage.
    def _apply_roth_a(cfg: Config, inputs: Inputs, x: float):
        return cfg, replace(inputs, spouse_a_roth_401k_pct=float(np.clip(x, 0, 1)))

    def _apply_roth_b(cfg: Config, inputs: Inputs, x: float):
        return cfg, replace(inputs, spouse_b_roth_401k_pct=float(np.clip(x, 0, 1)))

    meta.append({"name": "roth_401k_pct_a", "bounds": (0.0, 1.0), "apply": _apply_roth_a})
    meta.append({"name": "roth_401k_pct_b", "bounds": (0.0, 1.0), "apply": _apply_roth_b})

    # Axis 3: Roth conversion bracket index.
    def _apply_conv(cfg: Config, inputs: Inputs, x: float):
        idx = int(np.clip(round(x), 0, len(BRACKET_CHOICES) - 1))
        return replace(cfg, roth_conversion_target_bracket=BRACKET_CHOICES[idx]), inputs

    meta.append({
        "name": "conv_bracket_idx",
        "bounds": (0.0, float(len(BRACKET_CHOICES) - 1)),
        "apply": _apply_conv,
    })

    # TC-15 — mega-backdoor after-tax 401(k) percentage. Only added if
    # the user has enabled the path on the input scenario; otherwise
    # the optimizer can't legally search this axis.
    if inputs.spouse_a_mega_backdoor_enabled:
        def _apply_mega_a(cfg: Config, inputs: Inputs, x: float):
            return cfg, replace(inputs, spouse_a_after_tax_401k_pct=float(np.clip(x, 0, 1)))

        meta.append({
            "name": "mega_backdoor_pct_a",
            "bounds": (0.0, 1.0),
            "apply": _apply_mega_a,
        })
    if inputs.spouse_b_mega_backdoor_enabled:
        def _apply_mega_b(cfg: Config, inputs: Inputs, x: float):
            return cfg, replace(inputs, spouse_b_after_tax_401k_pct=float(np.clip(x, 0, 1)))

        meta.append({
            "name": "mega_backdoor_pct_b",
            "bounds": (0.0, 1.0),
            "apply": _apply_mega_b,
        })

    # TC-16 — per-spouse SS claim age (discrete grid).
    if cfg.optimize_ss_claim_age:
        def _apply_ss_a(cfg: Config, inputs: Inputs, x: float):
            idx = int(np.clip(round(x), 0, len(SS_CLAIM_AGE_CHOICES) - 1))
            new_ss = replace(inputs.ss, start_age_a=SS_CLAIM_AGE_CHOICES[idx])
            return cfg, replace(inputs, ss=new_ss)

        def _apply_ss_b(cfg: Config, inputs: Inputs, x: float):
            idx = int(np.clip(round(x), 0, len(SS_CLAIM_AGE_CHOICES) - 1))
            new_ss = replace(inputs.ss, start_age_b=SS_CLAIM_AGE_CHOICES[idx])
            return cfg, replace(inputs, ss=new_ss)

        meta.append({
            "name": "ss_claim_age_a",
            "bounds": (0.0, float(len(SS_CLAIM_AGE_CHOICES) - 1)),
            "apply": _apply_ss_a,
        })
        meta.append({
            "name": "ss_claim_age_b",
            "bounds": (0.0, float(len(SS_CLAIM_AGE_CHOICES) - 1)),
            "apply": _apply_ss_b,
        })

    return meta


def x_to_overrides(
    x: np.ndarray, base_cfg: Config, base_inputs: Inputs
) -> tuple[Config, Inputs]:
    """Apply decision vector `x` to (cfg, inputs) and return the patched pair.

    `x` shape must match `len(_build_decision_vector_meta(base_cfg, base_inputs))`.
    """
    meta = _build_decision_vector_meta(base_cfg, base_inputs)
    if len(x) < len(meta):
        # Defensive: pad with zeros (caller passed in legacy-shaped x).
        x = np.concatenate([x, np.zeros(len(meta) - len(x))])
    cfg = base_cfg
    inputs = base_inputs
    for value, axis in zip(x, meta):
        cfg, inputs = axis["apply"](cfg, inputs, float(value))
    return cfg, inputs


# ---------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------


def _terminal_objective(
    x: np.ndarray, base_cfg: Config, base_inputs: Inputs
) -> float:
    cfg, inputs = x_to_overrides(x, base_cfg, base_inputs)
    df = simulate(cfg, inputs)
    liquid = df["pretax_balance"] + df["roth_balance"] + df["taxable_balance"]
    floor = df["spending_need"]
    deficit = float((floor - liquid).clip(lower=0).sum())
    irmaa_total = float(df["irmaa"].sum())
    terminal = terminal_after_tax_nw(df, heir_marginal_rate=cfg.heir_marginal_rate)
    return -(terminal - 1e3 * deficit - 0.5 * irmaa_total)


def _cvar_objective(
    x: np.ndarray,
    base_cfg: Config,
    base_inputs: Inputs,
    n_paths: int,
    alpha: float,
    mc_seed: int,
) -> float:
    cfg, inputs = x_to_overrides(x, base_cfg, base_inputs)
    mc = simulate_paths(cfg, inputs, n_paths=n_paths, seed=mc_seed, keep_paths=False)
    cvar = mc.cvar_terminal(alpha)
    irmaa_pen = 0.5 * float(np.mean(mc.lifetime_irmaas))
    p_success = mc.prob_success()
    deficit_pen = (1 - p_success) * 1e6
    return -(cvar - irmaa_pen - deficit_pen)


def _p_success_objective(
    x: np.ndarray,
    base_cfg: Config,
    base_inputs: Inputs,
    n_paths: int,
    mc_seed: int,
) -> float:
    cfg, inputs = x_to_overrides(x, base_cfg, base_inputs)
    mc = simulate_paths(cfg, inputs, n_paths=n_paths, seed=mc_seed, keep_paths=False)
    return -(mc.prob_success() * 1e8 + float(np.median(mc.terminals)))


def make_objective(
    base_cfg: Config,
    base_inputs: Inputs,
    *,
    objective: ObjectiveType = "terminal",
    n_paths: int = 200,
    cvar_alpha: float = 0.10,
    mc_seed: int = 42,
) -> Callable[[np.ndarray], float]:
    """Build the objective fn used by `differential_evolution`.

    `mc_seed` (TC-17): controls the deterministic MC draw used by both
    `cvar` and `p_success` objectives. Pin a single value to compare
    optimizer runs apples-to-apples; sweep it to detect overfitting
    to one historical-style sequence.
    """
    if objective == "terminal":
        return lambda x: _terminal_objective(x, base_cfg, base_inputs)
    if objective == "cvar":
        return lambda x: _cvar_objective(
            x, base_cfg, base_inputs, n_paths, cvar_alpha, mc_seed
        )
    if objective == "p_success":
        return lambda x: _p_success_objective(
            x, base_cfg, base_inputs, n_paths, mc_seed
        )
    raise ValueError(f"Unknown objective {objective!r}")


# ---------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------


def optimize_household(
    base_cfg: Config,
    inputs: Inputs | None = None,
    *,
    objective: ObjectiveType = "terminal",
    n_paths: int = 200,
    cvar_alpha: float = 0.10,
    mc_seed: int = 42,
    seed: int = 0,
    maxiter: int = 40,
    popsize: int = 15,
) -> tuple[Config, Inputs, np.ndarray]:
    """Run the optimizer with the active Tier-C decision vector and
    return `(best_cfg, best_inputs, x_opt)`.

    `mc_seed` (TC-17) — the MC draw seed for stochastic objectives;
    independent of `seed` (the differential-evolution PRNG seed).

    `seed` — controls the population-init sampler in DE; doesn't
    affect the simulation paths themselves.
    """
    if inputs is None:
        inputs = Inputs()

    obj_fn = make_objective(
        base_cfg,
        inputs,
        objective=objective,
        n_paths=n_paths,
        cvar_alpha=cvar_alpha,
        mc_seed=mc_seed,
    )

    meta = _build_decision_vector_meta(base_cfg, inputs)
    bounds = [axis["bounds"] for axis in meta]

    # Coarse grid initialization for the first three axes (the Tier-A
    # baseline). Higher-dim runs skip the grid and rely on DE init.
    if len(meta) <= 3:
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
        best_grid = grid_results[0] if grid_results else (None, np.zeros(len(meta)))
    else:
        best_grid = (None, np.zeros(len(meta)))

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
    best_cfg, best_inputs = x_to_overrides(x_opt, base_cfg, inputs)
    return best_cfg, best_inputs, x_opt


# Backward-compat alias. New code should use `optimize_household`.
optimize_s3 = optimize_household
