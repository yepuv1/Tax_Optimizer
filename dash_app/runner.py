"""Run-mode dispatcher.

Wraps the three workloads the Dash app exposes:

  * `single`            - a single deterministic `simulate(cfg, inputs)`.
  * `four_strategies`   - the four canonical strategies (S0/S1/S2/S3),
                          including the differential-evolution optimizer.
  * `four_plus_mc`      - same four strategies + Monte Carlo on the S3
                          winner (n_paths configurable).

The structure mirrors `tax_optimizer.report._strategy_results` so the
results consumed by the figures match what the existing report renderer
already knows how to summarize.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from tax_optimizer.config import Config
from tax_optimizer.inputs import Inputs
from tax_optimizer.metrics import summarize
from tax_optimizer.monte_carlo import MonteCarloResult, simulate_paths
from tax_optimizer.optimizer import optimize_household
from tax_optimizer.simulator import simulate


@dataclass
class StrategyResult:
    name: str
    cfg: Config
    inputs: Inputs
    df: pd.DataFrame
    summary: dict[str, Any]


@dataclass
class RunResult:
    mode: str
    strategies: dict[str, StrategyResult] = field(default_factory=dict)
    mc: MonteCarloResult | None = None
    winner_name: str | None = None
    elapsed_s: float = 0.0

    def winner(self) -> StrategyResult:
        if self.winner_name is None:
            # Pick the strategy with the highest terminal_after_tax.
            self.winner_name = max(
                self.strategies,
                key=lambda n: self.strategies[n].summary["terminal_after_tax"],
            )
        return self.strategies[self.winner_name]


# ---------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------


def _strategy_result(name: str, cfg: Config, inputs: Inputs) -> StrategyResult:
    df = simulate(cfg, inputs)
    return StrategyResult(
        name=name,
        cfg=cfg,
        inputs=inputs,
        df=df,
        summary=summarize(
            df,
            heir_marginal_rate=cfg.heir_marginal_rate,
            starting_balances=inputs.starting,
            inflation=cfg.inflation,
            retire_age=inputs.spouse_a_retire_age,
        ),
    )


def _build_four(cfg: Config, inputs: Inputs, *, seed: int) -> dict[str, StrategyResult]:
    s0 = (cfg, inputs)
    s1 = (
        cfg,
        replace(inputs, spouse_a_roth_401k_pct=1.0, spouse_b_roth_401k_pct=1.0),
    )
    s2 = (replace(cfg, roth_conversion_target_bracket=0.22), inputs)
    s3_cfg, s3_inputs, _x = optimize_household(
        cfg, inputs,
        objective="terminal",
        maxiter=20,
        popsize=10,
        seed=seed,
    )
    pairs = [
        ("S0_baseline", s0),
        ("S1_all_roth_401k", s1),
        ("S2_bracket_fill_22", s2),
        ("S3_optimized", (s3_cfg, s3_inputs)),
    ]
    return {name: _strategy_result(name, c, i) for name, (c, i) in pairs}


def run_scenario(
    cfg: Config,
    inputs: Inputs,
    *,
    mode: str = "single",
    n_paths: int = 200,
    seed: int = 0,
) -> RunResult:
    """Dispatch on `mode` and return a populated `RunResult`.

    `mode` must be one of "single", "four_strategies", "four_plus_mc".
    Anything else raises ValueError.
    """
    import time

    t0 = time.perf_counter()

    if mode == "single":
        sr = _strategy_result("current", cfg, inputs)
        return RunResult(
            mode=mode,
            strategies={"current": sr},
            winner_name="current",
            elapsed_s=time.perf_counter() - t0,
        )

    if mode in ("four_strategies", "four_plus_mc"):
        strategies = _build_four(cfg, inputs, seed=seed)
        winner_name = max(
            strategies, key=lambda n: strategies[n].summary["terminal_after_tax"]
        )
        mc: MonteCarloResult | None = None
        if mode == "four_plus_mc":
            w = strategies[winner_name]
            mc = simulate_paths(
                w.cfg, w.inputs, n_paths=n_paths, seed=seed, keep_paths=True
            )
        return RunResult(
            mode=mode,
            strategies=strategies,
            mc=mc,
            winner_name=winner_name,
            elapsed_s=time.perf_counter() - t0,
        )

    raise ValueError(
        f"Unknown run mode {mode!r}. "
        f"Expected 'single', 'four_strategies', or 'four_plus_mc'."
    )


# ---------------------------------------------------------------------
# Serialization for dcc.Store
# ---------------------------------------------------------------------

def serialize_run_result(rr: RunResult) -> dict[str, Any]:
    """Serialize a RunResult into JSON-safe primitives.

    Each strategy DataFrame is split into orient='split' form so it can
    be reconstructed with `pd.DataFrame(**payload)`. The MonteCarloResult
    is reduced to the percentile/aggregate stats the figures need - the
    full `mc.paths` array isn't shipped to the browser (potentially
    100s of MB at high n_paths). We do keep one fan-chart aggregation
    (P10/P50/P90 of liquid NW per year) computed eagerly here.
    """
    strategies_out: dict[str, Any] = {}
    for name, s in rr.strategies.items():
        df_payload = {
            "columns": list(s.df.columns),
            "index": list(s.df.index),
            "data": s.df.astype(object).where(s.df.notna(), None).values.tolist(),
        }
        strategies_out[name] = {
            "name": s.name,
            "df": df_payload,
            "summary": _jsonable(s.summary),
            "cfg_summary": _cfg_summary(s.cfg, s.inputs),
        }

    mc_payload: dict[str, Any] | None = None
    if rr.mc is not None:
        mc = rr.mc
        terminals = np.asarray(mc.terminals, dtype=float)
        mc_payload = {
            "n_paths": int(mc.n_paths),
            "terminals": terminals.tolist(),
            "lifetime_taxes": np.asarray(mc.lifetime_taxes).tolist(),
            "lifetime_irmaas": np.asarray(mc.lifetime_irmaas).tolist(),
            "ruin_year_offsets": np.asarray(mc.ruin_year_offsets).tolist(),
            "prob_success": float(mc.prob_success()),
            "cvar_terminal": float(mc.cvar_terminal(0.10)),
            "percentiles": {
                "p10": float(np.percentile(terminals, 10)),
                "p50": float(np.percentile(terminals, 50)),
                "p90": float(np.percentile(terminals, 90)),
            },
            "fan": _build_fan(mc),
        }

    return {
        "mode": rr.mode,
        "strategies": strategies_out,
        "mc": mc_payload,
        "winner_name": rr.winner_name,
        "elapsed_s": rr.elapsed_s,
    }


def deserialize_strategy_df(payload: dict[str, Any]) -> pd.DataFrame:
    """Inverse of the orient='split' shape emitted by `serialize_run_result`."""
    return pd.DataFrame(
        data=payload["data"], index=payload["index"], columns=payload["columns"]
    )


def _build_fan(mc: MonteCarloResult) -> dict[str, list[float]] | None:
    """Per-year P10/P50/P90 of liquid NW across MC paths."""
    if not mc.paths:
        return None
    n_years = max(len(df) for df in mc.paths)
    arr = np.full((len(mc.paths), n_years), np.nan, dtype=float)
    for i, df in enumerate(mc.paths):
        liquid = (
            df["pretax_balance"].to_numpy()
            + df["roth_balance"].to_numpy()
            + df["taxable_balance"].to_numpy()
            + df["hsa_balance"].to_numpy()
        )
        arr[i, : len(liquid)] = liquid
    p10 = np.nanpercentile(arr, 10, axis=0)
    p50 = np.nanpercentile(arr, 50, axis=0)
    p90 = np.nanpercentile(arr, 90, axis=0)
    return {
        "year_offset": list(range(n_years)),
        "p10": p10.tolist(),
        "p50": p50.tolist(),
        "p90": p90.tolist(),
    }


def _cfg_summary(cfg: Config, inputs: Inputs) -> dict[str, Any]:
    """A few key knobs the Strategies tab surfaces in its callout.

    Also carries `heir_marginal_rate` and `inflation` so the
    Overview-tab growth chart can derive its after-tax NW series
    from a deserialized run payload without a fresh Config.
    """
    return {
        "spouse_a_roth_401k_pct": float(inputs.spouse_a_roth_401k_pct),
        "spouse_b_roth_401k_pct": float(inputs.spouse_b_roth_401k_pct),
        "roth_conversion_target_bracket": float(cfg.roth_conversion_target_bracket),
        "spouse_a_after_tax_401k_pct": float(inputs.spouse_a_after_tax_401k_pct),
        "spouse_b_after_tax_401k_pct": float(inputs.spouse_b_after_tax_401k_pct),
        "ss_start_age_a": int(inputs.ss.effective_start_age_a),
        "ss_start_age_b": int(inputs.ss.effective_start_age_b),
        "heir_marginal_rate": float(cfg.heir_marginal_rate),
        "inflation": float(cfg.inflation),
    }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
