"""Monte Carlo wrapper.

`simulate_paths(cfg, inputs, n_paths)` runs the deterministic simulator
N times with different RNG seeds (so a stochastic `MarketModel` produces
N different return paths) and aggregates the results into a
`MonteCarloResult` with percentile and probability-of-success summaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .inputs import Inputs
from .metrics import lifetime_irmaa_npv, lifetime_tax_npv, terminal_after_tax_nw
from .simulator import simulate


@dataclass
class MonteCarloResult:
    """Per-path frames and aggregate statistics for a Monte Carlo run."""

    paths: list[pd.DataFrame]
    terminals: np.ndarray  # one per path
    lifetime_taxes: np.ndarray
    lifetime_irmaas: np.ndarray
    ruin_year_offsets: np.ndarray  # year_offset of first ruin or -1
    cfg: Config

    @property
    def n_paths(self) -> int:
        return int(len(self.terminals))

    def prob_success(self) -> float:
        """Fraction of paths that never ran out of liquid assets.

        A path is "successful" when its `ruin_year_offset` is -1
        (computed inside `simulate_paths` from each per-year frame
        before the frame is optionally discarded). This works whether
        or not `keep_paths=True`.
        """
        if len(self.ruin_year_offsets) == 0:
            return 0.0
        return float(np.mean(self.ruin_year_offsets < 0))

    def percentiles(
        self, *, levels: tuple[int, ...] = (5, 25, 50, 75, 95)
    ) -> dict[str, dict[int, float]]:
        return {
            "terminal": {p: float(np.percentile(self.terminals, p)) for p in levels},
            "lifetime_tax": {
                p: float(np.percentile(self.lifetime_taxes, p)) for p in levels
            },
            "lifetime_irmaa": {
                p: float(np.percentile(self.lifetime_irmaas, p)) for p in levels
            },
        }

    def cvar_terminal(self, alpha: float = 0.10) -> float:
        """Expected terminal NW over the worst `alpha` fraction of paths.
        Lower = riskier. The optimizer's `cvar` mode maximizes this."""
        cutoff = np.percentile(self.terminals, alpha * 100)
        worst = self.terminals[self.terminals <= cutoff]
        return float(np.mean(worst)) if len(worst) else 0.0

    def summary(self) -> dict:
        pct = self.percentiles()
        return {
            "n_paths": self.n_paths,
            "prob_success": self.prob_success(),
            "terminal_p5": pct["terminal"][5],
            "terminal_p50": pct["terminal"][50],
            "terminal_p95": pct["terminal"][95],
            "cvar_terminal_p10": self.cvar_terminal(0.10),
            "cvar_terminal_p20": self.cvar_terminal(0.20),
            "lifetime_tax_p50": pct["lifetime_tax"][50],
            "lifetime_irmaa_p50": pct["lifetime_irmaa"][50],
            "median_ruin_year_offset": (
                float(np.median(self.ruin_year_offsets[self.ruin_year_offsets >= 0]))
                if (self.ruin_year_offsets >= 0).any()
                else -1.0
            ),
        }


def _ruin_year_offset(df: pd.DataFrame) -> int:
    """First year_offset where the plan failed to fund the year.

    A path is "ruined" the first year either:
      (a) `unfunded` > 0 -- the deficit-cascade exhausted every bucket
          and a genuine cash gap remained, OR
      (b) liquid < spending_need -- the heuristic that pre-existed the
          unfunded tracking; kept as a defense-in-depth check for paths
          that ran clean cascades but ended on fumes.
    """
    if "unfunded" in df.columns:
        bad_unfunded = (df["unfunded"] > 0).to_numpy()
    else:  # pragma: no cover - back-compat for frames without the column
        bad_unfunded = np.zeros(len(df), dtype=bool)
    liquid = df["pretax_balance"] + df["roth_balance"] + df["taxable_balance"]
    bad_liquid = (liquid < df["spending_need"]).to_numpy()
    bad = bad_unfunded | bad_liquid
    if not bad.any():
        return -1
    return int(np.argmax(bad))


def simulate_paths(
    cfg: Config,
    inputs: Inputs | None = None,
    *,
    n_paths: int = 500,
    seed: int = 0,
    keep_paths: bool = True,
) -> MonteCarloResult:
    """Run `n_paths` independent simulator paths.

    With a `DeterministicModel` market every path is identical (and the
    Monte Carlo summary collapses to point estimates). The intended use
    is with `LognormalModel` or `BootstrapModel` so each path samples a
    different return sequence. RNG is seeded from `seed` plus the path
    index so results are reproducible.

    `keep_paths=False` discards per-year DataFrames and returns only the
    aggregate statistics — useful when running thousands of paths.
    """
    rng_master = np.random.default_rng(seed)
    seeds = rng_master.integers(0, 2**31 - 1, size=n_paths)

    paths: list[pd.DataFrame] = []
    terminals = np.zeros(n_paths)
    taxes = np.zeros(n_paths)
    irmaas = np.zeros(n_paths)
    ruins = np.zeros(n_paths, dtype=int)

    for i, s in enumerate(seeds):
        rng = np.random.default_rng(int(s))
        df = simulate(cfg, inputs, rng=rng)
        terminals[i] = terminal_after_tax_nw(df)
        taxes[i] = lifetime_tax_npv(df)
        irmaas[i] = lifetime_irmaa_npv(df)
        ruins[i] = _ruin_year_offset(df)
        if keep_paths:
            paths.append(df)

    return MonteCarloResult(
        paths=paths,
        terminals=terminals,
        lifetime_taxes=taxes,
        lifetime_irmaas=irmaas,
        ruin_year_offsets=ruins,
        cfg=cfg,
    )
