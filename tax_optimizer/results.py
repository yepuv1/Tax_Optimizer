"""Per-strategy result container.

The deterministic strategy comparison (S0/S1/S2/S3) used to be stored
as a `dict[str, tuple[Config, DataFrame, dict]]`. That triple shape
spread implicit positional dependencies across the report renderer,
the recommendations engine, and the CLI. Whenever we needed an extra
piece of context per strategy (e.g. now: the per-strategy `Inputs`,
since the optimizer mutates contribution rates), every consumer broke.

`StrategyResult` is a named alternative. Adding new fields to it is a
backward-compatible operation: existing callers that read `r.cfg`,
`r.inputs`, `r.df`, `r.summary` keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from .config import Config
    from .inputs import Inputs


@dataclass
class StrategyResult:
    """A single named strategy's full outcome.

    Attributes:
        cfg:     `Config` actually used to simulate this strategy.
        inputs:  `Inputs` actually used (may differ from the base
                 `inputs` when the optimizer flips Roth-401(k) splits).
        df:      Year-by-year simulation frame returned by `simulate()`.
        summary: `summarize(df)` aggregates (terminal NW, lifetime tax /
                 IRMAA NPV, peak marginal rate, IRMAA exposure, etc.).
    """

    cfg: "Config"
    inputs: "Inputs"
    df: "pd.DataFrame"
    summary: dict
