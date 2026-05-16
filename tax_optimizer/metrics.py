"""Strategy summarizers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - avoid runtime circular import
    from .inputs import StartingBalances


def terminal_after_tax_nw(
    df: pd.DataFrame,
    *,
    heir_marginal_rate: float = 0.22,
    marginal_rate: float | None = None,
) -> float:
    """Terminal liquid net worth, **bequest-tax-aware**.

    The horizon balance is split across four buckets with very different
    after-tax values to heirs:

      * ``roth``   — fully tax-free to heirs.
      * ``taxable``— stepped-up cost basis at death, so heirs face ~zero
                     tax on the embedded gain. Treated here as $1 = $1.
      * ``hsa``    — to non-spouse heirs becomes ordinary-income at the
                     decedent's death; conservatively treated as pretax
                     for bequest purposes.
      * ``pretax`` — heirs must drain over 10 years (SECURE Act) at
                     their own marginal rate. Discounted by
                     ``heir_marginal_rate``.

    The 10-year drawdown timing within those years isn't modeled — a
    flat haircut by the heir's bracket-rate is the standard
    approximation. Use ``cfg.heir_marginal_rate`` to set this from a
    Config, defaulting to 22% (the typical adult-child marginal in
    today's bracket structure).

    The legacy ``marginal_rate`` keyword is kept as a back-compat
    alias and takes precedence over ``heir_marginal_rate`` when set.
    """
    rate = marginal_rate if marginal_rate is not None else heir_marginal_rate
    last = df.iloc[-1]
    pretax_after_tax = last["pretax_balance"] * (1 - rate)
    hsa_after_tax = last["hsa_balance"] * (1 - rate)
    return (
        pretax_after_tax
        + last["roth_balance"]
        + last["taxable_balance"]
        + hsa_after_tax
    )


def starting_after_tax_nw(
    starting: "StartingBalances", *, heir_marginal_rate: float = 0.22
) -> float:
    """Starting liquid net worth, **bequest-tax-aware**.

    Mirrors `terminal_after_tax_nw` so that the starting / terminal
    pair can be divided to produce a meaningful growth ratio. Uses
    the same four-bucket split: pretax + HSA discounted by the
    heir's marginal rate, Roth + taxable at face value.

    Pension reserve (``pension_balance``) is intentionally
    excluded — same as in `terminal_after_tax_nw`, which only
    reads the four "liquid" buckets off the simulator's last row.

    Pretax aggregates both spouses' 401(k) and IRA balances; Roth
    aggregates both spouses' Roth IRAs. Matches the pooling done
    by `state.initial_state`.
    """
    rate = heir_marginal_rate
    pretax = (
        starting.spouse_a_pretax_401k
        + starting.spouse_b_pretax_401k
        + starting.spouse_a_pretax_ira
        + starting.spouse_b_pretax_ira
    )
    roth = starting.spouse_a_roth_ira + starting.spouse_b_roth_ira
    taxable = starting.taxable_brokerage
    hsa = starting.hsa
    return pretax * (1 - rate) + roth + taxable + hsa * (1 - rate)


def total_growth_multiplier(
    starting_nw: float, terminal_nw: float
) -> float:
    """``terminal_nw / starting_nw`` (a multiplier like 3.15).

    Returns NaN when `starting_nw <= 0` — without a positive
    baseline the multiplier is undefined and any number we returned
    would be misleading.
    """
    if starting_nw <= 0:
        return float("nan")
    return float(terminal_nw / starting_nw)


def _signed_pow(base: float, exponent: float) -> float:
    """Python's ``base ** exponent`` raises for negative bases with
    fractional exponents (returns a complex number). For our CAGR
    work we want the *real-valued* root with the sign of the base
    preserved, so the rate stays interpretable when the plan
    actually loses money over the horizon.
    """
    if base == 0:
        return 0.0
    sign = 1.0 if base > 0 else -1.0
    return sign * (abs(base) ** exponent)


def effective_cagr(
    starting_nw: float, terminal_nw: float, years: int
) -> float:
    """Effective compound annual growth rate of after-tax NW.

    ``(terminal / starting) ** (1 / years) - 1``. Returns the
    per-year fraction (e.g. 0.042 → "4.2 %/yr").

    **Important caveat:** for a household-NW timeline this rate
    bundles market returns, contributions, withdrawals, and tax
    drag into one effective compounding rate. It is NOT a pure
    investment return; the CAGR is the right answer to "at what
    effective rate did the plan compound?" but the wrong answer to
    "what return did my portfolio earn?".

    Returns 0.0 for ``years <= 0`` (degenerate timeline) and NaN
    for non-positive ``starting_nw`` (consistent with
    `total_growth_multiplier`).
    """
    if years <= 0:
        return 0.0
    if starting_nw <= 0:
        return float("nan")
    ratio = terminal_nw / starting_nw
    return float(_signed_pow(ratio, 1.0 / years) - 1.0)


def real_cagr(
    starting_nw: float, terminal_nw: float, years: int, inflation: float
) -> float:
    """Inflation-adjusted CAGR.

    Discounts the terminal value by ``(1 + inflation) ** years``
    before computing CAGR — i.e. expresses terminal NW in today's
    dollars and then compares to the starting NW (which is already
    in today's dollars). The result is the rate of growth of
    *purchasing power*, not nominal dollars.

    For a 30-year horizon at 2.5 % inflation, real CAGR is
    ~ ``nominal - 0.025`` for small rates (Fisher equation).
    """
    if years <= 0:
        return 0.0
    if starting_nw <= 0:
        return float("nan")
    real_terminal = terminal_nw / ((1 + inflation) ** years)
    return effective_cagr(starting_nw, real_terminal, years)


def nw_after_tax_series(
    df: pd.DataFrame, *, heir_marginal_rate: float = 0.22
) -> pd.Series:
    """Vectorized version of `terminal_after_tax_nw`.

    Returns a Series of after-tax-NW per row, indexed like ``df``.
    Driven by the same four-bucket bequest-tax-aware formula:
    ``pretax × (1 - rate) + roth + taxable + hsa × (1 - rate)``.

    Used by the Overview growth chart and any caller that wants a
    "true wealth over time" line rather than separate balance
    columns.
    """
    rate = heir_marginal_rate
    pretax = df["pretax_balance"]
    roth = df["roth_balance"]
    taxable = df["taxable_balance"]
    hsa = df["hsa_balance"]
    return pretax * (1 - rate) + roth + taxable + hsa * (1 - rate)


def stage_cagr(
    df: pd.DataFrame,
    starting_nw: float,
    *,
    retire_age: int,
    heir_marginal_rate: float = 0.22,
) -> tuple[Optional[float], Optional[float]]:
    """CAGR split at the retirement boundary.

    Accumulation phase = years where ``spouse_a_age < retire_age``
    (matches `report.py`'s convention for the same split). The
    boundary year's after-tax NW is the *end* of accumulation /
    *start* of decumulation. We use that single anchor point for
    both phases so the two CAGRs compose to the full-horizon CAGR
    when re-annualized over their respective year counts.

    Returns ``(accumulation_cagr, decumulation_cagr)``. Either
    side can be `None` if the timeline doesn't span that phase
    (e.g. user already retired ⇒ no accumulation; or horizon
    age < retire_age ⇒ no decumulation).

    Decumulation CAGR is often negative — that's the whole point
    of surfacing it: it tells the user how fast their NW is
    drawing down once contributions stop.
    """
    if "spouse_a_age" not in df.columns or len(df) == 0:
        return None, None

    accum_mask = df["spouse_a_age"] < retire_age
    decum_mask = ~accum_mask

    accum_cagr: Optional[float] = None
    decum_cagr: Optional[float] = None

    # Pick the boundary row — the *last* accumulation year if any,
    # otherwise the starting NW serves as the boundary itself.
    if accum_mask.any():
        boundary_row = df[accum_mask].iloc[-1]
        boundary_nw = float(
            boundary_row["pretax_balance"] * (1 - heir_marginal_rate)
            + boundary_row["roth_balance"]
            + boundary_row["taxable_balance"]
            + boundary_row["hsa_balance"] * (1 - heir_marginal_rate)
        )
        accum_years = int(accum_mask.sum())
        accum_cagr = effective_cagr(starting_nw, boundary_nw, accum_years)
    else:
        # No accumulation rows: decumulation starts immediately
        # from the starting NW.
        boundary_nw = starting_nw

    if decum_mask.any():
        last_row = df.iloc[-1]
        terminal_nw = float(
            last_row["pretax_balance"] * (1 - heir_marginal_rate)
            + last_row["roth_balance"]
            + last_row["taxable_balance"]
            + last_row["hsa_balance"] * (1 - heir_marginal_rate)
        )
        decum_years = int(decum_mask.sum())
        decum_cagr = effective_cagr(boundary_nw, terminal_nw, decum_years)

    return accum_cagr, decum_cagr


def lifetime_tax_npv(df: pd.DataFrame, *, discount: float = 0.025) -> float:
    years = np.arange(len(df))
    return float((df["federal_tax"].values / (1 + discount) ** years).sum())


def lifetime_irmaa_npv(df: pd.DataFrame, *, discount: float = 0.025) -> float:
    years = np.arange(len(df))
    return float((df["irmaa"].values / (1 + discount) ** years).sum())


def summarize(
    df: pd.DataFrame,
    *,
    heir_marginal_rate: float = 0.22,
    starting_balances: Optional["StartingBalances"] = None,
    inflation: Optional[float] = None,
    retire_age: Optional[int] = None,
) -> dict:
    """Roll up a strategy DataFrame into a dict of summary metrics.

    The optional kwargs (``starting_balances``, ``inflation``,
    ``retire_age``) drive the growth-rate metrics
    (``starting_after_tax``, ``total_growth_mult``,
    ``effective_cagr``, ``real_cagr``, ``accumulation_cagr``,
    ``decumulation_cagr``). When any of those kwargs is omitted,
    the corresponding metric is set to ``None`` so legacy callers
    continue to work without surfacing a misleading half-computed
    growth number.
    """
    # `min_balance` reflects the minimum spendable wealth across the
    # plan. HSA was historically omitted, which understated risk for
    # HSA-heavy plans where the HSA is the dominant liquid bucket
    # post-65. HSA balance counts only once at least one spouse hits
    # 65 (the IRS no-penalty age) — before that the HSA is restricted
    # to qualified medical use.
    HSA_PENALTY_FREE_AGE = 65
    if "spouse_a_age" in df.columns and "spouse_b_age" in df.columns:
        hsa_unlocked = (
            (df["spouse_a_age"] >= HSA_PENALTY_FREE_AGE)
            | (df["spouse_b_age"] >= HSA_PENALTY_FREE_AGE)
        )
        liquid = (
            df["pretax_balance"]
            + df["roth_balance"]
            + df["taxable_balance"]
            + df["hsa_balance"].where(hsa_unlocked, 0.0)
        )
    else:
        # Defensive: legacy DataFrames missing the per-year ages —
        # fall back to the pre-v6.2 behavior (HSA omitted).
        liquid = (
            df["pretax_balance"] + df["roth_balance"] + df["taxable_balance"]
        )

    terminal_nw = terminal_after_tax_nw(
        df, heir_marginal_rate=heir_marginal_rate
    )

    # Growth metrics — pre-computed here so the dashboard / report
    # layer doesn't need to re-derive them from raw inputs every
    # render. Each metric is None when its required kwarg is missing
    # so legacy callers (most tests) still get a consistent dict
    # shape.
    starting_nw: Optional[float]
    growth_mult: Optional[float]
    nominal_cagr: Optional[float]
    inflation_cagr: Optional[float]
    accum_cagr: Optional[float]
    decum_cagr: Optional[float]
    if starting_balances is not None:
        starting_nw = starting_after_tax_nw(
            starting_balances, heir_marginal_rate=heir_marginal_rate
        )
        years = len(df)
        growth_mult = total_growth_multiplier(starting_nw, terminal_nw)
        nominal_cagr = effective_cagr(starting_nw, terminal_nw, years)
        if inflation is not None:
            inflation_cagr = real_cagr(
                starting_nw, terminal_nw, years, inflation
            )
        else:
            inflation_cagr = None
        if retire_age is not None:
            accum_cagr, decum_cagr = stage_cagr(
                df,
                starting_nw,
                retire_age=retire_age,
                heir_marginal_rate=heir_marginal_rate,
            )
        else:
            accum_cagr, decum_cagr = None, None
    else:
        starting_nw = None
        growth_mult = None
        nominal_cagr = None
        inflation_cagr = None
        accum_cagr = None
        decum_cagr = None

    return {
        "lifetime_tax_npv": lifetime_tax_npv(df),
        "lifetime_irmaa_npv": lifetime_irmaa_npv(df),
        "terminal_after_tax": terminal_nw,
        "peak_marginal": float(df["marginal"].max()),
        "years_irmaa": int((df["irmaa"] > 0).sum()),
        "peak_irmaa_tier": int(df["irmaa_tier"].max()),
        "min_balance": float(liquid.min()),
        # Growth metrics (None if `starting_balances` not supplied).
        "starting_after_tax": starting_nw,
        "total_growth_mult": growth_mult,
        "effective_cagr": nominal_cagr,
        "real_cagr": inflation_cagr,
        "accumulation_cagr": accum_cagr,
        "decumulation_cagr": decum_cagr,
    }
