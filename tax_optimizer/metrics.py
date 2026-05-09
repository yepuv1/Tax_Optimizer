"""Strategy summarizers."""

from __future__ import annotations

import numpy as np
import pandas as pd


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


def lifetime_tax_npv(df: pd.DataFrame, *, discount: float = 0.025) -> float:
    years = np.arange(len(df))
    return float((df["federal_tax"].values / (1 + discount) ** years).sum())


def lifetime_irmaa_npv(df: pd.DataFrame, *, discount: float = 0.025) -> float:
    years = np.arange(len(df))
    return float((df["irmaa"].values / (1 + discount) ** years).sum())


def summarize(
    df: pd.DataFrame, *, heir_marginal_rate: float = 0.22
) -> dict:
    return {
        "lifetime_tax_npv": lifetime_tax_npv(df),
        "lifetime_irmaa_npv": lifetime_irmaa_npv(df),
        "terminal_after_tax": terminal_after_tax_nw(
            df, heir_marginal_rate=heir_marginal_rate
        ),
        "peak_marginal": float(df["marginal"].max()),
        "years_irmaa": int((df["irmaa"] > 0).sum()),
        "peak_irmaa_tier": int(df["irmaa_tier"].max()),
        "min_balance": float(
            (df["pretax_balance"] + df["roth_balance"] + df["taxable_balance"]).min()
        ),
    }
