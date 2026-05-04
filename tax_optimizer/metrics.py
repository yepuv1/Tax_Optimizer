"""Strategy summarizers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def terminal_after_tax_nw(
    df: pd.DataFrame, *, marginal_rate: float = 0.22
) -> float:
    last = df.iloc[-1]
    pretax_after_tax = last["pretax_balance"] * (1 - marginal_rate)
    return (
        pretax_after_tax
        + last["roth_balance"]
        + last["taxable_balance"]
        + last["hsa_balance"]
    )


def lifetime_tax_npv(df: pd.DataFrame, *, discount: float = 0.025) -> float:
    years = np.arange(len(df))
    return float((df["federal_tax"].values / (1 + discount) ** years).sum())


def lifetime_irmaa_npv(df: pd.DataFrame, *, discount: float = 0.025) -> float:
    years = np.arange(len(df))
    return float((df["irmaa"].values / (1 + discount) ** years).sum())


def summarize(df: pd.DataFrame) -> dict:
    return {
        "lifetime_tax_npv": lifetime_tax_npv(df),
        "lifetime_irmaa_npv": lifetime_irmaa_npv(df),
        "terminal_after_tax": terminal_after_tax_nw(df),
        "peak_marginal": float(df["marginal"].max()),
        "years_irmaa": int((df["irmaa"] > 0).sum()),
        "peak_irmaa_tier": int(df["irmaa_tier"].max()),
        "min_balance": float(
            (df["pretax_balance"] + df["roth_balance"] + df["taxable_balance"]).min()
        ),
    }
