"""Matplotlib helpers (gated by caller; package never auto-shows)."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ImportError(
        "matplotlib is required for tax_optimizer.plots; install with "
        "`pip install matplotlib`."
    ) from e

from .monte_carlo import MonteCarloResult


def plot_balance_paths(results: dict, *, ax=None, top_n: int = 4):
    """Plot liquid balance per strategy. `results` is `dict[str, StrategyResult]`.

    Tolerates the old `(cfg, df, summary)` tuple shape too so notebooks
    that still cache the legacy form keep rendering.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    series = []
    for name, val in results.items():
        if hasattr(val, "df"):
            df = val.df
        else:  # legacy 3-tuple shape (cfg, df, summary)
            _cfg, df, _sum = val
        liquid = df["pretax_balance"] + df["roth_balance"] + df["taxable_balance"]
        series.append((name, df["spouse_a_age"].to_numpy(), liquid.to_numpy()))
    series = series[:top_n]
    for name, ages, vals in series:
        ax.plot(ages, vals, label=name)
    ax.set_title("Liquid balance (pretax + Roth + taxable)")
    ax.set_xlabel("Spouse A age")
    ax.set_ylabel("$")
    ax.legend()
    return ax


def plot_monte_carlo_fan(
    mc: MonteCarloResult, *, metric: str = "pretax_balance", ax=None
):
    """Fan chart of `metric` across all paths."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    if not mc.paths:
        raise ValueError("MonteCarloResult has no path data; rerun with keep_paths=True")
    sample = mc.paths[0]
    ages = sample["spouse_a_age"].to_numpy()
    arr = np.stack(
        [df[metric].to_numpy() if metric in df.columns
         else (df["pretax_balance"] + df["roth_balance"] + df["taxable_balance"]).to_numpy()
         for df in mc.paths]
    )
    p5 = np.percentile(arr, 5, axis=0)
    p25 = np.percentile(arr, 25, axis=0)
    p50 = np.percentile(arr, 50, axis=0)
    p75 = np.percentile(arr, 75, axis=0)
    p95 = np.percentile(arr, 95, axis=0)
    ax.fill_between(ages, p5, p95, alpha=0.15, label="5-95%")
    ax.fill_between(ages, p25, p75, alpha=0.30, label="25-75%")
    ax.plot(ages, p50, lw=2, label="median")
    ax.set_title(f"Monte Carlo fan: {metric} ({mc.n_paths} paths)")
    ax.set_xlabel("Spouse A age")
    ax.set_ylabel("$")
    ax.legend()
    return ax


def plot_terminal_distribution(mc: MonteCarloResult, *, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(mc.terminals / 1e6, bins=40, alpha=0.85)
    ax.axvline(np.median(mc.terminals) / 1e6, color="black", lw=2, label="median")
    ax.axvline(np.percentile(mc.terminals, 5) / 1e6, color="red", ls="--", label="p5")
    ax.set_title(f"Terminal after-tax NW distribution ({mc.n_paths} paths)")
    ax.set_xlabel("Terminal NW ($M)")
    ax.set_ylabel("count")
    ax.legend()
    return ax


def plot_tornado(sens_df: pd.DataFrame, *, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    df = sens_df.copy()
    df["param"] = df["param"].str.replace("_", " ")
    y = np.arange(len(df))
    ax.barh(y, df["delta_high"].values / 1e3, alpha=0.7, label="high - base")
    ax.barh(y, df["delta_low"].values / 1e3, alpha=0.7, label="low - base")
    ax.set_yticks(y)
    ax.set_yticklabels(df["param"])
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Change in terminal NW ($K)")
    ax.set_title("Tornado: parameter sensitivity")
    ax.legend()
    return ax
