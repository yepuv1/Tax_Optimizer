"""Plotly figure builders.

These are pure functions of pandas DataFrames / numpy arrays - no Dash
imports, no global state. That keeps them trivially unit-testable and
makes it easy to render the same charts from a notebook.

Every builder returns a `plotly.graph_objects.Figure` ready to drop into
`dcc.Graph(figure=...)`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Reusable layout fragments
_AXIS_DOLLAR = dict(tickprefix="$", tickformat=",.0f")
_AXIS_PCT = dict(ticksuffix="%", tickformat=".0f")
_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=60, r=20, t=50, b=50),
    legend=dict(orientation="h", y=-0.15, x=0),
    hovermode="x unified",
)


def empty_figure(message: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        annotations=[
            dict(
                text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14, color="#64748b"),
            ),
        ],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


# ---------------------------------------------------------------------
# Balances over time
# ---------------------------------------------------------------------


def balance_stack(df: pd.DataFrame, *, title: str = "Account balances") -> go.Figure:
    if df is None or len(df) == 0:
        return empty_figure("No simulation rows")
    x = df["spouse_a_age"] if "spouse_a_age" in df.columns else df.index
    fig = go.Figure()
    series = [
        ("pretax_balance", "Pretax", "#0ea5e9"),
        ("roth_balance", "Roth", "#22c55e"),
        ("taxable_balance", "Taxable", "#f59e0b"),
        ("hsa_balance", "HSA", "#a855f7"),
    ]
    for col, label, color in series:
        if col not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=x, y=df[col], name=label, mode="lines",
                stackgroup="balances", line=dict(width=0.5, color=color),
                hovertemplate=f"{label}: $%{{y:,.0f}}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Spouse A age",
        yaxis_title="Balance",
        yaxis=_AXIS_DOLLAR,
        **_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------
# Taxes & marginal bracket
# ---------------------------------------------------------------------


def taxes_panel(df: pd.DataFrame, *, title: str = "Taxes & marginal bracket") -> go.Figure:
    if df is None or len(df) == 0:
        return empty_figure("No simulation rows")
    x = df["spouse_a_age"] if "spouse_a_age" in df.columns else df.index

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.62, 0.38], vertical_spacing=0.07,
        subplot_titles=("AGI / federal / state tax / IRMAA",
                        "Marginal bracket"),
    )

    if "agi" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x, y=df["agi"], name="AGI", mode="lines",
                line=dict(color="#0ea5e9", width=2),
                hovertemplate="AGI: $%{y:,.0f}<extra></extra>",
            ),
            row=1, col=1,
        )
    if "federal_tax" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x, y=df["federal_tax"], name="Federal tax", mode="lines",
                line=dict(color="#ef4444", width=2),
                hovertemplate="Federal: $%{y:,.0f}<extra></extra>",
            ),
            row=1, col=1,
        )
    if "state_tax" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x, y=df["state_tax"], name="State tax", mode="lines",
                line=dict(color="#f97316", width=2, dash="dash"),
                hovertemplate="State: $%{y:,.0f}<extra></extra>",
            ),
            row=1, col=1,
        )
    if "irmaa" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x, y=df["irmaa"], name="IRMAA", mode="lines",
                line=dict(color="#a855f7", width=2, dash="dot"),
                hovertemplate="IRMAA: $%{y:,.0f}<extra></extra>",
            ),
            row=1, col=1,
        )

    if "marginal" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x, y=df["marginal"] * 100, name="Marginal %",
                mode="lines+markers",
                line=dict(color="#0f172a", width=2),
                hovertemplate="Marginal: %{y:.0f}%<extra></extra>",
            ),
            row=2, col=1,
        )

    fig.update_yaxes(title_text="Dollars", **_AXIS_DOLLAR, row=1, col=1)
    fig.update_yaxes(title_text="Marginal %", **_AXIS_PCT, row=2, col=1)
    fig.update_xaxes(title_text="Spouse A age", row=2, col=1)
    fig.update_layout(title=title, **_LAYOUT, height=560)
    return fig


# ---------------------------------------------------------------------
# Conversion + RMD timeline
# ---------------------------------------------------------------------


def conversion_panel(
    df: pd.DataFrame, *, title: str = "Roth conversions & RMDs"
) -> go.Figure:
    if df is None or len(df) == 0:
        return empty_figure("No simulation rows")
    x = df["spouse_a_age"] if "spouse_a_age" in df.columns else df.index

    fig = go.Figure()
    if "roth_conversion" in df.columns:
        fig.add_trace(
            go.Bar(
                x=x, y=df["roth_conversion"], name="Roth conversion",
                marker_color="#22c55e",
                hovertemplate="Conv: $%{y:,.0f}<extra></extra>",
            )
        )
    if "rmd" in df.columns:
        fig.add_trace(
            go.Bar(
                x=x, y=df["rmd"], name="RMD",
                marker_color="#f59e0b",
                hovertemplate="RMD: $%{y:,.0f}<extra></extra>",
            )
        )

    # Mark years where the liquidity guard trimmed the conversion.
    if "roth_conv_capped_by_liquidity" in df.columns and "roth_conversion" in df.columns:
        cap_mask = df["roth_conv_capped_by_liquidity"].astype(bool).fillna(False)
        if cap_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=x[cap_mask],
                    y=df.loc[cap_mask, "roth_conversion"],
                    name="Liquidity-capped",
                    mode="markers",
                    marker=dict(symbol="x", size=10, color="#ef4444"),
                    hovertemplate="Capped by liquidity<extra></extra>",
                )
            )

    fig.update_layout(
        title=title,
        barmode="group",
        xaxis_title="Spouse A age",
        yaxis_title="Dollars",
        yaxis=_AXIS_DOLLAR,
        **_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------
# Strategy comparison
# ---------------------------------------------------------------------


def strategy_comparison(
    strategies: dict[str, dict[str, Any]],
    *,
    metric: str = "terminal_after_tax",
    title: str = "Strategy comparison",
) -> go.Figure:
    """Horizontal bar of `summary[metric]` across strategies.

    `strategies` is the deserialized payload: each value is a dict with
    a "summary" sub-dict.
    """
    if not strategies:
        return empty_figure("Run the simulator to populate strategies")

    names = list(strategies.keys())
    values = [strategies[n]["summary"].get(metric, np.nan) for n in names]
    baseline = strategies.get("S0_baseline", {}).get("summary", {}).get(metric, None)

    text = []
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            text.append("")
        elif baseline:
            text.append(f"${v:,.0f}  ({(v - baseline) / max(abs(baseline), 1) * 100:+.1f}%)")
        else:
            text.append(f"${v:,.0f}")

    fig = go.Figure(
        go.Bar(
            x=values, y=names, orientation="h",
            marker_color=["#0ea5e9" if n != "S3_optimized" else "#22c55e" for n in names],
            text=text, textposition="outside",
            hovertemplate="%{y}: $%{x:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{title} - {metric}",
        xaxis=_AXIS_DOLLAR,
        yaxis=dict(autorange="reversed"),
        **_LAYOUT,
        height=300,
    )
    return fig


def strategy_compare_panel(
    strategies: dict[str, dict[str, Any]],
) -> go.Figure:
    """3-panel view of terminal NW, lifetime tax NPV, and lifetime IRMAA NPV."""
    if not strategies:
        return empty_figure("Run the simulator to populate strategies")
    names = list(strategies.keys())
    metrics = [
        ("terminal_after_tax", "Terminal after-tax NW"),
        ("lifetime_tax_npv", "Lifetime federal tax (NPV)"),
        ("lifetime_irmaa_npv", "Lifetime IRMAA (NPV)"),
    ]
    fig = make_subplots(
        rows=1, cols=len(metrics),
        subplot_titles=[m[1] for m in metrics],
        shared_yaxes=True,
    )
    for i, (key, _label) in enumerate(metrics, start=1):
        values = [strategies[n]["summary"].get(key, 0.0) for n in names]
        fig.add_trace(
            go.Bar(
                x=values, y=names, orientation="h", showlegend=False,
                marker_color=["#0ea5e9" if n != "S3_optimized" else "#22c55e" for n in names],
                text=[f"${v:,.0f}" for v in values], textposition="outside",
                hovertemplate="%{y}: $%{x:,.0f}<extra></extra>",
            ),
            row=1, col=i,
        )
        fig.update_xaxes(**_AXIS_DOLLAR, row=1, col=i)
    fig.update_layout(
        title="Strategy comparison",
        yaxis=dict(autorange="reversed"),
        **_LAYOUT,
        height=380,
    )
    return fig


# ---------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------


def mc_terminal_histogram(mc_payload: dict[str, Any] | None) -> go.Figure:
    if not mc_payload:
        return empty_figure("Monte Carlo not run")
    terminals = np.asarray(mc_payload["terminals"], dtype=float)
    pcts = mc_payload["percentiles"]
    fig = go.Figure(
        go.Histogram(
            x=terminals, nbinsx=40,
            marker_color="#0ea5e9", opacity=0.85,
            hovertemplate="Terminal: $%{x:,.0f}<br>Paths: %{y}<extra></extra>",
        )
    )
    for tag, color in (("p10", "#ef4444"), ("p50", "#0f172a"), ("p90", "#22c55e")):
        v = pcts[tag]
        fig.add_vline(
            x=v, line=dict(color=color, dash="dash"),
            annotation_text=f"{tag.upper()}: ${v:,.0f}",
            annotation_position="top",
        )
    fig.update_layout(
        title=(
            f"Terminal NW distribution"
            f" - p_success={mc_payload['prob_success']:.0%},"
            f" CVaR(10%)=${mc_payload['cvar_terminal']:,.0f}"
        ),
        xaxis_title="Terminal after-tax NW",
        yaxis_title="Paths",
        xaxis=_AXIS_DOLLAR,
        **_LAYOUT,
    )
    return fig


def mc_fan_chart(mc_payload: dict[str, Any] | None) -> go.Figure:
    if not mc_payload or not mc_payload.get("fan"):
        return empty_figure("Monte Carlo fan unavailable")
    fan = mc_payload["fan"]
    x = fan["year_offset"]
    fig = go.Figure()
    # P10-P90 band
    fig.add_trace(
        go.Scatter(
            x=x + x[::-1],
            y=fan["p90"] + fan["p10"][::-1],
            fill="toself",
            fillcolor="rgba(14, 165, 233, 0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="P10-P90",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=fan["p50"], mode="lines", name="P50 (median)",
            line=dict(color="#0f172a", width=2),
            hovertemplate="Year %{x}: $%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=fan["p10"], mode="lines", name="P10",
            line=dict(color="#ef4444", width=1, dash="dot"),
            hovertemplate="P10 year %{x}: $%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=fan["p90"], mode="lines", name="P90",
            line=dict(color="#22c55e", width=1, dash="dot"),
            hovertemplate="P90 year %{x}: $%{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Liquid net worth - Monte Carlo fan (P10 / P50 / P90)",
        xaxis_title="Year offset",
        yaxis_title="Liquid NW",
        yaxis=_AXIS_DOLLAR,
        **_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------
# KPI tile data (consumed by layout, not a figure)
# ---------------------------------------------------------------------


def overview_kpis(
    summary: dict[str, Any], mc_payload: dict[str, Any] | None
) -> list[tuple[str, str]]:
    """Returns a list of (label, value-string) for the Overview KPI tiles."""
    tiles: list[tuple[str, str]] = []
    tiles.append(("Terminal after-tax NW",
                  _fmt_dollars(summary.get("terminal_after_tax"))))
    tiles.append(("Lifetime federal tax (NPV)",
                  _fmt_dollars(summary.get("lifetime_tax_npv"))))
    tiles.append(("Lifetime IRMAA (NPV)",
                  _fmt_dollars(summary.get("lifetime_irmaa_npv"))))
    pm = summary.get("peak_marginal")
    tiles.append(("Peak marginal rate",
                  f"{pm * 100:.0f}%" if pm is not None else "-"))
    yi = summary.get("years_irmaa")
    tiles.append(("Years with IRMAA",
                  f"{int(yi)}" if yi is not None else "-"))
    if mc_payload:
        tiles.append(("Probability of success",
                      f"{mc_payload['prob_success']:.0%}"))
        tiles.append(("CVaR (10%)",
                      _fmt_dollars(mc_payload["cvar_terminal"])))
    return tiles


def _fmt_dollars(v: Any) -> str:
    if v is None:
        return "-"
    try:
        return f"${float(v):,.0f}"
    except (TypeError, ValueError):
        return "-"


def detail_columns() -> list[str]:
    """Year-by-year drill-down columns (mirror the notebook's detail_cols)."""
    return [
        "year", "spouse_a_age", "filing_status",
        "wages", "pension", "ssn",
        "rmd", "roth_conversion",
        "pretax_withdrawal", "roth_withdrawal", "taxable_withdrawal",
        "agi", "federal_tax", "state_tax", "marginal",
        "irmaa", "medicare_base_premium", "spending_need",
        "pretax_balance", "roth_balance", "taxable_balance", "hsa_balance",
    ]


def filter_to_detail_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in detail_columns() if c in df.columns]
    out = df[cols].copy()
    if "marginal" in out.columns:
        out["marginal"] = (out["marginal"] * 100).round(0).astype("Int64")
        out = out.rename(columns={"marginal": "bracket_pct"})
    for c in out.columns:
        if c in {"year", "spouse_a_age", "bracket_pct", "filing_status"}:
            continue
        out[c] = out[c].round(0)
    return out
