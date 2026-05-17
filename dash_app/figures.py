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

# Plotly figures render their text in SVG and do NOT inherit
# ``font-family`` from the surrounding HTML the way ordinary Dash
# components do. So we mirror the dashboard's CSS monospace stack
# (see ``dash_app/assets/fira-code.css`` :root) inside every
# figure's layout. The fallback chain reaches the same
# system-monospace fonts the CSS does so a viewer who blocked
# Google Fonts still gets a coherent look.
#
# Single Plotly font string (comma-separated, quoted multi-word
# names) covers every text element automatically: chart title,
# axis titles + tick labels, legend, hover tooltips, annotations.
# Per-element ``font`` overrides scattered through this module
# only set ``size`` / ``color`` — the family still inherits from
# the layout default below.
_FIRA_FONT_FAMILY = (
    "'Fira Code', 'Fira Mono', 'JetBrains Mono', "
    "'SFMono-Regular', ui-monospace, Menlo, Monaco, Consolas, "
    "'Liberation Mono', monospace"
)

_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=60, r=20, t=50, b=50),
    legend=dict(orientation="h", y=-0.15, x=0),
    hovermode="x unified",
    # Default text size matches Bootstrap's base-font scaling on
    # the rest of the dashboard. Per-element overrides can still
    # bump this up (chart titles use 15px, callouts 11px, etc.)
    # via their own ``font=dict(size=...)`` kwargs.
    font=dict(family=_FIRA_FONT_FAMILY, size=12),
)


# ---------------------------------------------------------------------
# Color-universal-design (CUD) palette — Okabe-Ito
# ---------------------------------------------------------------------
#
# We use the Okabe-Ito palette (Okabe & Ito 2008,
# https://jfly.uni-koeln.de/color/) as the canonical color set for
# every figure in the dashboard. Okabe-Ito is the de-facto standard
# in scientific publishing for accessibility because the eight hues
# are designed to be pairwise-distinguishable across the three most
# common forms of color blindness (deuteranopia, protanopia,
# tritanopia) AND under simulated monochrome conditions (each color
# also has a distinct *luminance*).
#
# Where a figure encodes more than one categorical dimension with
# color (e.g. the multi-strategy charts encode strategy via line
# color), we ALSO encode it with a redundant non-color cue —
# distinct marker symbols, line dash patterns, or text icons — so
# that achromatopsia (total color blindness) and B&W printing both
# still convey the information.
_OKABE_ITO: dict[str, str] = {
    "black":          "#000000",
    "orange":         "#E69F00",
    "sky_blue":       "#56B4E9",
    "bluish_green":   "#009E73",
    "yellow":         "#F0E442",
    "blue":           "#0072B2",
    "vermillion":     "#D55E00",
    "reddish_purple": "#CC79A7",
}


def empty_figure(message: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        # Inherit the Fira Code stack so the placeholder message
        # matches the look of populated charts. The annotation's
        # own ``font`` only sets size + color so family flows
        # through from this layout default.
        font=dict(family=_FIRA_FONT_FAMILY, size=12),
        annotations=[
            dict(
                text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14, color="#475569"),
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


# Account-type → Okabe-Ito color. Same hues as the strategy
# palette but mapped to the four account buckets the simulator
# tracks. The "Roth" bucket gets bluish-green (the same "positive"
# hue we give the optimizer) since tax-free growth is the
# household's most valuable bucket dollar-for-dollar; "Pretax"
# gets blue as the workhorse default; "Taxable" gets orange (warm
# = drag from annual yield); "HSA" gets reddish-purple (small,
# specialized).
_ACCOUNT_COLORS: dict[str, str] = {
    "pretax":  _OKABE_ITO["blue"],
    "roth":    _OKABE_ITO["bluish_green"],
    "taxable": _OKABE_ITO["orange"],
    "hsa":     _OKABE_ITO["reddish_purple"],
}


def balance_stack(df: pd.DataFrame, *, title: str = "Account balances") -> go.Figure:
    if df is None or len(df) == 0:
        return empty_figure("No simulation rows")
    x = df["spouse_a_age"] if "spouse_a_age" in df.columns else df.index
    fig = go.Figure()
    series = [
        ("pretax_balance",  "Pretax",  _ACCOUNT_COLORS["pretax"]),
        ("roth_balance",    "Roth",    _ACCOUNT_COLORS["roth"]),
        ("taxable_balance", "Taxable", _ACCOUNT_COLORS["taxable"]),
        ("hsa_balance",     "HSA",     _ACCOUNT_COLORS["hsa"]),
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

    # Tax-component palette — Okabe-Ito hues + dash-pattern
    # redundancy. Vermillion replaces the previous `#ef4444` red
    # for federal tax (red-on-green is the canonical CB-bad
    # pairing); reddish-purple replaces the previous purple for
    # IRMAA so it stays distinct from federal. Every line also
    # gets a unique dash pattern so the components are
    # identifiable in B&W.
    if "agi" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x, y=df["agi"], name="AGI", mode="lines",
                line=dict(color=_OKABE_ITO["blue"], width=2),
                hovertemplate="AGI: $%{y:,.0f}<extra></extra>",
            ),
            row=1, col=1,
        )
    if "federal_tax" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x, y=df["federal_tax"], name="Federal tax", mode="lines",
                line=dict(color=_OKABE_ITO["vermillion"], width=2),
                hovertemplate="Federal: $%{y:,.0f}<extra></extra>",
            ),
            row=1, col=1,
        )
    if "state_tax" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x, y=df["state_tax"], name="State tax", mode="lines",
                line=dict(color=_OKABE_ITO["orange"], width=2, dash="dash"),
                hovertemplate="State: $%{y:,.0f}<extra></extra>",
            ),
            row=1, col=1,
        )
    if "irmaa" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x, y=df["irmaa"], name="IRMAA", mode="lines",
                line=dict(color=_OKABE_ITO["reddish_purple"], width=2, dash="dot"),
                hovertemplate="IRMAA: $%{y:,.0f}<extra></extra>",
            ),
            row=1, col=1,
        )
    # FICA + SDI — surfaced on the same Dollars axis so the reader
    # sees the *full* tax bill (federal + state + IRMAA + payroll),
    # not just the income-tax stack. Pre-fix the simulator already
    # emitted `fica_oasdi`, `fica_medicare`, `fica_additional_medicare`,
    # and `state_sdi` columns but no chart surfaced them.
    fica_components = [
        c for c in (
            "fica_oasdi",
            "fica_medicare",
            "fica_additional_medicare",
        ) if c in df.columns
    ]
    if fica_components:
        fica_total = sum(df[c] for c in fica_components)
        if (fica_total != 0).any():
            fig.add_trace(
                go.Scatter(
                    x=x, y=fica_total, name="FICA (OASDI + Medicare)",
                    mode="lines",
                    line=dict(
                        color=_OKABE_ITO["bluish_green"],
                        width=2,
                        dash="dashdot",
                    ),
                    hovertemplate="FICA: $%{y:,.0f}<extra></extra>",
                ),
                row=1, col=1,
            )
    if "state_sdi" in df.columns and (df["state_sdi"] != 0).any():
        fig.add_trace(
            go.Scatter(
                x=x, y=df["state_sdi"], name="State SDI",
                mode="lines",
                line=dict(
                    color=_OKABE_ITO["yellow"],
                    width=2,
                    dash="longdash",
                ),
                hovertemplate="SDI: $%{y:,.0f}<extra></extra>",
            ),
            row=1, col=1,
        )

    if "marginal" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x, y=df["marginal"] * 100, name="Marginal %",
                mode="lines+markers",
                line=dict(color=_OKABE_ITO["black"], width=2),
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
    # Conversion = bluish-green ("the proactive move"); RMD =
    # orange ("the forced move"). Both Okabe-Ito hues are
    # CB-distinguishable from each other and from the
    # liquidity-cap vermillion below.
    if "roth_conversion" in df.columns:
        fig.add_trace(
            go.Bar(
                x=x, y=df["roth_conversion"], name="Roth conversion",
                marker_color=_OKABE_ITO["bluish_green"],
                hovertemplate="Conv: $%{y:,.0f}<extra></extra>",
            )
        )
    if "rmd" in df.columns:
        fig.add_trace(
            go.Bar(
                x=x, y=df["rmd"], name="RMD",
                marker_color=_OKABE_ITO["orange"],
                hovertemplate="RMD: $%{y:,.0f}<extra></extra>",
            )
        )

    # Mark years where the liquidity guard trimmed the conversion.
    # The `x` symbol is itself a non-color cue (no other trace on
    # this chart uses an X); the vermillion fill is the
    # CB-friendlier red.
    if "roth_conv_capped_by_liquidity" in df.columns and "roth_conversion" in df.columns:
        cap_mask = df["roth_conv_capped_by_liquidity"].astype(bool).fillna(False)
        if cap_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=x[cap_mask],
                    y=df.loc[cap_mask, "roth_conversion"],
                    name="Liquidity-capped",
                    mode="markers",
                    marker=dict(symbol="x", size=10, color=_OKABE_ITO["vermillion"]),
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
# Per-strategy palette (used by every multi-strategy figure)
# ---------------------------------------------------------------------
#
# Four maximally-distinct Okabe-Ito hues, plus marker + dash
# patterns so the four strategies stay distinguishable in monochrome
# (achromatopsia, B&W print). The winner (S3) keeps the most
# positive hue (bluish-green) and a solid line so it reads as the
# "verdict" line at a glance.
_STRATEGY_COLORS: dict[str, str] = {
    "S0_baseline":         _OKABE_ITO["blue"],            # #0072B2
    "S1_all_roth_401k":    _OKABE_ITO["reddish_purple"],  # #CC79A7
    "S2_bracket_fill_22":  _OKABE_ITO["orange"],          # #E69F00
    "S3_optimized":        _OKABE_ITO["bluish_green"],    # #009E73
}
_STRATEGY_FALLBACK_COLOR = _OKABE_ITO["sky_blue"]         # #56B4E9

# Marker symbols per strategy. Plotly's named symbols stay legible
# at the small marker_size we use (4.5 px) and are mutually
# unambiguous: a circle never looks like a triangle even at 1px.
_STRATEGY_MARKER: dict[str, str] = {
    "S0_baseline":         "circle",
    "S1_all_roth_401k":    "square",
    "S2_bracket_fill_22":  "diamond",
    "S3_optimized":        "triangle-up",
}
_STRATEGY_FALLBACK_MARKER = "x"

# Line dash patterns per strategy. Solid for the baseline and
# winner (the two anchor strategies the eye should track first);
# dash + dot for the alternatives. With this, the four strategies
# remain distinguishable even on a monochrome printer: solid
# circle, dashed square, dotted diamond, solid triangle.
_STRATEGY_DASH: dict[str, str] = {
    "S0_baseline":         "solid",
    "S1_all_roth_401k":    "dash",
    "S2_bracket_fill_22":  "dot",
    "S3_optimized":        "solid",
}
_STRATEGY_FALLBACK_DASH = "longdash"


def _color_for(strategy_name: str) -> str:
    return _STRATEGY_COLORS.get(strategy_name, _STRATEGY_FALLBACK_COLOR)


def _marker_for(strategy_name: str) -> str:
    return _STRATEGY_MARKER.get(strategy_name, _STRATEGY_FALLBACK_MARKER)


def _dash_for(strategy_name: str) -> str:
    return _STRATEGY_DASH.get(strategy_name, _STRATEGY_FALLBACK_DASH)


def _x_for(df: pd.DataFrame) -> pd.Series:
    """Pick the x-axis column shared across multi-strategy traces.

    Every strategy was simulated against the same household so the
    `spouse_a_age` column is identical across strategies — using it
    everywhere keeps the x-axis aligned and lets `hovermode="x
    unified"` group all four strategies' values into a single
    tooltip per year.
    """
    if "spouse_a_age" in df.columns:
        return df["spouse_a_age"]
    return df.index


def _stable_strategy_order(names: list[str]) -> list[str]:
    """Sort canonical strategies S0→S3, then everything else last.

    The runner emits strategies in insertion order, which already
    matches S0→S3 today, but pinning the order here means
    multi-strategy charts stay visually consistent even if the
    runner's iteration order changes (or if a custom run produces
    a different mix).
    """
    canonical = ["S0_baseline", "S1_all_roth_401k",
                 "S2_bracket_fill_22", "S3_optimized"]
    pinned = [n for n in canonical if n in names]
    extras = [n for n in names if n not in canonical]
    return pinned + extras


# Line widths: the winner gets a slightly heavier stroke so it
# reads as the verdict at a glance, while the alternatives stay
# at a thinner default. Bumping the non-winner lines from 2 → 2.25
# also makes the curves easier to follow in the dense overlap
# regions (e.g. S2 vs S3 conversion plateau).
_LINE_WIDTH_DEFAULT: float = 2.25
_LINE_WIDTH_WINNER: float = 3.25
# Marker size on top of each line. Small enough not to dominate
# the line but big enough to pin individual years (especially
# around the RMD ramp-up at age 75 and the conversion ramp-down
# around age 65).
_MARKER_SIZE: float = 4.5


def _add_strategy_lines(
    fig: go.Figure,
    strategies: dict[str, dict[str, Any]],
    *,
    column: str,
    metric_label: str,
    row: int,
    col: int,
    y_scale: float = 1.0,
    y_format: str = "$%{y:,.0f}",
    legendgroup_per_strategy: bool = True,
    showlegend_for_first_row: bool = True,
    winner_name: str | None = None,
) -> None:
    """Overlay one line per strategy on a subplot.

    `column` selects which DataFrame column to plot; `y_scale` lets
    callers convert e.g. fractions to percentages (`marginal` is
    stored as 0.24, but we render as 24%). `legendgroup` ties all
    of a strategy's traces together so clicking the legend toggles
    every metric for that strategy at once.

    If `winner_name` is provided, that strategy's line is drawn
    slightly thicker so the verdict reads at a glance — non-winner
    strategies stay at the default stroke. The winner trace is
    also rendered LAST (drawn on top) so it isn't visually buried
    under whichever strategy it happens to track closely.

    Color-blindness handling: every strategy is rendered with
    *three* redundant cues — Okabe-Ito hue (`_color_for`), a
    distinct marker symbol (`_marker_for`), and a distinct line
    dash pattern (`_dash_for`). Even under achromatopsia (total
    color blindness) or on a B&W printer, the four strategies
    remain distinguishable: solid+circle vs dash+square vs
    dot+diamond vs solid+triangle.
    """
    # Render order: non-winners first, winner last so the heavier
    # stroke sits on top of the alternatives.
    ordered = _stable_strategy_order(list(strategies.keys()))
    if winner_name and winner_name in ordered:
        ordered = [n for n in ordered if n != winner_name] + [winner_name]

    for name in ordered:
        s = strategies[name]
        df = s["df"] if isinstance(s.get("df"), pd.DataFrame) else None
        if df is None:
            # Caller passed a serialized payload; deserialize lazily
            # so the figure module stays import-light (no Dash deps).
            from .runner import deserialize_strategy_df  # local import
            df = deserialize_strategy_df(s["df"])
        if column not in df.columns or len(df) == 0:
            continue
        x = _x_for(df)
        y = df[column].astype(float) * y_scale
        is_winner = (name == winner_name)
        width = _LINE_WIDTH_WINNER if is_winner else _LINE_WIDTH_DEFAULT
        line_kwargs: dict[str, Any] = dict(
            color=_color_for(name),
            width=width,
            dash=_dash_for(name),
        )
        # `lines+markers` so individual years are visible — at 30+
        # year horizons the markers also serve as visual anchors
        # when several strategies overlap on the same path
        # (e.g. S0 and S1 both flat at zero in the conversion
        # panel: the markers preserve the "two strategies stacked
        # here" cue even though the lines are coincident).
        fig.add_trace(
            go.Scatter(
                x=x, y=y,
                name=name,
                mode="lines+markers",
                line=line_kwargs,
                marker=dict(
                    size=_MARKER_SIZE,
                    color=_color_for(name),
                    symbol=_marker_for(name),
                ),
                legendgroup=name if legendgroup_per_strategy else None,
                showlegend=showlegend_for_first_row,
                hovertemplate=f"<b>{name}</b><br>{metric_label}: {y_format}<extra></extra>",
            ),
            row=row, col=col,
        )


def multi_strategy_taxes_panel(
    strategies: dict[str, dict[str, Any]],
    *,
    title: str = "Taxes & marginal bracket — strategy comparison",
    winner_name: str | None = None,
) -> go.Figure:
    """Overlay AGI / federal / state / IRMAA / marginal across strategies.

    Five stacked subplots (one per metric, all sharing the same
    x-axis and one color per strategy) so the user can answer
    "which strategy's federal tax curve dips earliest?" by
    glancing at one panel rather than playing strategy-roulette
    in the dropdown.

    Subplots that have no signal across any strategy (e.g. state
    tax for a stateless household, IRMAA for a young household)
    are silently skipped to keep the figure compact.

    Layout decisions tuned for readability:

    * Metric label lives on the **y-axis title**, not as a
      subplot annotation. Subplot annotations sit at y≈1.0 of
      each panel and routinely crash into trace data when AGI
      / federal tax peak in late retirement; y-axis titles
      sit in the left margin where there's always room.
    * Generous `vertical_spacing` (0.085) so adjacent panels
      stay distinct without an extra divider per panel.
    * Per-panel height bumped to 220px (was 180px) so a
      typical 30+ year curve has enough vertical resolution
      to read year-over-year deltas.
    * Winner line gets a slightly heavier stroke (handled in
      `_add_strategy_lines`) so the verdict reads instantly.
    """
    if not strategies:
        return empty_figure("Run the simulator to populate strategies")

    # Deserialize once per strategy to decide which subplots actually
    # have signal (so we don't waste vertical real estate on a flat
    # zero-line for state tax in stateless runs).
    from .runner import deserialize_strategy_df  # local import
    rendered: dict[str, pd.DataFrame] = {}
    for name, s in strategies.items():
        df = s["df"] if isinstance(s.get("df"), pd.DataFrame) else (
            deserialize_strategy_df(s["df"])
        )
        rendered[name] = df

    def _any_nonzero(col: str) -> bool:
        return any(
            col in df.columns and bool(df[col].abs().sum() > 0)
            for df in rendered.values()
        )

    # Build the active subplot list. The order here is the visual
    # order top-to-bottom; "headline" metrics (AGI, federal tax)
    # always render so the panel is never empty.
    panels: list[tuple[str, str, str, float, str]] = [
        # (column, label, hover_format, y_scale, y_axis_title)
        ("agi", "AGI", "$%{y:,.0f}", 1.0, "AGI ($)"),
        ("federal_tax", "Federal tax", "$%{y:,.0f}", 1.0, "Federal tax ($)"),
    ]
    if _any_nonzero("state_tax"):
        panels.append(("state_tax", "State tax", "$%{y:,.0f}", 1.0, "State tax ($)"))
    if _any_nonzero("irmaa"):
        panels.append(("irmaa", "IRMAA", "$%{y:,.0f}", 1.0, "IRMAA ($)"))
    panels.append(("marginal", "Marginal", "%{y:.0f}%", 100.0, "Marginal (%)"))

    fig = make_subplots(
        rows=len(panels), cols=1, shared_xaxes=True,
        # 0.085 (was 0.045) is enough vertical space that the
        # panels read as distinct cards without needing
        # subplot annotations to mark the boundary.
        vertical_spacing=0.085,
    )

    # Light-weight payload for `_add_strategy_lines` so it doesn't
    # pay the deserialization cost twice.
    light_payload = {n: {"df": df} for n, df in rendered.items()}

    for idx, (column, label, hover_fmt, y_scale, y_title) in enumerate(panels):
        is_first = idx == 0  # only show the legend once
        _add_strategy_lines(
            fig, light_payload,
            column=column, metric_label=label,
            row=idx + 1, col=1,
            y_scale=y_scale,
            y_format=hover_fmt,
            showlegend_for_first_row=is_first,
            winner_name=winner_name,
        )
        # Per-row y-axis formatting + the metric label as the
        # axis title (replaces the old subplot annotation).
        if column == "marginal":
            fig.update_yaxes(
                title_text=y_title, **_AXIS_PCT, row=idx + 1, col=1
            )
        else:
            fig.update_yaxes(
                title_text=y_title, **_AXIS_DOLLAR, row=idx + 1, col=1
            )

    fig.update_xaxes(title_text="Spouse A age", row=len(panels), col=1)
    # 220px per panel (was 180) plus 100px of chrome (chart
    # title + horizontal legend at the bottom). At 5 panels this
    # is ~1200px total — pushes scrolling on small viewports
    # but at this density we'd rather give each curve room to
    # breathe than cram them into a single screen.
    height = 220 * len(panels) + 100
    # Wider left margin (l=80, default in `_LAYOUT` is 60) to
    # make room for the dollar-amount tick labels + y-axis title
    # on each panel.
    layout = {**_LAYOUT, "margin": dict(l=80, r=20, t=60, b=60)}
    fig.update_layout(title=title, height=height, **layout)
    return fig


def multi_strategy_conversion_panel(
    strategies: dict[str, dict[str, Any]],
    *,
    title: str = "Roth conversions & RMDs — strategy comparison",
    winner_name: str | None = None,
) -> go.Figure:
    """Overlay Roth-conversion + RMD timelines across strategies.

    Two stacked subplots:

    * **Roth conversion (annual $)** — one line per strategy. The
      shape of this curve is the optimizer's signature move
      (typically a flat plateau during the gap years that fades
      out at RMD age), so seeing all four overlaid makes
      "did the optimizer convert more aggressively than S2?"
      a one-glance question.
    * **RMD (annual $)** — one line per strategy. The strategies
      diverge here precisely *because* of the conversion choice:
      heavy converters end up with smaller RMDs late in
      retirement (less pretax left to distribute), so the RMD
      panel is the reverse of the conversion panel for the same
      strategies.

    A small marker is dropped on conversion-curve points where
    the liquidity guard trimmed the conversion (any strategy
    whose ``roth_conv_capped_by_liquidity`` is true that year).
    Surfaces the "the optimizer wanted to convert more but
    couldn't fund the tax bill" pattern visually.

    Layout decisions tuned for readability (mirroring the taxes
    panel):

    * Metric label is the **y-axis title**, not a subplot
      annotation — the previous annotation-based titles
      ("Roth conversion (annual $)") were colliding with the
      conversion plateau values around $80k.
    * `vertical_spacing` bumped from 0.07 → 0.10 so the two
      panels are visually distinct.
    * Total height bumped from 560 → 660 px so each subplot
      gets ~290 px of vertical real estate (was ~245 px).
    """
    if not strategies:
        return empty_figure("Run the simulator to populate strategies")

    from .runner import deserialize_strategy_df  # local import

    rendered: dict[str, pd.DataFrame] = {}
    for name, s in strategies.items():
        df = s["df"] if isinstance(s.get("df"), pd.DataFrame) else (
            deserialize_strategy_df(s["df"])
        )
        rendered[name] = df

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        # Slightly emphasize the conversion panel — it's the
        # "decision" plot. RMDs are the *consequence*, so they
        # earn slightly less vertical real estate.
        row_heights=[0.55, 0.45],
        vertical_spacing=0.10,
    )

    light_payload = {n: {"df": df} for n, df in rendered.items()}

    _add_strategy_lines(
        fig, light_payload,
        column="roth_conversion", metric_label="Conv",
        row=1, col=1,
        showlegend_for_first_row=True,
        winner_name=winner_name,
    )
    _add_strategy_lines(
        fig, light_payload,
        column="rmd", metric_label="RMD",
        row=2, col=1,
        # Don't double-render the legend — strategies share their
        # legendgroup across rows so a single click in row 1 toggles
        # both panels.
        showlegend_for_first_row=False,
        winner_name=winner_name,
    )

    # Liquidity-cap markers: small ✕ on each strategy's curve at any
    # year that strategy was capped. Per-strategy color so the user
    # can tell which strategy was constrained.
    for name in _stable_strategy_order(list(rendered.keys())):
        df = rendered[name]
        if "roth_conv_capped_by_liquidity" not in df.columns:
            continue
        if "roth_conversion" not in df.columns:
            continue
        cap_mask = df["roth_conv_capped_by_liquidity"].astype(bool).fillna(False)
        if not cap_mask.any():
            continue
        x = _x_for(df)
        fig.add_trace(
            go.Scatter(
                x=x[cap_mask],
                y=df.loc[cap_mask, "roth_conversion"],
                name=f"{name} (liquidity-capped)",
                mode="markers",
                marker=dict(symbol="x", size=11, color=_color_for(name),
                            line=dict(width=1.5, color="#0f172a")),
                legendgroup=name,
                showlegend=False,
                hovertemplate=(
                    f"<b>{name}</b><br>Capped by liquidity<extra></extra>"
                ),
            ),
            row=1, col=1,
        )

    fig.update_yaxes(
        title_text="Roth conversion ($)", **_AXIS_DOLLAR, row=1, col=1
    )
    fig.update_yaxes(title_text="RMD ($)", **_AXIS_DOLLAR, row=2, col=1)
    fig.update_xaxes(title_text="Spouse A age", row=2, col=1)
    # Wider left margin to make room for the y-axis titles and
    # dollar tick labels (overrides the narrower `_LAYOUT.margin`).
    layout = {**_LAYOUT, "margin": dict(l=80, r=20, t=60, b=60)}
    fig.update_layout(title=title, height=660, **layout)
    return fig


def multi_strategy_growth_panel(
    strategies: dict[str, dict[str, Any]],
    *,
    heir_marginal_rate: float = 0.22,
    title: str = "Liquid NW after tax & year-over-year growth",
    winner_name: str | None = None,
) -> go.Figure:
    """Overlay after-tax NW + YoY growth across all strategies.

    Two stacked subplots, mirroring the multi-strategy taxes /
    conversions pattern:

    * **Top — Liquid NW after tax ($)**: per-year application of
      the bequest-tax-aware terminal formula
      (`pretax × (1 - rate) + roth + taxable + hsa × (1 - rate)`)
      so the chart line *is* the same number the Terminal NW
      tile shows, traced through every year.
    * **Bottom — YoY growth (%)**: pct-change of the same series
      with a horizontal reference line at 0 %. Decumulation
      years go negative — surfacing this visually matches the
      "Decumulation CAGR" KPI tile.

    Each strategy uses the same Okabe-Ito hue + marker + dash
    redundancy as the other multi-strategy charts; the winner
    gets the heavier stroke and is drawn last (handled by
    `_add_strategy_lines`).
    """
    if not strategies:
        return empty_figure("Run the simulator to populate strategies")

    from .runner import deserialize_strategy_df  # local import

    # Compute the derived columns once per strategy so
    # `_add_strategy_lines` can read them by name.
    enriched: dict[str, pd.DataFrame] = {}
    for name, s in strategies.items():
        df = s["df"] if isinstance(s.get("df"), pd.DataFrame) else (
            deserialize_strategy_df(s["df"])
        )
        if df is None or len(df) == 0:
            continue
        # Defensive copy so we don't mutate caller-owned frames
        # (the runner caches DataFrames and would propagate the
        # extra columns on re-renders).
        df = df.copy()
        from tax_optimizer.metrics import nw_after_tax_series  # local import
        nw = nw_after_tax_series(df, heir_marginal_rate=heir_marginal_rate)
        df["_nw_after_tax"] = nw
        # `pct_change()` on the first row is NaN — Plotly skips
        # NaN points, which is exactly what we want (no spurious
        # "infinite growth in year 0" spike).
        df["_nw_yoy_pct"] = nw.pct_change() * 100.0
        enriched[name] = df

    if not enriched:
        return empty_figure("No simulated balances available for growth chart")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        # Match the conversions panel: the "headline" trace gets
        # slightly more vertical room than the derivative.
        row_heights=[0.58, 0.42],
        vertical_spacing=0.10,
    )

    light_payload = {n: {"df": df} for n, df in enriched.items()}

    _add_strategy_lines(
        fig, light_payload,
        column="_nw_after_tax", metric_label="After-tax NW",
        row=1, col=1,
        y_format="$%{y:,.0f}",
        showlegend_for_first_row=True,
        winner_name=winner_name,
    )
    _add_strategy_lines(
        fig, light_payload,
        column="_nw_yoy_pct", metric_label="YoY growth",
        row=2, col=1,
        y_format="%{y:+.1f}%",
        showlegend_for_first_row=False,
        winner_name=winner_name,
    )

    # 0% reference line for the YoY panel — separates accumulation
    # (positive) from decumulation (negative). Drawn as a shape so
    # it doesn't pollute the legend.
    fig.add_hline(
        y=0,
        line=dict(color="#94a3b8", width=1, dash="dot"),
        row=2, col=1,
    )

    fig.update_yaxes(
        title_text="Liquid NW after tax ($)", **_AXIS_DOLLAR, row=1, col=1
    )
    fig.update_yaxes(
        title_text="YoY growth (%)", **_AXIS_PCT, row=2, col=1,
    )
    fig.update_xaxes(title_text="Spouse A age", row=2, col=1)
    # `cliponaxis=False` on every trace so end-of-horizon points
    # don't get clipped at the panel edge — matters when the YoY
    # values trend toward zero late in the plan and we want the
    # final marker visible.
    fig.update_traces(cliponaxis=False)
    layout = {**_LAYOUT, "margin": dict(l=80, r=20, t=60, b=60)}
    fig.update_layout(title=title, height=640, **layout)
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

    # Bar value labels use abbreviated dollar formatting
    # ("$17.4M" not "$17,454,663") so they fit alongside the bar
    # without being clipped at the axis edge. Without this, the
    # default `textposition="outside"` consistently truncated the
    # labels in tight viewports — see CHANGELOG entry "Fixed —
    # Strategies tab: bar value labels truncated".
    text = []
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            text.append("")
        elif baseline:
            pct = (v - baseline) / max(abs(baseline), 1) * 100
            text.append(f"{_abbrev_dollars(v)}  ({pct:+.1f}%)")
        else:
            text.append(_abbrev_dollars(v))

    # Pad the x-axis so outside labels have somewhere to go.
    numeric_values = [
        v for v in values
        if v is not None and not (isinstance(v, float) and np.isnan(v))
    ]
    max_val = max(numeric_values) if numeric_values else 0.0

    fig = go.Figure(
        go.Bar(
            x=values, y=names, orientation="h",
            marker_color=[_color_for(n) for n in names],
            text=text,
            # `auto` = inside if the label fits, outside otherwise
            # (vs. `outside` which always pushes the label past
            # the bar end and routinely overflowed the axis).
            textposition="auto",
            textfont=dict(size=12),
            # Keep outside labels visible even when they spill
            # past the axis range — defensive against tiny
            # bars (e.g. zero rows) where the label has to live
            # outside the bar.
            cliponaxis=False,
            hovertemplate="%{y}: $%{x:,.0f}<extra></extra>",
        )
    )
    xaxis_kwargs = dict(_AXIS_DOLLAR)
    if max_val > 0:
        # 22% headroom — bigger than the 18% used by the
        # multi-subplot panel because here the label also
        # carries a "(+X.X%)" suffix.
        xaxis_kwargs["range"] = [0, max_val * 1.22]
    fig.update_layout(
        title=dict(text=f"{title} - {metric}", font=dict(size=15)),
        xaxis=xaxis_kwargs,
        yaxis=dict(autorange="reversed"),
        **_LAYOUT,
        height=300,
    )
    return fig


def strategy_compare_panel(
    strategies: dict[str, dict[str, Any]],
) -> go.Figure:
    """3-panel view of terminal NW, lifetime tax NPV, and lifetime IRMAA NPV.

    Layout decisions tuned to keep the bar value labels readable:

    * **Abbreviated dollar text** (`_abbrev_dollars`) — full
      precision (`$17,454,663`) doesn't fit alongside the bars
      at typical Strategies-tab widths and was getting clipped
      at the subplot edges. `$17.4M` cuts label width by ~60%.
    * **`textposition="auto"`** so each label sits inside the
      bar when it fits, outside when not — vs. the previous
      `"outside"` which always pushed the label past the bar
      end and routinely overflowed.
    * **X-axis range padded 18%** beyond the largest value so
      outside labels have somewhere to go.
    * **`cliponaxis=False`** as a defensive belt-and-suspenders
      against very small bars whose label *has* to render
      outside.
    * Larger `horizontal_spacing` (0.10) so subplot titles like
      "Lifetime federal tax (NPV)" don't crash into adjacent
      subplots.
    """
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
        horizontal_spacing=0.10,
    )
    for i, (key, _label) in enumerate(metrics, start=1):
        values = [strategies[n]["summary"].get(key, 0.0) for n in names]
        max_val = max(values) if values else 0.0
        fig.add_trace(
            go.Bar(
                x=values, y=names, orientation="h", showlegend=False,
                marker_color=[_color_for(n) for n in names],
                text=[_abbrev_dollars(v) for v in values],
                textposition="auto",
                textfont=dict(size=12),
                cliponaxis=False,
                hovertemplate="%{y}: $%{x:,.0f}<extra></extra>",
            ),
            row=1, col=i,
        )
        # Per-subplot x-axis range with 18% headroom for any
        # label that ends up rendering outside the bar.
        xaxis_kwargs = dict(_AXIS_DOLLAR)
        if max_val > 0:
            xaxis_kwargs["range"] = [0, max_val * 1.18]
        fig.update_xaxes(**xaxis_kwargs, row=1, col=i)

    # Bump the subplot title font (`make_subplots` exposes them
    # as layout annotations) so they read at the same size as
    # the table column headers above.
    for ann in fig.layout.annotations:
        ann.font = dict(size=13)

    fig.update_layout(
        title=dict(text="Strategy comparison", font=dict(size=15)),
        yaxis=dict(autorange="reversed"),
        **_LAYOUT,
        height=400,
    )
    return fig


# ---------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------


def _abbrev_dollars(v: float) -> str:
    """Abbreviate large dollar amounts so they fit in chart annotations.

    `$5,826,124` → `$5.8M`, `$43,815,711` → `$43.8M`,
    `$1,234,000,000` → `$1.2B`, `$12,300` → `$12K`.

    Used by `mc_terminal_histogram` to keep the P10/P50/P90
    callouts compact — at full precision the labels overlap in
    typical Monte Carlo runs (see screenshot in the changelog).
    Sub-thousand values fall through to the standard `$1,234`
    format because at that magnitude there's no compression to
    win and abbreviating "$0" → "$0K" would be misleading.
    """
    a = abs(v)
    if a >= 1_000_000_000:
        return f"${v/1_000_000_000:.1f}B"
    if a >= 1_000_000:
        return f"${v/1_000_000:.1f}M"
    if a >= 10_000:
        return f"${v/1_000:.0f}K"
    return f"${v:,.0f}"


def mc_terminal_histogram(mc_payload: dict[str, Any] | None) -> go.Figure:
    if not mc_payload:
        return empty_figure("Monte Carlo not run")
    terminals = np.asarray(mc_payload["terminals"], dtype=float)
    pcts = mc_payload["percentiles"]
    fig = go.Figure(
        go.Histogram(
            x=terminals, nbinsx=40,
            marker_color=_OKABE_ITO["blue"], opacity=0.85,
            hovertemplate="Terminal: $%{x:,.0f}<br>Paths: %{y}<extra></extra>",
        )
    )
    # Percentile guide lines + annotations.
    #
    # Color: vermillion (worst case) → black (median) →
    # bluish-green (best case). Okabe-Ito hues, all three
    # CB-distinguishable and luminance-distinct.
    #
    # Anti-overlap layout: the leftmost label (P10) anchors to
    # the LEFT of its line, the rightmost (P90) to the RIGHT,
    # and the middle (P50) is CENTERED above its line. The
    # three different horizontal anchor directions push the
    # labels into three different visual zones, so the
    # callouts never collide for any plausible percentile
    # set.
    #
    # **Title clearance**: every callout sits at the top of
    # the plot area (``yshift=0``) — INSIDE the chart, not
    # above it. An earlier version pushed the P50 callout up
    # by ``yshift=22`` to give it a separate "row" from the
    # side labels, but that put it in the chart-title margin
    # band where it visibly collided with the title text. We
    # rely on the horizontal-anchor scheme alone for overlap
    # protection now.
    #
    # We also abbreviate the dollar values via
    # ``_abbrev_dollars`` ("$5.8M" instead of "$5,826,124") —
    # cuts label width by ~60% and gives the layout extra
    # headroom for unusually-clustered percentile sets.
    #
    # ``bgcolor`` at 88 % white on each annotation prevents
    # the text from disappearing into the histogram bars
    # behind it.
    percentile_styles = (
        ("p10", _OKABE_ITO["vermillion"],   "top left",   0),
        ("p50", _OKABE_ITO["black"],        "top",        0),
        ("p90", _OKABE_ITO["bluish_green"], "top right",  0),
    )
    for tag, color, position, yshift in percentile_styles:
        v = pcts[tag]
        fig.add_vline(
            x=v,
            line=dict(color=color, dash="dash"),
            annotation_text=f"{tag.upper()}: {_abbrev_dollars(v)}",
            annotation_position=position,
            annotation_yshift=yshift,
            annotation_bgcolor="rgba(255,255,255,0.88)",
            annotation_bordercolor=color,
            annotation_borderwidth=1,
            annotation_borderpad=3,
            annotation_font=dict(size=11, color=color),
        )
    # Extra top margin so the percentile callouts at the top
    # of the plot area have visible breathing room beneath the
    # chart title.
    layout = {**_LAYOUT, "margin": dict(l=60, r=20, t=80, b=50)}
    fig.update_layout(
        title=(
            f"Terminal NW distribution"
            f" - p_success={mc_payload['prob_success']:.0%},"
            f" CVaR(10%)={_abbrev_dollars(mc_payload['cvar_terminal'])}"
        ),
        xaxis_title="Terminal after-tax NW",
        yaxis_title="Paths",
        xaxis=_AXIS_DOLLAR,
        **layout,
    )
    return fig


def mc_fan_chart(mc_payload: dict[str, Any] | None) -> go.Figure:
    if not mc_payload or not mc_payload.get("fan"):
        return empty_figure("Monte Carlo fan unavailable")
    fan = mc_payload["fan"]
    x = fan["year_offset"]
    fig = go.Figure()
    # P10-P90 band (Okabe-Ito blue at 18% alpha)
    fig.add_trace(
        go.Scatter(
            x=x + x[::-1],
            y=fan["p90"] + fan["p10"][::-1],
            fill="toself",
            fillcolor="rgba(0, 114, 178, 0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="P10-P90",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=fan["p50"], mode="lines", name="P50 (median)",
            line=dict(color=_OKABE_ITO["black"], width=2),
            hovertemplate="Year %{x}: $%{y:,.0f}<extra></extra>",
        )
    )
    # P10 (worst case) → vermillion + dot dash
    # P90 (best case) → bluish-green + dash dash
    # Distinct dash patterns are the non-color cue: even in
    # monochrome the user can tell which percentile is which.
    fig.add_trace(
        go.Scatter(
            x=x, y=fan["p10"], mode="lines", name="P10",
            line=dict(color=_OKABE_ITO["vermillion"], width=1.5, dash="dot"),
            hovertemplate="P10 year %{x}: $%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=fan["p90"], mode="lines", name="P90",
            line=dict(color=_OKABE_ITO["bluish_green"], width=1.5, dash="dash"),
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


# ---------------------------------------------------------------------
# Overview-tab KPI tile help text
# ---------------------------------------------------------------------
#
# Every tile gets a one-or-two-sentence explanation that surfaces
# via the same ⓘ icon + Bootstrap tooltip pattern used by the
# scenario form fields (see `dash_app/layout.py:_help_components`).
# Co-locating the hint dictionary with the tile builder keeps
# label / value / hint in lockstep — adding a new tile requires
# adding the hint here, not in a separate i18n file.
_OVERVIEW_TILE_HINTS: dict[str, str] = {
    # ---- Outcomes section ----------------------------------------
    "Terminal after-tax NW": (
        "Bequest-tax-aware net worth at the end of the plan "
        "(spouse_a_age == horizon_age, default 90). Pretax + HSA "
        "are discounted by the heir's marginal rate "
        "(cfg.heir_marginal_rate, default 22%); Roth is tax-free; "
        "taxable is at face value (heirs receive a step-up in "
        "basis at death). "
        "Formula: pretax × (1 − rate) + roth + taxable + hsa × (1 − rate)."
    ),
    "Lifetime federal tax (NPV)": (
        "Net present value of every year's federal income tax "
        "across the horizon, discounted at 2.5%/yr. Lower is better — "
        "this is the tile the optimizer's terminal-NW objective "
        "implicitly drives down."
    ),
    "Lifetime IRMAA (NPV)": (
        "NPV of Medicare IRMAA premium surcharges across the "
        "horizon (discount 2.5%/yr). Surcharges kick in for "
        "Medicare enrollees whose AGI from two years prior crosses "
        "an IRMAA tier — large Roth conversions can push you across "
        "a tier and inflate this number."
    ),
    "Peak marginal rate": (
        "Highest year-end marginal federal bracket the plan ever "
        "touches. Mirrors the bracket-fill ceiling — e.g. an S2 "
        "(fill-to-22%) plan should peak at 22%, while an "
        "unbounded plan can peak much higher in late RMD years."
    ),
    "Years with IRMAA": (
        "Number of years during which an IRMAA surcharge is paid "
        "by either spouse. Uses the standard 2-year AGI lookback. "
        "Zero is ideal; > 0 means you're crossing a tier in some year."
    ),
    "Probability of success": (
        "Fraction of Monte Carlo paths that finish without ever "
        "running out of liquid net worth. \"Success\" is the "
        "complement of \"ruin\" — see `MonteCarloResult.prob_success`."
    ),
    "CVaR (10%)": (
        "Conditional value-at-risk at the 10% tail: the average "
        "terminal NW across the worst 10% of Monte Carlo paths. "
        "Tells you \"how bad is bad?\" — a P10 percentile gives one "
        "point, CVaR averages everything below it."
    ),
    # ---- Growth section ------------------------------------------
    "Starting after-tax NW": (
        "Bequest-tax-aware starting net worth — same lens as the "
        "Terminal-NW tile, applied to today's StartingBalances. "
        "Using the same lens on both ends matters: otherwise the "
        "growth ratio bundles a methodology change in with the "
        "actual growth."
    ),
    "Total growth": (
        "Terminal NW divided by Starting NW. A multiplier (e.g. "
        "3.15× = ended with 3.15 times the starting after-tax NW). "
        "Returns \"-\" if Starting NW is zero or negative."
    ),
    "Effective CAGR": (
        "Compound annual growth rate of after-tax NW across the "
        "full horizon. Caveat: this is NOT a pure investment "
        "return — it bundles market returns + contributions + "
        "withdrawals + tax drag into one effective compounding "
        "rate. The accumulation-vs-decumulation split below "
        "separates \"growth\" from \"drawdown\"."
    ),
    "Real CAGR": (
        "Effective CAGR adjusted for inflation. Terminal NW is "
        "first discounted by (1 + cfg.inflation)^years (today's "
        "dollars), then compared to the starting NW. Rule of "
        "thumb for small rates: real ≈ nominal − inflation "
        "(Fisher equation)."
    ),
    "Accumulation CAGR": (
        "CAGR for the years before retirement "
        "(spouse_a_age < spouse_a_retire_age). Usually positive — "
        "you're contributing AND markets are growing the balance."
    ),
    "Decumulation CAGR": (
        "CAGR for the years from retirement onward "
        "(spouse_a_age >= spouse_a_retire_age). Often negative — "
        "the whole point of the decumulation phase is to draw "
        "down liquid NW. A negative number here is healthy; a "
        "deeply-negative number means you're drawing down faster "
        "than markets can keep up."
    ),
}


def _hint_for(label: str) -> str:
    """Look up the tooltip for a tile label.

    Returns ``""`` (empty string, NOT ``None``) for an unknown
    label so the renderer can pass the value to the DOM unchanged
    without dealing with optionals. New tiles must add an entry
    to ``_OVERVIEW_TILE_HINTS``.
    """
    return _OVERVIEW_TILE_HINTS.get(label, "")


def overview_kpis(
    summary: dict[str, Any], mc_payload: dict[str, Any] | None
) -> list[tuple[str, list[tuple[str, str, str]]]]:
    """Sectioned KPI data for the Overview tab.

    Returns ``[(section_label, [(tile_label, tile_value, tile_hint), ...]), ...]``.

    Each tile entry is a 3-tuple ``(label, value, hint)`` — the
    hint is the help text that surfaces via the ⓘ-icon tooltip
    next to the tile's label. Hints come from
    ``_OVERVIEW_TILE_HINTS`` so adding a tile requires touching
    one place.

    **Outcomes** — bottom-line plan results: terminal after-tax NW,
    lifetime tax / IRMAA NPV, peak marginal, MC stats. The
    "Terminal after-tax NW" tile reflects the bequest-tax-aware
    formula in `tax_optimizer.metrics.terminal_after_tax_nw`:

        pretax × (1 - heir_marginal_rate)   # heirs pay ordinary tax
        + roth                                # tax-free
        + taxable                             # step-up basis at death
        + hsa × (1 - heir_marginal_rate)      # ordinary tax to heirs

    **Growth** — how the plan got from `Starting NW` to `Terminal
    NW`: total growth multiplier, effective (nominal) and real
    CAGRs, and the accumulation-vs-decumulation phase split. The
    CAGR tiles use the household's full balance-sheet timeline,
    so they bundle market returns + contributions + withdrawals +
    tax drag — they're an effective compounding rate, not a pure
    investment return. Decumulation CAGR is often negative (the
    plan is meant to draw down).
    """
    def _tile(label: str, value: str) -> tuple[str, str, str]:
        return (label, value, _hint_for(label))

    outcomes: list[tuple[str, str, str]] = []
    outcomes.append(
        _tile("Terminal after-tax NW",
              _fmt_dollars(summary.get("terminal_after_tax")))
    )
    outcomes.append(
        _tile("Lifetime federal tax (NPV)",
              _fmt_dollars(summary.get("lifetime_tax_npv")))
    )
    outcomes.append(
        _tile("Lifetime IRMAA (NPV)",
              _fmt_dollars(summary.get("lifetime_irmaa_npv")))
    )
    pm = summary.get("peak_marginal")
    outcomes.append(
        _tile("Peak marginal rate",
              f"{pm * 100:.0f}%" if pm is not None else "-")
    )
    yi = summary.get("years_irmaa")
    outcomes.append(
        _tile("Years with IRMAA", f"{int(yi)}" if yi is not None else "-")
    )
    if mc_payload:
        outcomes.append(
            _tile("Probability of success",
                  f"{mc_payload['prob_success']:.0%}")
        )
        outcomes.append(
            _tile("CVaR (10%)", _fmt_dollars(mc_payload["cvar_terminal"]))
        )

    growth: list[tuple[str, str, str]] = []
    growth.append(
        _tile("Starting after-tax NW",
              _fmt_dollars(summary.get("starting_after_tax")))
    )
    growth.append(
        _tile("Total growth",
              _fmt_multiplier(summary.get("total_growth_mult")))
    )
    growth.append(
        _tile("Effective CAGR", _fmt_cagr(summary.get("effective_cagr")))
    )
    growth.append(
        _tile("Real CAGR", _fmt_cagr(summary.get("real_cagr")))
    )
    growth.append(
        _tile("Accumulation CAGR",
              _fmt_cagr(summary.get("accumulation_cagr")))
    )
    growth.append(
        _tile("Decumulation CAGR",
              _fmt_cagr(summary.get("decumulation_cagr")))
    )

    return [("Outcomes", outcomes), ("Growth", growth)]


def _fmt_dollars(v: Any) -> str:
    if v is None:
        return "-"
    try:
        return f"${float(v):,.0f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_multiplier(v: Any) -> str:
    """Format a growth multiplier like 3.15 → "3.15×"."""
    if v is None:
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "-"
    if f != f:  # NaN check
        return "-"
    return f"{f:.2f}\u00d7"


def _fmt_cagr(v: Any) -> str:
    """Format a CAGR fraction like 0.042 → "+4.2%/yr"."""
    if v is None:
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "-"
    if f != f:  # NaN check
        return "-"
    return f"{f * 100:+.1f}%/yr"


# ---------------------------------------------------------------------
# Year-by-year detail table — column groups, ordering, and styling
# ---------------------------------------------------------------------
#
# The Year-by-year tab's drill-down DataTable used to be a flat list
# of 22 column names whose order had drifted into "approximate
# functional groups separated by source-line breaks". The table
# below promotes that implicit grouping to a first-class data
# structure so we can:
#
#   1. Generate `detail_columns()` (the column-filter list applied
#      against the simulator output) directly from the groups, so
#      adding a column = editing one tuple.
#   2. Generate per-column DataTable conditional styles — light cell
#      tints + medium header tints + a left-border divider on the
#      first column of each group — directly from the same source
#      of truth, keeping color and order in lockstep.
#
# Each entry is `(group_id, color_key_or_None, columns)`:
#
#   - `group_id`     — internal identifier; surfaces in test names.
#   - `color_key`    — key into `_OKABE_ITO`, or `None` for the
#                      "identity" group which keeps the default
#                      slate header (no group hue).
#   - `columns`      — the user-facing column names (post-rename),
#                      in display order. `bracket_pct` is the
#                      DataTable-side rename of the simulator-side
#                      `marginal` (see `_DISPLAY_TO_NATIVE` and
#                      `filter_to_detail_cols`).
_YEARLY_COLUMN_GROUPS: list[tuple[str, str | None, list[str]]] = [
    ("identity",  None,             ["year", "spouse_a_age", "filing_status"]),
    ("income",    "blue",           ["wages", "pension", "ssn",
                                     "annuity_taxable",
                                     "qualified_dividends", "interest_income"]),
    ("pretax",    "orange",         ["rmd", "roth_conversion"]),
    ("withdraw",  "yellow",         ["pretax_withdrawal", "roth_withdrawal",
                                     "taxable_withdrawal"]),
    ("spending",  "vermillion",     ["spending_need", "unfunded"]),
    ("tax",       "reddish_purple", ["agi", "taxable_income", "federal_tax",
                                     "state_tax", "bracket_pct"]),
    ("medicare",  "sky_blue",       ["medicare_base_premium", "irmaa"]),
    ("balances",  "bluish_green",   ["pretax_balance", "roth_balance",
                                     "taxable_balance", "hsa_balance"]),
]

# Display column → simulator-native column. Keeps `detail_columns()`
# in lockstep with what `filter_to_detail_cols` renames at runtime
# (`marginal × 100 → bracket_pct`). Grow this dict if more renames
# are added — every entry should have a paired transform in
# `filter_to_detail_cols`.
_DISPLAY_TO_NATIVE: dict[str, str] = {
    "bracket_pct": "marginal",
}


def detail_columns() -> list[str]:
    """Year-by-year drill-down columns in display order.

    Returns simulator-native column names (so the list can be used
    as a DataFrame filter). The post-rename display name
    `bracket_pct` is translated back to its native `marginal`
    via `_DISPLAY_TO_NATIVE` here — `filter_to_detail_cols`
    performs the actual rename + percentage scaling at runtime.
    """
    cols: list[str] = []
    for _group_id, _color_key, group_cols in _YEARLY_COLUMN_GROUPS:
        for c in group_cols:
            cols.append(_DISPLAY_TO_NATIVE.get(c, c))
    return cols


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


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert ``"#0072B2"`` + ``0.08`` to ``"rgba(0, 114, 178, 0.08)"``.

    Used to derive subtly-tinted DataTable cell / header backgrounds
    from the canonical Okabe-Ito hex palette without having to
    pre-mix and store hand-picked pastels for every group.
    """
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"expected 6-digit hex color, got {hex_color!r}")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def yearly_table_styles() -> dict[str, list[dict[str, Any]]]:
    """Generate Dash DataTable conditional-style lists from the
    column-group declaration.

    The Year-by-year DataTable receives three conditional-style
    arrays:

    * ``style_header_conditional`` — one rule per colored column
      with a `rgba(<hue>, 0.22)` background. Strong enough that
      the eight group blocks are obvious at a glance, light
      enough that the column-name text stays readable.
    * ``style_data_conditional`` — one rule per colored column
      with a `rgba(<hue>, 0.08)` background. The light tint
      carries the group cue down through every data row without
      competing visually with the digit text.
    * ``style_cell_conditional`` — one rule per *first column of
      each non-identity group* with a `2px solid rgba(<hue>, 0.6)`
      left border. Vertical break between groups without needing
      empty separator columns (which would break sort/filter).

    Identity columns (year / spouse_a_age / filing_status) keep
    the default styling — they're the table's anchor and don't
    need a group cue.
    """
    style_header_conditional: list[dict[str, Any]] = []
    style_data_conditional: list[dict[str, Any]] = []
    style_cell_conditional: list[dict[str, Any]] = []

    for _group_id, color_key, group_cols in _YEARLY_COLUMN_GROUPS:
        if color_key is None:
            # Identity group: leave defaults alone.
            continue
        hue = _OKABE_ITO[color_key]
        cell_bg = _hex_to_rgba(hue, 0.08)
        header_bg = _hex_to_rgba(hue, 0.22)
        border_color = _hex_to_rgba(hue, 0.60)

        for idx, col in enumerate(group_cols):
            style_header_conditional.append({
                "if": {"column_id": col},
                "backgroundColor": header_bg,
            })
            style_data_conditional.append({
                "if": {"column_id": col},
                "backgroundColor": cell_bg,
            })
            if idx == 0:
                # Left-border divider on the first column of the
                # group. Applied to *both* header and data rows
                # via `style_cell_conditional` (which targets cells
                # in both header and body).
                style_cell_conditional.append({
                    "if": {"column_id": col},
                    "borderLeft": f"2px solid {border_color}",
                })

    return {
        "style_header_conditional": style_header_conditional,
        "style_data_conditional": style_data_conditional,
        "style_cell_conditional": style_cell_conditional,
    }
