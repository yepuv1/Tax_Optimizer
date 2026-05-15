"""Tests for the multi-strategy Taxes-tab figure builders.

The Taxes tab used to render two charts for the **winner only**.
``multi_strategy_taxes_panel`` and
``multi_strategy_conversion_panel`` overlay all four canonical
strategies (S0/S1/S2/S3) on the same axis so the user can compare
strategies in one glance instead of switching dropdowns.

These tests pin:

* trace counts (one line per metric panel × strategy),
* color stability (each canonical strategy maps to a fixed hue),
* deterministic strategy ordering (S0 → S3, custom keys after),
* graceful degeneration on single-strategy / empty inputs,
* skipping zero-signal panels (state tax / IRMAA in stateless,
  young households),
* legend grouping (clicking a strategy in the legend toggles
  every metric for that strategy at once),
* the liquidity-cap marker emitting only on years where the
  guard fired.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import pytest

# Module is dash-free except for the runner pull-in inside the
# multi-strategy builders, but that import is lazy so we can import
# `figures` without dash present. Still gate the suite on dash
# because the smoke test below uses the runner.
pytest.importorskip("dash")

from dash_app import figures  # noqa: E402
from dash_app.figures import (  # noqa: E402
    _OKABE_ITO,
    _STRATEGY_COLORS,
    _STRATEGY_DASH,
    _STRATEGY_MARKER,
    _color_for,
    _dash_for,
    _marker_for,
    _stable_strategy_order,
    multi_strategy_conversion_panel,
    multi_strategy_taxes_panel,
)


# ---------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------


def _make_df(
    *,
    n_years: int = 20,
    federal_scale: float = 1.0,
    state_value: float = 0.0,
    irmaa_value: float = 0.0,
    conv_value: float = 0.0,
    rmd_value: float = 0.0,
    cap_year: int | None = None,
) -> pd.DataFrame:
    """Tiny per-year DataFrame in the shape the simulator emits.

    Only the columns the multi-strategy builders read are populated;
    everything else stays absent so tests fail loudly if the
    builders ever try to fish out a column that isn't documented
    as required.
    """
    ages = np.arange(54, 54 + n_years)
    df = pd.DataFrame({
        "spouse_a_age": ages,
        "agi": 100_000 + 5_000 * np.arange(n_years),
        "federal_tax": (15_000 + 1_000 * np.arange(n_years)) * federal_scale,
        "state_tax": np.full(n_years, state_value),
        "irmaa": np.full(n_years, irmaa_value),
        "marginal": np.linspace(0.22, 0.32, n_years),
        "roth_conversion": np.full(n_years, conv_value),
        "rmd": np.full(n_years, rmd_value),
        "roth_conv_capped_by_liquidity": np.zeros(n_years, dtype=bool),
    })
    if cap_year is not None:
        df.loc[cap_year, "roth_conv_capped_by_liquidity"] = True
    return df


def _strategy_payload(name: str, **kwargs: Any) -> dict[str, Any]:
    df = _make_df(**kwargs)
    return {
        "name": name,
        "df": df,
        "summary": {"terminal_after_tax": 1_000_000.0},
        "cfg_summary": {},
    }


@pytest.fixture
def four_strategies() -> dict[str, dict[str, Any]]:
    """Synthetic 4-strategy payload with a deliberate signature on
    each strategy:

    - S0 baseline:   no conversion / no RMD spike.
    - S1 all-Roth:   slightly higher federal tax (Roth is post-tax).
    - S2 fill-22%:   nonzero conversions, smaller RMDs.
    - S3 optimized:  largest conversions + a single liquidity cap
                     event in year 7.
    """
    return {
        "S0_baseline": _strategy_payload("S0_baseline"),
        "S1_all_roth_401k": _strategy_payload(
            "S1_all_roth_401k", federal_scale=1.05
        ),
        "S2_bracket_fill_22": _strategy_payload(
            "S2_bracket_fill_22", conv_value=20_000, rmd_value=5_000
        ),
        "S3_optimized": _strategy_payload(
            "S3_optimized", conv_value=40_000, rmd_value=2_500, cap_year=7
        ),
    }


# ---------------------------------------------------------------------
# Color palette + ordering
# ---------------------------------------------------------------------


class TestPaletteAndOrdering:
    def test_canonical_strategies_have_stable_colors(self) -> None:
        # Each canonical strategy must map to a unique color so the
        # multi-strategy charts stay readable without a legend lookup.
        colors = [_color_for(s) for s in [
            "S0_baseline", "S1_all_roth_401k",
            "S2_bracket_fill_22", "S3_optimized",
        ]]
        assert len(set(colors)) == 4, "canonical strategies must have unique colors"
        # Optimizer keeps the most positive hue (bluish-green) by convention.
        assert _color_for("S3_optimized") == _STRATEGY_COLORS["S3_optimized"]

    def test_strategy_palette_is_okabe_ito(self) -> None:
        """Every strategy color must be drawn from the Okabe-Ito
        palette, the de-facto color-universal-design standard.

        Pinning the palette membership stops a future refactor
        from silently swapping in a Tailwind-style hue (which is
        what we *had* before — green-500/orange-500/violet-500/
        slate-500 — and which fails for deuteranopia).
        """
        cud_colors = set(_OKABE_ITO.values())
        for name, color in _STRATEGY_COLORS.items():
            assert color in cud_colors, (
                f"{name} color {color} is not in the Okabe-Ito palette"
            )
        # Spot-check the canonical Okabe-Ito hex codes themselves
        # so we'd notice if somebody redefined the palette.
        assert _OKABE_ITO["blue"] == "#0072B2"
        assert _OKABE_ITO["bluish_green"] == "#009E73"
        assert _OKABE_ITO["orange"] == "#E69F00"
        assert _OKABE_ITO["reddish_purple"] == "#CC79A7"
        assert _OKABE_ITO["vermillion"] == "#D55E00"

    def test_each_strategy_has_unique_marker_symbol(self) -> None:
        """Marker shape is the redundant cue for users with
        achromatopsia (total color blindness) and for B&W print.
        Every canonical strategy must have a distinct symbol.
        """
        markers = [_marker_for(s) for s in _STRATEGY_COLORS]
        assert len(set(markers)) == len(_STRATEGY_COLORS), (
            f"strategies must have unique markers, got {markers}"
        )
        # Spot-check: the four pinned shapes should all be present.
        # If somebody adds a 5th strategy with no marker entry, it
        # falls through to the fallback ("x") — that's intentional
        # so the test still passes for the canonical 4.
        assert set(markers) >= {"circle", "square", "diamond", "triangle-up"}

    def test_each_strategy_has_dash_pattern(self) -> None:
        """Dash pattern is the second redundant cue. The baseline
        and the winner both render solid (the two anchor lines
        the eye should track first); the alternatives render
        dash + dot so the four lines stay distinguishable in
        monochrome.
        """
        dashes = {s: _dash_for(s) for s in _STRATEGY_COLORS}
        assert dashes["S0_baseline"] == "solid"
        assert dashes["S3_optimized"] == "solid"
        # The two "alternative" strategies must each be NON-solid
        # (dash or dot or longdash etc.) so they're distinct from
        # the two solid lines and from each other.
        assert dashes["S1_all_roth_401k"] != "solid"
        assert dashes["S2_bracket_fill_22"] != "solid"
        assert dashes["S1_all_roth_401k"] != dashes["S2_bracket_fill_22"]

    def test_unknown_strategy_uses_fallback(self) -> None:
        unknown = _color_for("custom_strategy_42")
        # Falls through to a neutral hue — not a canonical color so
        # we can spot it visually.
        assert unknown not in _STRATEGY_COLORS.values()

    def test_stable_order_pins_canonical_first(self) -> None:
        order = _stable_strategy_order(
            ["custom_a", "S3_optimized", "S0_baseline", "S2_bracket_fill_22"]
        )
        # S0 → S3 always come before custom keys regardless of input
        # iteration order; custom keys keep their relative order.
        assert order == [
            "S0_baseline", "S2_bracket_fill_22", "S3_optimized", "custom_a"
        ]

    def test_stable_order_is_idempotent(self) -> None:
        names = ["S1_all_roth_401k", "S0_baseline"]
        ordered = _stable_strategy_order(names)
        assert _stable_strategy_order(ordered) == ordered


# ---------------------------------------------------------------------
# multi_strategy_taxes_panel
# ---------------------------------------------------------------------


class TestMultiStrategyTaxesPanel:
    def test_empty_payload_returns_placeholder(self) -> None:
        fig = multi_strategy_taxes_panel({})
        assert len(fig.data) == 0  # placeholder is annotation-only

    def test_one_trace_per_strategy_per_panel(self, four_strategies) -> None:
        # Default fixture has no state tax / no IRMAA, so the
        # active panels are AGI + Federal tax + Marginal = 3 panels
        # × 4 strategies = 12 traces.
        fig = multi_strategy_taxes_panel(four_strategies)
        names = Counter(t.name for t in fig.data)
        assert dict(names) == {
            "S0_baseline": 3,
            "S1_all_roth_401k": 3,
            "S2_bracket_fill_22": 3,
            "S3_optimized": 3,
        }

    def test_state_tax_panel_added_when_signal_present(self) -> None:
        strategies = {
            "S0_baseline": _strategy_payload("S0_baseline", state_value=2_000),
            "S3_optimized": _strategy_payload("S3_optimized", state_value=2_500),
        }
        fig = multi_strategy_taxes_panel(strategies)
        # AGI + Federal + State + Marginal = 4 panels × 2 strategies = 8 traces.
        assert len(fig.data) == 4 * 2

    def test_irmaa_panel_added_when_signal_present(self) -> None:
        strategies = {
            "S0_baseline": _strategy_payload("S0_baseline", irmaa_value=1_500),
            "S3_optimized": _strategy_payload("S3_optimized", irmaa_value=3_000),
        }
        fig = multi_strategy_taxes_panel(strategies)
        # AGI + Federal + IRMAA + Marginal = 4 panels × 2 strategies = 8 traces.
        assert len(fig.data) == 4 * 2

    def test_zero_signal_panels_are_skipped(self, four_strategies) -> None:
        """Stateless / pre-IRMAA scenarios should NOT render a
        zeroed-out state-tax row — that's just visual noise.

        Detection: panel labels live on the y-axis title of each
        subplot. ``fig.layout`` exposes them as ``yaxis.title.text``,
        ``yaxis2.title.text``, ... — one per row. We collect all
        of them and assert which metrics rendered.
        """
        fig = multi_strategy_taxes_panel(four_strategies)
        y_titles: list[str] = []
        # `fig.layout.yaxis`, `yaxis2`, ... — there's no clean
        # iterator so we walk the keys.
        for key in dir(fig.layout):
            if key.startswith("yaxis"):
                axis = getattr(fig.layout, key, None)
                if axis is not None and getattr(axis, "title", None) is not None:
                    text = getattr(axis.title, "text", None)
                    if text:
                        y_titles.append(text)
        joined = " ".join(y_titles)
        # No state tax / no IRMAA in the default fixture.
        assert "State tax" not in joined
        assert "IRMAA" not in joined
        # Marginal panel always renders.
        assert "Marginal" in joined

    def test_legend_only_renders_once_per_strategy(self, four_strategies) -> None:
        """With 12 traces (4 strategies × 3 panels) but only 4
        strategies, the legend should show 4 entries — not 12 — so
        clicking a name toggles all panels for that strategy.
        """
        fig = multi_strategy_taxes_panel(four_strategies)
        legend_traces = [t for t in fig.data if t.showlegend]
        # showlegend defaults to True if not set explicitly; we explicitly
        # set it to False for non-first-row traces. Count = strategies × 1.
        assert len(legend_traces) == 4

    def test_marginal_is_rendered_as_percentage(self, four_strategies) -> None:
        fig = multi_strategy_taxes_panel(four_strategies)
        # Find a marginal-row trace (hover format is `%{y:.0f}%`).
        marginal_traces = [
            t for t in fig.data if "Marginal" in (t.hovertemplate or "")
        ]
        assert marginal_traces, "no marginal-bracket traces emitted"
        # 0.22-0.32 range × 100 should land in the 22-32 band.
        for t in marginal_traces:
            assert 20 < min(t.y) < 35
            assert 20 < max(t.y) < 35

    def test_strategy_color_pins_per_trace(self, four_strategies) -> None:
        """Every trace for `S3_optimized` must use the same color so
        the user can pick out the optimizer at a glance.
        """
        fig = multi_strategy_taxes_panel(four_strategies)
        s3_traces = [t for t in fig.data if t.name == "S3_optimized"]
        s3_colors = {t.line.color for t in s3_traces}
        assert s3_colors == {_color_for("S3_optimized")}

    def test_winner_line_is_thicker_than_alternatives(
        self, four_strategies
    ) -> None:
        """When ``winner_name`` is supplied, the winner's traces
        render with a heavier stroke than the non-winners. Without
        this, the four strategies all read at the same weight and
        the user has to match color → strategy in the legend to
        see the verdict.
        """
        fig = multi_strategy_taxes_panel(
            four_strategies, winner_name="S3_optimized"
        )
        winner_widths = {
            t.line.width for t in fig.data if t.name == "S3_optimized"
        }
        non_winner_widths = {
            t.line.width for t in fig.data
            if t.name in {"S0_baseline", "S1_all_roth_401k", "S2_bracket_fill_22"}
        }
        # Single value each (every panel uses the same width per
        # strategy) and the winner's stroke is strictly heavier.
        assert len(winner_widths) == 1
        assert len(non_winner_widths) == 1
        assert winner_widths.pop() > non_winner_widths.pop()

    def test_winner_drawn_last_so_it_sits_on_top(self, four_strategies) -> None:
        """Plotly draws traces in the order they were added — last
        wins the z-stack. The winner must be added LAST per panel
        so its line isn't visually buried under whichever strategy
        it tracks closely (frequently S2 in the conversion panel).

        Detection: per row, the LAST trace (highest index in
        `fig.data` for that subplot) must be the winner.
        """
        fig = multi_strategy_taxes_panel(
            four_strategies, winner_name="S3_optimized"
        )
        # Group traces by their subplot anchor (`yaxis`,
        # `yaxis2`, ...).
        by_axis: dict[str, list[str]] = {}
        for t in fig.data:
            axis = (t.yaxis or "y")
            by_axis.setdefault(axis, []).append(t.name)
        # Each panel's last trace is the winner.
        for axis, names in by_axis.items():
            assert names[-1] == "S3_optimized", (
                f"axis {axis} ended with {names[-1]} not the winner"
            )

    def test_traces_use_lines_plus_markers(self, four_strategies) -> None:
        """Every per-strategy trace renders with line + marker so
        individual years are visible (not just the aggregate
        curve). At 30+ years the markers double as visual anchors
        when several strategies overlap on the same path (e.g. S0
        + S1 both flat at zero in the conversion panel).
        """
        fig = multi_strategy_taxes_panel(four_strategies)
        modes = {t.mode for t in fig.data}
        assert modes == {"lines+markers"}

    def test_rendered_traces_apply_per_strategy_marker_symbol(
        self, four_strategies
    ) -> None:
        """Pin the rendered trace properties — not just the
        dictionaries. A regression where the dicts stay correct
        but `_add_strategy_lines` stops applying them would still
        ship a non-CB-friendly chart.
        """
        fig = multi_strategy_taxes_panel(four_strategies)
        # Map strategy name → set of marker symbols seen across
        # all panels for that strategy. With per-strategy markers
        # the set should be size 1.
        symbols_by_strategy: dict[str, set[str]] = {}
        for t in fig.data:
            symbols_by_strategy.setdefault(t.name, set()).add(t.marker.symbol)
        for name, symbols in symbols_by_strategy.items():
            assert len(symbols) == 1, (
                f"{name} renders with multiple marker symbols: {symbols}"
            )
            assert next(iter(symbols)) == _STRATEGY_MARKER[name]
        # And the four canonical strategies use four DIFFERENT
        # symbols (so the legend at the bottom is unambiguous in
        # monochrome).
        all_symbols = {
            next(iter(syms)) for syms in symbols_by_strategy.values()
        }
        assert len(all_symbols) == 4

    def test_rendered_traces_apply_per_strategy_dash(
        self, four_strategies
    ) -> None:
        """Same as above but for line dash patterns."""
        fig = multi_strategy_taxes_panel(four_strategies)
        dashes_by_strategy: dict[str, set[str]] = {}
        for t in fig.data:
            dashes_by_strategy.setdefault(t.name, set()).add(t.line.dash)
        for name, dashes in dashes_by_strategy.items():
            assert len(dashes) == 1, (
                f"{name} renders with multiple line dashes: {dashes}"
            )
            assert next(iter(dashes)) == _STRATEGY_DASH[name]


# ---------------------------------------------------------------------
# multi_strategy_conversion_panel
# ---------------------------------------------------------------------


class TestMultiStrategyConversionPanel:
    def test_empty_payload_returns_placeholder(self) -> None:
        fig = multi_strategy_conversion_panel({})
        assert len(fig.data) == 0

    def test_two_panels_per_strategy(self, four_strategies) -> None:
        # 2 panels (Roth conversion + RMD) × 4 strategies = 8 line
        # traces. Plus the liquidity-cap marker on S3 only = 1 extra.
        fig = multi_strategy_conversion_panel(four_strategies)
        # Filter to the per-strategy lines. Strategy traces use
        # `lines+markers` (so individual years are visible);
        # liquidity-cap markers use `markers` only.
        line_traces = [t for t in fig.data if t.mode == "lines+markers"]
        assert len(line_traces) == 4 * 2

    def test_liquidity_cap_marker_only_on_capped_years(
        self, four_strategies
    ) -> None:
        fig = multi_strategy_conversion_panel(four_strategies)
        markers = [t for t in fig.data if t.mode == "markers"]
        # Only S3 has cap_year=7 in the fixture.
        assert len(markers) == 1
        m = markers[0]
        assert "S3_optimized" in m.name
        assert "(liquidity-capped)" in m.name
        # Exactly one capped year → exactly one marker point.
        assert len(m.x) == 1

    def test_no_cap_markers_when_no_strategy_was_capped(self) -> None:
        strategies = {
            "S0_baseline": _strategy_payload("S0_baseline", conv_value=10_000),
            "S3_optimized": _strategy_payload("S3_optimized", conv_value=20_000),
        }
        fig = multi_strategy_conversion_panel(strategies)
        markers = [t for t in fig.data if t.mode == "markers"]
        assert markers == []

    def test_legend_only_shown_once_per_strategy(self, four_strategies) -> None:
        """Both rows reference the same legend entry per strategy."""
        fig = multi_strategy_conversion_panel(four_strategies)
        legend_visible = [t for t in fig.data if t.showlegend]
        # 4 line traces in row 1 (with showlegend=True) + zero in row 2
        # (showlegend=False). The marker also has showlegend=False.
        assert len(legend_visible) == 4


# ---------------------------------------------------------------------
# End-to-end via the runner (catches drift between simulator output
# columns and the column names hard-coded in the figure builder).
# ---------------------------------------------------------------------


class TestEndToEnd:
    def test_real_run_payload_renders_both_charts(self) -> None:
        from tax_optimizer import Config, Inputs
        from dash_app.runner import run_scenario, serialize_run_result

        cfg = Config(horizon_age=58)
        rr = run_scenario(cfg, Inputs(), mode="four_strategies", seed=0)
        payload = serialize_run_result(rr)
        strategies = payload["strategies"]
        # Multi-strategy builders accept the serialized payload (dict
        # with a "df" sub-payload) directly — they deserialize on
        # demand.
        fig1 = multi_strategy_taxes_panel(strategies)
        fig2 = multi_strategy_conversion_panel(strategies)
        # Both produce traces (smoke).
        assert len(fig1.data) > 0
        assert len(fig2.data) > 0
        # One color per strategy in fig1.
        s3_traces = [t for t in fig1.data if t.name == "S3_optimized"]
        assert s3_traces, "no S3_optimized traces in real-run payload"
        s3_colors = {t.line.color for t in s3_traces}
        assert s3_colors == {_color_for("S3_optimized")}
