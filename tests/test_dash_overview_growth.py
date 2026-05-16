"""Tests for the Overview tab's growth surfaces.

Covers:

* ``figures.overview_kpis`` returns the new sectioned structure
  (Outcomes / Growth) with all six growth tiles in the second
  section.
* CAGR tile values render with an explicit sign (`+` / `-`).
* ``figures.multi_strategy_growth_panel`` produces 4 strategies ×
  2 panels = 8 line traces, reuses the Okabe-Ito palette + shape
  redundancy via ``_add_strategy_lines``, and stamps a 0 % YoY
  reference line.
* End-to-end: a real ``run_scenario(mode='four_strategies')``
  payload renders all six growth tiles + the chart with no
  exceptions.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("dash")

from tax_optimizer import Config, Inputs  # noqa: E402
from dash_app import figures  # noqa: E402
from dash_app.app import _build_kpi_tiles  # noqa: E402
from dash_app.runner import (  # noqa: E402
    run_scenario,
    serialize_run_result,
)


# ---------------------------------------------------------------------
# Sectioned KPI structure
# ---------------------------------------------------------------------


class TestOverviewKPIs:
    def test_returns_sectioned_structure(self) -> None:
        """Sectioned shape: list of (section_label, list-of-tiles).
        Two sections: "Outcomes" first (the bottom-line tiles), then
        "Growth" (the new tiles).
        """
        summary = _full_growth_summary()
        sections = figures.overview_kpis(summary, mc_payload=None)
        assert isinstance(sections, list)
        assert len(sections) == 2
        labels = [s[0] for s in sections]
        assert labels == ["Outcomes", "Growth"]

    def test_growth_section_has_six_tiles(self) -> None:
        summary = _full_growth_summary()
        sections = figures.overview_kpis(summary, mc_payload=None)
        growth = dict(sections)["Growth"]
        assert len(growth) == 6
        labels = [label for label, _ in growth]
        # Pin both the labels and the order — the dashboard's KPI
        # row reads top-to-bottom, left-to-right.
        assert labels == [
            "Starting after-tax NW",
            "Total growth",
            "Effective CAGR",
            "Real CAGR",
            "Accumulation CAGR",
            "Decumulation CAGR",
        ]

    def test_outcomes_includes_terminal_nw_first(self) -> None:
        """Terminal after-tax NW remains the headline tile of the
        Outcomes section even after the sectioning.
        """
        summary = _full_growth_summary()
        sections = figures.overview_kpis(summary, mc_payload=None)
        outcomes = dict(sections)["Outcomes"]
        assert outcomes[0][0] == "Terminal after-tax NW"

    def test_cagr_tiles_render_with_sign(self) -> None:
        """``+`` for positive CAGR, ``-`` for negative — exposes the
        accumulation-vs-decumulation distinction visually.
        """
        summary = _full_growth_summary(
            effective_cagr=0.042,
            real_cagr=0.018,
            accumulation_cagr=0.062,
            decumulation_cagr=-0.012,
        )
        sections = figures.overview_kpis(summary, mc_payload=None)
        growth = dict(sections)["Growth"]
        values = dict(growth)
        assert values["Effective CAGR"] == "+4.2%/yr"
        assert values["Real CAGR"] == "+1.8%/yr"
        assert values["Accumulation CAGR"] == "+6.2%/yr"
        # Negative CAGR carries a minus sign (not a hyphen-only "-").
        assert values["Decumulation CAGR"].startswith("-")
        assert values["Decumulation CAGR"].endswith("%/yr")

    def test_multiplier_tile_renders_with_x_suffix(self) -> None:
        summary = _full_growth_summary(total_growth_mult=3.15)
        sections = figures.overview_kpis(summary, mc_payload=None)
        values = dict(dict(sections)["Growth"])
        assert values["Total growth"] == "3.15\u00d7"

    def test_missing_growth_keys_render_dash(self) -> None:
        """Legacy summarize without the new kwargs sets every growth
        key to None — every tile must safely fall back to ``"-"``.
        """
        summary = _full_growth_summary(
            starting_after_tax=None,
            total_growth_mult=None,
            effective_cagr=None,
            real_cagr=None,
            accumulation_cagr=None,
            decumulation_cagr=None,
        )
        sections = figures.overview_kpis(summary, mc_payload=None)
        values = dict(dict(sections)["Growth"])
        for label in (
            "Starting after-tax NW",
            "Total growth",
            "Effective CAGR",
            "Real CAGR",
            "Accumulation CAGR",
            "Decumulation CAGR",
        ):
            assert values[label] == "-"

    def test_nan_growth_values_render_dash(self) -> None:
        """``total_growth_multiplier`` returns NaN for non-positive
        starting NW; the formatter must turn that into ``"-"`` and
        not leak ``"nan×"`` into the UI.
        """
        summary = _full_growth_summary(total_growth_mult=float("nan"))
        sections = figures.overview_kpis(summary, mc_payload=None)
        values = dict(dict(sections)["Growth"])
        assert values["Total growth"] == "-"


# ---------------------------------------------------------------------
# _build_kpi_tiles renders the sectioned structure
# ---------------------------------------------------------------------


class TestBuildKpiTiles:
    def test_renders_section_label_and_tile_row(self) -> None:
        sections = [
            ("Outcomes", [("A", "$1"), ("B", "$2")]),
            ("Growth", [("C", "3.15\u00d7")]),
        ]
        children = _build_kpi_tiles(sections)
        # 2 section labels + 2 tile rows
        assert len(children) == 4

    def test_legacy_flat_input_still_works(self) -> None:
        """Older callers passing the flat ``[(label, value), ...]``
        list shape must still render — back-compat keeps any
        downstream tests / scripts working.
        """
        children = _build_kpi_tiles([("A", "$1"), ("B", "$2")])
        # No section label (empty), so just the single tile row.
        assert len(children) == 1

    def test_empty_input_returns_empty(self) -> None:
        assert _build_kpi_tiles([]) == []


# ---------------------------------------------------------------------
# multi_strategy_growth_panel — line traces, palette, reference line
# ---------------------------------------------------------------------


class TestMultiStrategyGrowthPanel:
    def test_eight_line_traces_for_four_strategies(self) -> None:
        """Two stacked subplots × four strategies = 8 line traces."""
        strategies = _toy_strategies()
        fig = figures.multi_strategy_growth_panel(
            strategies, heir_marginal_rate=0.22
        )
        # Strategy lines + horizontal-zero shape (shape, not trace).
        line_traces = [t for t in fig.data if t.type == "scatter"]
        assert len(line_traces) == 8

    def test_strategy_names_present_for_each_panel(self) -> None:
        strategies = _toy_strategies()
        fig = figures.multi_strategy_growth_panel(
            strategies, heir_marginal_rate=0.22
        )
        names = [t.name for t in fig.data]
        # Each strategy appears twice (once per panel).
        for n in (
            "S0_baseline", "S1_all_roth_401k",
            "S2_bracket_fill_22", "S3_optimized",
        ):
            assert names.count(n) == 2

    def test_palette_membership(self) -> None:
        """Every line color must come from the canonical Okabe-Ito
        palette so the chart stays color-blind-friendly.
        """
        strategies = _toy_strategies()
        fig = figures.multi_strategy_growth_panel(
            strategies, heir_marginal_rate=0.22
        )
        palette = set(figures._OKABE_ITO.values())
        # Plus the documented fallback for non-canonical strategies.
        palette.add(figures._STRATEGY_FALLBACK_COLOR)
        for trace in fig.data:
            assert trace.line.color in palette

    def test_marker_redundancy(self) -> None:
        """At least three distinct marker symbols across the four
        canonical strategies — pins the shape-redundancy contract
        even if the chart is printed in black-and-white.
        """
        strategies = _toy_strategies()
        fig = figures.multi_strategy_growth_panel(
            strategies, heir_marginal_rate=0.22
        )
        symbols = {t.marker.symbol for t in fig.data}
        assert len(symbols) >= 3

    def test_dash_redundancy(self) -> None:
        strategies = _toy_strategies()
        fig = figures.multi_strategy_growth_panel(
            strategies, heir_marginal_rate=0.22
        )
        dashes = {t.line.dash for t in fig.data}
        assert len(dashes) >= 3

    def test_winner_line_thicker(self) -> None:
        strategies = _toy_strategies()
        fig = figures.multi_strategy_growth_panel(
            strategies,
            heir_marginal_rate=0.22,
            winner_name="S3_optimized",
        )
        winner_traces = [t for t in fig.data if t.name == "S3_optimized"]
        assert len(winner_traces) == 2
        for t in winner_traces:
            assert t.line.width == figures._LINE_WIDTH_WINNER

    def test_zero_reference_line_on_yoy_panel(self) -> None:
        """The bottom panel (YoY growth) must have a horizontal
        reference at y = 0 — separates accumulation from
        decumulation visually.
        """
        strategies = _toy_strategies()
        fig = figures.multi_strategy_growth_panel(
            strategies, heir_marginal_rate=0.22
        )
        shapes = list(fig.layout.shapes) if fig.layout.shapes else []
        zero_lines = [
            s for s in shapes
            if getattr(s, "y0", None) == 0 and getattr(s, "y1", None) == 0
        ]
        assert len(zero_lines) >= 1

    def test_lines_plus_markers_mode(self) -> None:
        strategies = _toy_strategies()
        fig = figures.multi_strategy_growth_panel(
            strategies, heir_marginal_rate=0.22
        )
        for t in fig.data:
            assert t.mode == "lines+markers"

    def test_empty_strategies_returns_empty_figure(self) -> None:
        fig = figures.multi_strategy_growth_panel(
            {}, heir_marginal_rate=0.22
        )
        # No real traces; figure still renders.
        assert len(fig.data) == 0


# ---------------------------------------------------------------------
# End-to-end: real run payload + render
# ---------------------------------------------------------------------


class TestEndToEnd:
    @pytest.fixture(scope="class")
    def payload(self) -> dict[str, Any]:
        cfg = Config(horizon_age=58)
        inputs = Inputs()
        rr = run_scenario(cfg, inputs, mode="four_strategies", seed=0)
        return serialize_run_result(rr)

    def test_winner_summary_has_growth_keys(self, payload) -> None:
        winner = payload["strategies"][payload["winner_name"]]
        for key in (
            "starting_after_tax",
            "total_growth_mult",
            "effective_cagr",
            "real_cagr",
            "accumulation_cagr",
            "decumulation_cagr",
        ):
            assert key in winner["summary"]

    def test_cfg_summary_carries_heir_marginal_rate(self, payload) -> None:
        for s in payload["strategies"].values():
            assert "heir_marginal_rate" in s["cfg_summary"]
            assert "inflation" in s["cfg_summary"]

    def test_overview_kpis_renders_no_exceptions(self, payload) -> None:
        winner = payload["strategies"][payload["winner_name"]]
        sections = figures.overview_kpis(winner["summary"], payload.get("mc"))
        assert len(sections) == 2
        # Every Growth tile must format to a non-error string.
        growth = dict(sections)["Growth"]
        for label, value in growth:
            assert isinstance(value, str)
            assert value  # non-empty

    def test_growth_chart_renders_no_exceptions(self, payload) -> None:
        winner = payload["strategies"][payload["winner_name"]]
        rate = winner["cfg_summary"]["heir_marginal_rate"]
        fig = figures.multi_strategy_growth_panel(
            payload["strategies"],
            heir_marginal_rate=rate,
            winner_name=payload["winner_name"],
        )
        assert len(fig.data) >= 4

    def test_kpi_tiles_render_no_exceptions(self, payload) -> None:
        winner = payload["strategies"][payload["winner_name"]]
        sections = figures.overview_kpis(winner["summary"], payload.get("mc"))
        children = _build_kpi_tiles(sections)
        assert len(children) > 0


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _full_growth_summary(**overrides: Any) -> dict[str, Any]:
    base = {
        # Outcomes-section keys
        "terminal_after_tax": 1_500_000.0,
        "lifetime_tax_npv": 250_000.0,
        "lifetime_irmaa_npv": 0.0,
        "peak_marginal": 0.22,
        "years_irmaa": 0,
        # Growth-section keys
        "starting_after_tax": 500_000.0,
        "total_growth_mult": 3.0,
        "effective_cagr": 0.038,
        "real_cagr": 0.013,
        "accumulation_cagr": 0.058,
        "decumulation_cagr": -0.012,
    }
    base.update(overrides)
    return base


def _toy_strategies() -> dict[str, dict[str, Any]]:
    """Run a real four-strategies simulation and return the
    serialized strategies dict. Lets the multi-strategy chart
    builder exercise its full deserialize path.
    """
    cfg = Config(horizon_age=58)
    inputs = Inputs()
    rr = run_scenario(cfg, inputs, mode="four_strategies", seed=0)
    payload = serialize_run_result(rr)
    return payload["strategies"]
