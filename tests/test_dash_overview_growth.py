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
        growth = _section(sections, "Growth")
        assert len(growth) == 6
        labels = [t[0] for t in growth]
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
        outcomes = _section(sections, "Outcomes")
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
        values = _values(sections, "Growth")
        assert values["Effective CAGR"] == "+4.2%/yr"
        assert values["Real CAGR"] == "+1.8%/yr"
        assert values["Accumulation CAGR"] == "+6.2%/yr"
        # Negative CAGR carries a minus sign (not a hyphen-only "-").
        assert values["Decumulation CAGR"].startswith("-")
        assert values["Decumulation CAGR"].endswith("%/yr")

    def test_multiplier_tile_renders_with_x_suffix(self) -> None:
        summary = _full_growth_summary(total_growth_mult=3.15)
        sections = figures.overview_kpis(summary, mc_payload=None)
        values = _values(sections, "Growth")
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
        values = _values(sections, "Growth")
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
        values = _values(sections, "Growth")
        assert values["Total growth"] == "-"


# ---------------------------------------------------------------------
# Tile hints — every Overview tile carries a non-empty help string
# ---------------------------------------------------------------------


class TestOverviewTileHints:
    """Every KPI tile in the Outcomes / Growth sections must surface
    a non-empty ``hint`` string so the UI can show the ⓘ-icon
    tooltip. Pinning this contract here means a future tile addition
    can't silently ship without explanatory help text.
    """

    def test_every_outcomes_tile_has_hint(self) -> None:
        summary = _full_growth_summary()
        sections = figures.overview_kpis(summary, mc_payload=None)
        outcomes = _section(sections, "Outcomes")
        for label, _value, hint in outcomes:
            assert hint, f"Outcomes tile {label!r} is missing a hint"

    def test_every_growth_tile_has_hint(self) -> None:
        summary = _full_growth_summary()
        sections = figures.overview_kpis(summary, mc_payload=None)
        growth = _section(sections, "Growth")
        for label, _value, hint in growth:
            assert hint, f"Growth tile {label!r} is missing a hint"

    def test_mc_tiles_have_hints_when_present(self) -> None:
        """The two MC-only tiles (Probability of success, CVaR(10%))
        only render when an MC payload exists. Both must carry a
        hint when they appear.
        """
        summary = _full_growth_summary()
        mc_payload = {"prob_success": 0.92, "cvar_terminal": 1_200_000.0}
        sections = figures.overview_kpis(summary, mc_payload=mc_payload)
        outcomes = _section(sections, "Outcomes")
        labels = [t[0] for t in outcomes]
        assert "Probability of success" in labels
        assert "CVaR (10%)" in labels
        for label, _value, hint in outcomes:
            assert hint, f"MC tile {label!r} is missing a hint"

    def test_hints_are_non_trivial(self) -> None:
        """A one-word stub hint would defeat the whole point of
        adding tooltips. Pin a minimum length so future
        contributors don't accidentally strip help text.
        """
        summary = _full_growth_summary()
        sections = figures.overview_kpis(summary, mc_payload=None)
        for _section_label, tiles in sections:
            for label, _value, hint in tiles:
                assert len(hint) >= 40, (
                    f"Hint for {label!r} is too short ({len(hint)} chars)"
                )


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

    def test_three_tuple_tiles_render(self) -> None:
        """The new 3-tuple ``(label, value, hint)`` shape must
        render cleanly — that's the shape ``overview_kpis`` emits.
        """
        sections = [
            ("Outcomes", [("A", "$1", "first tile help"),
                           ("B", "$2", "second tile help")]),
            ("Growth", [("C", "3.15\u00d7", "growth help")]),
        ]
        children = _build_kpi_tiles(sections)
        assert len(children) == 4

    def test_three_tuple_tiles_render_tooltip_targeting_card(self) -> None:
        """When a tile has a non-empty hint, the rendered tile must:

        * carry an ``id`` on the outer ``dbc.Card`` (the tooltip
          target — hovering anywhere on the tile triggers it),
        * include a single ``dbc.Tooltip`` whose ``target``
          matches that card id,
        * NOT carry a native ``title`` attribute anywhere — that
          was the source of the duplicate-tooltip bug (browser
          ``title`` fires immediately, then ``dbc.Tooltip``
          fires after its 150 ms delay).
        """
        from dash import html
        import dash_bootstrap_components as dbc

        sections = [
            ("Outcomes", [("A", "$1", "first tile help")]),
        ]
        children = _build_kpi_tiles(sections)
        # One section label + one tile row.
        tile_row = children[1]
        first_tile = tile_row.children[0]
        assert isinstance(first_tile, dbc.Card)
        # Card carries the tooltip target id.
        assert first_tile.id, "Card must have an id when a hint is set"

        body = first_tile.children
        body_children = body.children
        tooltips = [c for c in body_children if isinstance(c, dbc.Tooltip)]
        assert len(tooltips) == 1, "Expected exactly one dbc.Tooltip"
        assert tooltips[0].target == first_tile.id

        # No Dash component in the rendered tree should still carry
        # a native ``title`` attribute — that was the duplicate.
        # We only inspect Dash components (Component subclasses);
        # plain ``str`` children expose a built-in ``str.title``
        # method that is truthy but irrelevant here.
        from dash.development.base_component import Component

        def _component_titles(node: Any) -> list[str]:
            found: list[str] = []
            if isinstance(node, Component):
                t = getattr(node, "title", None)
                if isinstance(t, str) and t:
                    found.append(t)
                kids = getattr(node, "children", None)
                if isinstance(kids, list):
                    for k in kids:
                        found.extend(_component_titles(k))
                elif kids is not None:
                    found.extend(_component_titles(kids))
            return found

        leaked = _component_titles(first_tile)
        assert not leaked, (
            f"Native `title` attribute leaked into the tile tree "
            f"({leaked!r}) — would re-introduce the duplicate-tooltip bug"
        )

    def test_two_tuple_tile_renders_without_tooltip(self) -> None:
        """Legacy 2-tuple ``(label, value)`` tiles must still
        render — just without the tooltip surface and without an
        id (no tooltip target needed).
        """
        import dash_bootstrap_components as dbc

        sections = [("Outcomes", [("A", "$1")])]
        children = _build_kpi_tiles(sections)
        tile_row = children[1]
        first_tile = tile_row.children[0]
        body = first_tile.children
        body_children = body.children
        assert not any(
            isinstance(c, dbc.Tooltip) for c in body_children
        ), "2-tuple tiles must not produce a Tooltip"
        # The Card has no ``id`` for legacy 2-tuple tiles — Dash
        # components only expose attributes that were explicitly
        # set, so accessing ``.id`` raises ``AttributeError`` here.
        assert not hasattr(first_tile, "id") or not first_tile.id, (
            "2-tuple tiles must not stamp an id on the card"
        )


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
        growth = _section(sections, "Growth")
        for tile in growth:
            label, value, hint = tile
            assert isinstance(value, str)
            assert value  # non-empty
            assert isinstance(hint, str) and hint  # every tile has help

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


def _section(
    sections: list[tuple[str, list[tuple[str, str, str]]]], name: str
) -> list[tuple[str, str, str]]:
    """Look up a section's tile list by its label.

    Tiles are 3-tuples ``(label, value, hint)`` so the simple
    ``dict(sections)`` pattern still works since each section
    label is unique. Pulled into a helper to keep test bodies
    declarative.
    """
    return dict(sections)[name]


def _values(
    sections: list[tuple[str, list[tuple[str, str, str]]]], name: str
) -> dict[str, str]:
    """Build a ``{label: value}`` dict for one section's tiles —
    skipping the hint third element. Replaces the old
    ``dict(dict(sections)[name])`` shorthand which assumed
    2-tuple tiles.
    """
    return {t[0]: t[1] for t in _section(sections, name)}


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
