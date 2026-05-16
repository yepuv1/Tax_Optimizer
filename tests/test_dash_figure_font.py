"""Regression test: every Plotly figure in the dashboard renders
with the Fira Code monospace stack so chart typography matches
the surrounding HTML chrome.

Plotly figures render their text inside the chart's SVG and do
NOT inherit ``font-family`` from the page's CSS — without an
explicit ``layout.font.family`` setting, every chart would fall
back to Plotly's default sans-serif (``Open Sans, verdana, arial``)
and look visually disconnected from the rest of the dashboard,
which is themed with Fira Code via
``dash_app/assets/fira-code.css``.

The shared ``_LAYOUT`` dict in ``dash_app/figures.py`` carries
``font=dict(family=_FIRA_FONT_FAMILY, size=12)``, and every
figure builder is expected to spread ``**_LAYOUT`` into its
``update_layout`` call. ``empty_figure`` doesn't use
``_LAYOUT`` but sets the same family directly.

These tests pin both pieces:

* the ``_FIRA_FONT_FAMILY`` constant lists "Fira Code" first
  followed by sensible fallbacks,
* ``_LAYOUT`` carries the ``font.family`` setting,
* every public figure builder produces a figure whose
  ``layout.font.family`` contains "Fira Code".

A future refactor can't accidentally drop the family setting on
one builder without this test failing loudly.
"""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("dash")

from tax_optimizer import Config, Inputs  # noqa: E402
from dash_app import figures  # noqa: E402
from dash_app.runner import (  # noqa: E402
    run_scenario,
    serialize_run_result,
    deserialize_strategy_df,
)


# ---------------------------------------------------------------------
# Shared run payload (a real four_plus_mc run gives every builder
# something to chew on, including the Monte Carlo figures).
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def payload() -> dict:
    cfg = Config(horizon_age=58)
    inputs = Inputs()
    rr = run_scenario(
        cfg, inputs, mode="four_plus_mc", n_paths=30, seed=0
    )
    return serialize_run_result(rr)


@pytest.fixture(scope="module")
def winner_df(payload) -> pd.DataFrame:
    return deserialize_strategy_df(
        payload["strategies"][payload["winner_name"]]["df"]
    )


def _has_fira_code(fig) -> bool:
    """Return True iff ``layout.font.family`` contains "Fira Code"."""
    fam = (fig.layout.font.family or "")
    return "Fira Code" in fam


# ---------------------------------------------------------------------
# Constant + layout shape
# ---------------------------------------------------------------------


class TestFiraFontConstant:
    def test_fira_code_listed_first_in_stack(self) -> None:
        """Fira Code must be the *first* family in the stack so
        browsers that have it loaded use it; the rest of the
        stack is the system-monospace fallback chain.
        """
        fam = figures._FIRA_FONT_FAMILY
        assert fam.startswith("'Fira Code'"), fam

    def test_stack_ends_in_generic_monospace(self) -> None:
        """The stack must end in the generic ``monospace`` family
        so even systems without any of the named fonts still
        get a coherent monospace look.
        """
        assert figures._FIRA_FONT_FAMILY.rstrip().endswith("monospace")

    def test_layout_carries_fira_font(self) -> None:
        """``_LAYOUT`` is the shared layout fragment every figure
        builder spreads into ``update_layout`` — it must carry the
        font family.
        """
        assert "font" in figures._LAYOUT
        assert figures._LAYOUT["font"]["family"] == figures._FIRA_FONT_FAMILY


# ---------------------------------------------------------------------
# Per-builder coverage
# ---------------------------------------------------------------------


class TestEmptyFigureFont:
    def test_empty_figure_uses_fira(self) -> None:
        fig = figures.empty_figure("nothing yet")
        assert _has_fira_code(fig), fig.layout.font.family


class TestSingleStrategyFigureFonts:
    def test_balance_stack(self, winner_df) -> None:
        assert _has_fira_code(figures.balance_stack(winner_df))

    def test_taxes_panel(self, winner_df) -> None:
        assert _has_fira_code(figures.taxes_panel(winner_df))

    def test_conversion_panel(self, winner_df) -> None:
        assert _has_fira_code(figures.conversion_panel(winner_df))


class TestMultiStrategyFigureFonts:
    def test_multi_strategy_taxes_panel(self, payload) -> None:
        fig = figures.multi_strategy_taxes_panel(payload["strategies"])
        assert _has_fira_code(fig)

    def test_multi_strategy_conversion_panel(self, payload) -> None:
        fig = figures.multi_strategy_conversion_panel(
            payload["strategies"]
        )
        assert _has_fira_code(fig)

    def test_multi_strategy_growth_panel(self, payload) -> None:
        fig = figures.multi_strategy_growth_panel(
            payload["strategies"], heir_marginal_rate=0.22
        )
        assert _has_fira_code(fig)


class TestStrategyComparisonFonts:
    def test_strategy_comparison_bar(self, payload) -> None:
        fig = figures.strategy_comparison(payload["strategies"])
        assert _has_fira_code(fig)

    def test_strategy_compare_panel(self, payload) -> None:
        fig = figures.strategy_compare_panel(payload["strategies"])
        assert _has_fira_code(fig)


class TestMonteCarloFigureFonts:
    def test_mc_terminal_histogram(self, payload) -> None:
        fig = figures.mc_terminal_histogram(payload["mc"])
        assert _has_fira_code(fig)

    def test_mc_fan_chart(self, payload) -> None:
        fig = figures.mc_fan_chart(payload["mc"])
        assert _has_fira_code(fig)


# ---------------------------------------------------------------------
# Per-element font overrides do NOT clobber the family
# ---------------------------------------------------------------------


class TestPerElementFontOverridesInheritFamily:
    """Some figures set ``font=dict(size=15)`` or
    ``font=dict(size=11, color=...)`` on specific elements (chart
    title, percentile callouts, subplot titles). Plotly merges
    those partial dicts with the layout-level default — leaving
    ``family`` at the layout setting unless the override
    explicitly sets it. These tests pin that we don't accidentally
    introduce an override that wipes the family.
    """

    def test_strategy_compare_panel_subplot_titles(self, payload) -> None:
        """``strategy_compare_panel`` bumps subplot-title font
        sizes via direct annotation editing. Family must still
        inherit from the layout default.
        """
        fig = figures.strategy_compare_panel(payload["strategies"])
        for ann in fig.layout.annotations:
            # Annotations may set size/color but not family —
            # family must be either ``None`` (inherits) or the
            # Fira stack.
            fam = (ann.font.family if ann.font else None) or ""
            if fam:
                assert "Fira Code" in fam, (
                    f"Annotation {ann.text!r} has family={fam!r} "
                    "(family must inherit or include Fira Code)"
                )

    def test_mc_histogram_callouts_inherit_family(self, payload) -> None:
        """The P10 / P50 / P90 percentile callouts on the MC
        histogram set ``annotation_font=dict(size=11, color=...)``.
        Family must still flow through from the layout.
        """
        fig = figures.mc_terminal_histogram(payload["mc"])
        for ann in fig.layout.annotations:
            fam = (ann.font.family if ann.font else None) or ""
            if fam:
                assert "Fira Code" in fam
