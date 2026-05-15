"""Tests for the Strategies-tab horizontal bar charts.

Two figure builders share the same anti-truncation layout:

* ``figures.strategy_comparison(strategies, metric=...)`` — single
  bar chart for one metric (used by Monte Carlo and any callsite
  that wants one metric in isolation).
* ``figures.strategy_compare_panel(strategies)`` — 3-subplot
  panel covering Terminal after-tax NW, Lifetime federal tax NPV,
  Lifetime IRMAA NPV (the canonical Strategies-tab view).

The previous layout used ``f"${v:,.0f}"`` for bar text labels and
``textposition="outside"`` with no x-axis padding. With realistic
Strategies-tab values (e.g. ``$17,454,663``) the labels routinely
got clipped at the subplot edge — see CHANGELOG entry "Fixed —
Strategies tab: bar value labels truncated".

These tests pin the fix:

* abbreviated dollar formatting on every bar text label,
* ``textposition="auto"`` so labels render inside when they fit
  and outside when they don't,
* ``cliponaxis=False`` so outside labels are never clipped at
  the axis edge,
* an x-axis range that extends 15-25% beyond the largest value
  so outside labels have somewhere to land,
* font sizes that match the surrounding table chrome.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("dash")

from dash_app.figures import (  # noqa: E402
    strategy_comparison,
    strategy_compare_panel,
)


# ---------------------------------------------------------------------
# Fixtures — replay the exact values from the regression screenshot
# ---------------------------------------------------------------------


@pytest.fixture
def strategies_realistic() -> dict[str, dict[str, Any]]:
    """Verbatim from the regression screenshot. The optimizer
    (S3) wins terminal NW by ~7%, cuts lifetime tax by ~13%,
    and cuts IRMAA by ~53%.
    """
    return {
        "S0_baseline": {
            "summary": {
                "terminal_after_tax": 17_454_663,
                "lifetime_tax_npv":    1_978_147,
                "lifetime_irmaa_npv":     83_378,
            }
        },
        "S1_all_roth_401k": {
            "summary": {
                "terminal_after_tax": 17_454_663,
                "lifetime_tax_npv":    1_978_147,
                "lifetime_irmaa_npv":     83_378,
            }
        },
        "S2_bracket_fill_22": {
            "summary": {
                "terminal_after_tax": 17_454_663,
                "lifetime_tax_npv":    1_978_147,
                "lifetime_irmaa_npv":     83_378,
            }
        },
        "S3_optimized": {
            "summary": {
                "terminal_after_tax": 18_626_834,
                "lifetime_tax_npv":    1_713_631,
                "lifetime_irmaa_npv":     39_457,
            }
        },
    }


# ---------------------------------------------------------------------
# strategy_compare_panel (the 3-subplot Strategies-tab chart)
# ---------------------------------------------------------------------


class TestStrategyComparePanel:
    def test_empty_returns_placeholder(self) -> None:
        fig = strategy_compare_panel({})
        assert len(fig.data) == 0

    def test_three_subplots_one_per_metric(self, strategies_realistic) -> None:
        fig = strategy_compare_panel(strategies_realistic)
        assert len(fig.data) == 3

    def test_bar_labels_use_abbreviated_format(
        self, strategies_realistic
    ) -> None:
        """The whole point of the fix: full-precision labels
        (`$17,454,663`) get truncated. Pin the abbreviated form
        so a future refactor can't bring back the ugly version.
        """
        fig = strategy_compare_panel(strategies_realistic)
        for trace in fig.data:
            for label in trace.text:
                # Full-precision must NOT appear.
                assert "$17,454,663" not in label
                assert "$18,626,834" not in label
                assert "$1,978,147" not in label
                assert "$83,378" not in label
        # And the abbreviated forms ARE present somewhere.
        all_labels = sum(([*t.text] for t in fig.data), [])
        assert "$17.5M" in all_labels
        assert "$18.6M" in all_labels
        assert "$83K" in all_labels
        assert "$39K" in all_labels

    def test_textposition_is_auto(self, strategies_realistic) -> None:
        """`auto` picks inside-vs-outside per label so short
        bars don't pile their label inside the subplot's right
        edge. Previously every label was forced `outside`."""
        fig = strategy_compare_panel(strategies_realistic)
        for trace in fig.data:
            assert trace.textposition == "auto"

    def test_cliponaxis_is_disabled(self, strategies_realistic) -> None:
        """Defensive against tiny bars whose label has to
        render outside the axis range — without this, Plotly
        clips outside labels to the axis box."""
        fig = strategy_compare_panel(strategies_realistic)
        for trace in fig.data:
            # `cliponaxis` is False (so the trace can render
            # text outside the axis without being clipped).
            assert trace.cliponaxis is False

    def test_xaxis_padded_beyond_max_value(
        self, strategies_realistic
    ) -> None:
        """Outside labels need somewhere to go. Each subplot's
        xaxis range extends 15-25% beyond the largest value.
        """
        fig = strategy_compare_panel(strategies_realistic)
        # Subplot 1: terminal NW. Max is $18,626,834 (S3).
        # 18% headroom = ~$21.98M ceiling.
        assert fig.layout.xaxis.range is not None
        assert fig.layout.xaxis.range[0] == 0
        assert fig.layout.xaxis.range[1] >= 18_626_834 * 1.15
        assert fig.layout.xaxis.range[1] <= 18_626_834 * 1.25
        # Subplot 2: lifetime tax. Max is $1,978,147.
        assert fig.layout.xaxis2.range[1] >= 1_978_147 * 1.15
        # Subplot 3: lifetime IRMAA. Max is $83,378.
        assert fig.layout.xaxis3.range[1] >= 83_378 * 1.15

    def test_subplot_titles_have_font_size(
        self, strategies_realistic
    ) -> None:
        """Subplot titles default to a small font that doesn't
        match the dashboard's table chrome. Pin an explicit size.
        """
        fig = strategy_compare_panel(strategies_realistic)
        # `make_subplots` exposes subplot titles as layout
        # annotations.
        title_annotations = [
            a for a in fig.layout.annotations
            if a.text in {
                "Terminal after-tax NW",
                "Lifetime federal tax (NPV)",
                "Lifetime IRMAA (NPV)",
            }
        ]
        assert len(title_annotations) == 3
        for ann in title_annotations:
            assert ann.font.size == 13

    def test_chart_title_has_font_size(self, strategies_realistic) -> None:
        fig = strategy_compare_panel(strategies_realistic)
        assert fig.layout.title.font.size == 15

    def test_bar_textfont_size(self, strategies_realistic) -> None:
        """Bar value labels (the `$17.5M` etc.) must use an
        explicit font size — Plotly's default (~10-12) is
        smaller than the rest of the dashboard's chrome."""
        fig = strategy_compare_panel(strategies_realistic)
        for trace in fig.data:
            assert trace.textfont.size == 12


# ---------------------------------------------------------------------
# strategy_comparison (single-metric bar chart)
# ---------------------------------------------------------------------


class TestStrategyComparison:
    def test_empty_returns_placeholder(self) -> None:
        fig = strategy_comparison({})
        assert len(fig.data) == 0

    def test_bar_labels_abbreviated(self, strategies_realistic) -> None:
        fig = strategy_comparison(
            strategies_realistic, metric="terminal_after_tax"
        )
        bar = fig.data[0]
        for label in bar.text:
            assert "$17,454,663" not in label
            assert "$18,626,834" not in label
        # Abbreviated values are present, with the percentage suffix
        # ("(+X.X%)") attached.
        joined = " ".join(bar.text)
        assert "$17.5M" in joined
        assert "$18.6M" in joined
        assert "(+6.7%)" in joined  # S3's lift over baseline

    def test_textposition_is_auto(self, strategies_realistic) -> None:
        fig = strategy_comparison(
            strategies_realistic, metric="terminal_after_tax"
        )
        assert fig.data[0].textposition == "auto"

    def test_cliponaxis_disabled(self, strategies_realistic) -> None:
        fig = strategy_comparison(
            strategies_realistic, metric="terminal_after_tax"
        )
        assert fig.data[0].cliponaxis is False

    def test_xaxis_padded(self, strategies_realistic) -> None:
        fig = strategy_comparison(
            strategies_realistic, metric="terminal_after_tax"
        )
        # 22% headroom (bigger than the panel's 18%) because
        # this view also carries the "(+X.X%)" suffix.
        assert fig.layout.xaxis.range is not None
        assert fig.layout.xaxis.range[1] >= 18_626_834 * 1.18
        assert fig.layout.xaxis.range[1] <= 18_626_834 * 1.30

    def test_chart_title_font_size(self, strategies_realistic) -> None:
        fig = strategy_comparison(
            strategies_realistic, metric="terminal_after_tax"
        )
        assert fig.layout.title.font.size == 15

    def test_handles_nan_and_none_values(self) -> None:
        """Defensive: the original layout used ``""`` for
        NaN/None labels. Make sure abbreviation doesn't choke
        on those.
        """
        import math
        strategies = {
            "S0_baseline": {"summary": {"terminal_after_tax": 17_000_000}},
            "S1_all_roth_401k": {"summary": {"terminal_after_tax": float("nan")}},
            "S2_bracket_fill_22": {"summary": {"terminal_after_tax": None}},
        }
        fig = strategy_comparison(strategies, metric="terminal_after_tax")
        labels = list(fig.data[0].text)
        # Three labels, the first abbreviated, the other two empty.
        assert labels[0].startswith("$17.0M")
        assert labels[1] == ""
        assert labels[2] == ""
