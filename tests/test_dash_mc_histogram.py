"""Tests for the Monte-Carlo terminal histogram figure.

The histogram has three percentile callouts (P10/P50/P90) attached
to vertical guide lines. With long fully-formatted dollar amounts
(``$5,826,124`` etc.) the labels collide horizontally — see the
screenshot in CHANGELOG.md under
"Fixed — Monte Carlo histogram percentile labels overlap".

These tests pin the anti-overlap layout:

* abbreviated dollar formatting (`$5.8M` not `$5,826,124`),
* three different ``xanchor`` values so the labels land in three
  different horizontal zones,
* a non-zero ``yshift`` on the middle (P50) callout so even if
  P10 and P50 lines are close in x, the labels sit in different
  vertical rows,
* the chart title also uses the abbreviated formatter so the
  CVaR readout stays compact.
"""

from __future__ import annotations

import pytest

pytest.importorskip("dash")

from dash_app.figures import _abbrev_dollars, mc_terminal_histogram  # noqa: E402


# ---------------------------------------------------------------------
# _abbrev_dollars
# ---------------------------------------------------------------------


class TestAbbrevDollars:
    @pytest.mark.parametrize("value, expected", [
        # Millions: 1 decimal precision (typical Monte Carlo terminal range)
        (5_826_124, "$5.8M"),
        (18_189_053, "$18.2M"),
        (43_815_711, "$43.8M"),
        (1_000_000, "$1.0M"),
        # Billions: 1 decimal precision
        (1_234_000_000, "$1.2B"),
        (12_500_000_000, "$12.5B"),
        # Tens of thousands and up: round to whole K
        (50_000, "$50K"),
        (123_456, "$123K"),
        # Below 10K stays in full-precision dollar form — at this
        # magnitude there's nothing to abbreviate, and "$0K" for a
        # zero would be misleading.
        (9_999, "$9,999"),
        (0, "$0"),
        # Negative values are handled symmetrically.
        (-5_826_124, "$-5.8M"),
    ])
    def test_abbreviation(self, value: float, expected: str) -> None:
        assert _abbrev_dollars(value) == expected


# ---------------------------------------------------------------------
# mc_terminal_histogram percentile annotation layout
# ---------------------------------------------------------------------


@pytest.fixture
def mc_payload_with_skewed_distribution() -> dict:
    """Payload with the exact percentiles from the regression
    screenshot — these are the ones that overlapped in the old
    layout.
    """
    return {
        "terminals": [5e6, 10e6, 18e6, 25e6, 40e6, 60e6, 80e6, 100e6, 175e6],
        "percentiles": {
            "p10": 5_826_124,
            "p50": 18_189_053,
            "p90": 43_815_711,
        },
        "prob_success": 1.0,
        "cvar_terminal": 3_823_858,
        "fan": None,
    }


class TestPercentileAnnotations:
    def test_empty_payload_returns_placeholder(self) -> None:
        fig = mc_terminal_histogram(None)
        # Placeholder figure has annotations for the message but no
        # histogram data.
        assert len(fig.data) == 0

    def test_three_percentile_annotations_present(
        self, mc_payload_with_skewed_distribution
    ) -> None:
        fig = mc_terminal_histogram(mc_payload_with_skewed_distribution)
        texts = [a.text for a in fig.layout.annotations]
        assert "P10: $5.8M" in texts
        assert "P50: $18.2M" in texts
        assert "P90: $43.8M" in texts

    def test_dollar_amounts_are_abbreviated_not_full_precision(
        self, mc_payload_with_skewed_distribution
    ) -> None:
        """The whole point of the fix: long fully-formatted amounts
        like `$5,826,124` are what caused the visual overlap.
        Pin that they're never emitted by this builder.
        """
        fig = mc_terminal_histogram(mc_payload_with_skewed_distribution)
        for ann in fig.layout.annotations:
            assert "$5,826,124" not in (ann.text or "")
            assert "$18,189,053" not in (ann.text or "")
            assert "$43,815,711" not in (ann.text or "")

    def test_percentile_labels_use_distinct_xanchors(
        self, mc_payload_with_skewed_distribution
    ) -> None:
        """Anti-overlap layout pin #1: P10 / P50 / P90 must each
        anchor differently so the three labels never compete for
        the same horizontal slot.

        Concretely: the leftmost line's label extends LEFT
        (xanchor=right), the middle's label is CENTERED, and the
        rightmost's label extends RIGHT (xanchor=left). Even if
        two lines are visually close in x, their labels splay
        out into different zones.
        """
        fig = mc_terminal_histogram(mc_payload_with_skewed_distribution)
        anchors_by_tag: dict[str, str] = {}
        for ann in fig.layout.annotations:
            tag = (ann.text or "").split(":")[0]
            anchors_by_tag[tag] = ann.xanchor
        # All three present
        assert set(anchors_by_tag) == {"P10", "P50", "P90"}
        # All three different — the core invariant
        assert len(set(anchors_by_tag.values())) == 3
        # Specific anchors: leftmost line label anchors RIGHT (so
        # the label extends leftward away from the line); rightmost
        # anchors LEFT.
        assert anchors_by_tag["P10"] == "right"
        assert anchors_by_tag["P50"] == "center"
        assert anchors_by_tag["P90"] == "left"

    def test_p50_label_is_vertically_offset(
        self, mc_payload_with_skewed_distribution
    ) -> None:
        """Anti-overlap layout pin #2: even with horizontal
        anchoring fixed, in pathological cases (e.g. P10 ≈ P50)
        the labels could still collide. Bumping P50 up by
        `yshift>0` puts it in a different vertical row from the
        side labels so the worst case stays readable.
        """
        fig = mc_terminal_histogram(mc_payload_with_skewed_distribution)
        yshifts: dict[str, int] = {}
        for ann in fig.layout.annotations:
            tag = (ann.text or "").split(":")[0]
            yshifts[tag] = ann.yshift or 0
        # P50 sits in a higher row than the side labels.
        assert yshifts["P50"] > yshifts["P10"]
        assert yshifts["P50"] > yshifts["P90"]

    def test_annotations_have_background_for_legibility(
        self, mc_payload_with_skewed_distribution
    ) -> None:
        """A solid-ish bgcolor prevents the text from disappearing
        into the histogram bars or the chart title.
        """
        fig = mc_terminal_histogram(mc_payload_with_skewed_distribution)
        for ann in fig.layout.annotations:
            # Either a Plotly color string, an rgb tuple, or rgba
            # string. We just need it to be set (not None / empty).
            assert ann.bgcolor, f"annotation {ann.text!r} has no bgcolor"

    def test_title_uses_abbreviated_cvar(
        self, mc_payload_with_skewed_distribution
    ) -> None:
        """The chart title shows CVaR(10%) which can also be a
        7-figure number. Same `_abbrev_dollars` keeps the title
        from wrapping or running off the right edge.
        """
        fig = mc_terminal_histogram(mc_payload_with_skewed_distribution)
        title = fig.layout.title.text or ""
        # `$3,823,858` would be the un-abbreviated form.
        assert "$3,823,858" not in title
        assert "$3.8M" in title

    def test_top_margin_is_extended(
        self, mc_payload_with_skewed_distribution
    ) -> None:
        """Bumping P50 up by `yshift=22` pushes it close to the
        chart title. Extra top margin keeps them from crashing
        into each other.
        """
        fig = mc_terminal_histogram(mc_payload_with_skewed_distribution)
        # Default `_LAYOUT.margin.t` is 50; this builder overrides
        # it to 80.
        assert (fig.layout.margin.t or 0) >= 70
