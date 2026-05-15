"""Tests for the Year-by-year tab's column grouping + per-group coloring.

The Year-by-year drill-down DataTable is now structured around a
single source of truth — ``_YEARLY_COLUMN_GROUPS`` — that drives
both:

* ``detail_columns()`` — the simulator-DataFrame filter applied
  by ``filter_to_detail_cols()``,
* ``yearly_table_styles()`` — per-column DataTable conditional
  styles (cell tints + header tints + left-border dividers per
  non-identity group).

These tests pin the contract so a future refactor can't silently
break either the column order, the group-to-column mapping, the
display-column rename (``marginal -> bracket_pct``), or the
color-blind-friendly Okabe-Ito palette membership.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("dash")

from dash_app.figures import (  # noqa: E402
    _DISPLAY_TO_NATIVE,
    _OKABE_ITO,
    _YEARLY_COLUMN_GROUPS,
    _hex_to_rgba,
    detail_columns,
    filter_to_detail_cols,
    yearly_table_styles,
)


# ---------------------------------------------------------------------
# Group declaration is the single source of truth
# ---------------------------------------------------------------------


class TestColumnGroups:
    def test_eight_groups_in_documented_order(self) -> None:
        """The visual order of the table top-to-left-to-right is
        defined by ``_YEARLY_COLUMN_GROUPS``. Pin the eight
        functional sections so a future refactor can't silently
        reshuffle them.
        """
        ids = [g[0] for g in _YEARLY_COLUMN_GROUPS]
        assert ids == [
            "identity",
            "income",
            "pretax",
            "withdraw",
            "spending",
            "tax",
            "medicare",
            "balances",
        ]

    def test_only_identity_group_has_no_color(self) -> None:
        """Identity columns (year / spouse_a_age / filing_status)
        are the table's anchor and intentionally render in default
        chrome. Every other group MUST have a color key, otherwise
        the visual grouping breaks.
        """
        for group_id, color_key, _cols in _YEARLY_COLUMN_GROUPS:
            if group_id == "identity":
                assert color_key is None
            else:
                assert color_key is not None, (
                    f"group {group_id!r} must have a color key"
                )

    def test_color_keys_are_okabe_ito_members(self) -> None:
        """Every color key must resolve via ``_OKABE_ITO`` so the
        Year-by-year tab inherits the same color-blind-friendly
        palette used by the rest of the dashboard.
        """
        for _gid, color_key, _cols in _YEARLY_COLUMN_GROUPS:
            if color_key is None:
                continue
            assert color_key in _OKABE_ITO, (
                f"unknown Okabe-Ito key: {color_key!r}"
            )

    def test_each_color_key_used_at_most_once(self) -> None:
        """Distinct groups should map to distinct hues — otherwise
        two adjacent groups paint the same color and the visual
        grouping disappears.
        """
        used = [c for _g, c, _cols in _YEARLY_COLUMN_GROUPS if c is not None]
        assert len(used) == len(set(used)), f"duplicate color keys: {used}"

    def test_column_names_are_unique_across_groups(self) -> None:
        """A column appearing in two groups would generate
        conflicting style rules. Pin uniqueness across the whole
        declaration.
        """
        all_cols: list[str] = []
        for _g, _c, cols in _YEARLY_COLUMN_GROUPS:
            all_cols.extend(cols)
        assert len(all_cols) == len(set(all_cols)), (
            f"duplicate column across groups: {all_cols}"
        )

    @pytest.mark.parametrize("group_id, expected", [
        ("identity",  ["year", "spouse_a_age", "filing_status"]),
        ("income",    ["wages", "pension", "ssn",
                       "qualified_dividends", "interest_income"]),
        ("pretax",    ["rmd", "roth_conversion"]),
        ("withdraw",  ["pretax_withdrawal", "roth_withdrawal",
                       "taxable_withdrawal"]),
        ("spending",  ["spending_need", "unfunded"]),
        ("tax",       ["agi", "taxable_income", "federal_tax",
                       "state_tax", "bracket_pct"]),
        ("medicare",  ["medicare_base_premium", "irmaa"]),
        ("balances",  ["pretax_balance", "roth_balance",
                       "taxable_balance", "hsa_balance"]),
    ])
    def test_group_membership(self, group_id: str, expected: list[str]) -> None:
        """Spot-check each group's column list. If any of these
        change, the change should be deliberate."""
        for gid, _ck, cols in _YEARLY_COLUMN_GROUPS:
            if gid == group_id:
                assert cols == expected
                return
        pytest.fail(f"group {group_id!r} not found in _YEARLY_COLUMN_GROUPS")


# ---------------------------------------------------------------------
# detail_columns() is derived from _YEARLY_COLUMN_GROUPS
# ---------------------------------------------------------------------


class TestDetailColumns:
    def test_returns_simulator_native_names(self) -> None:
        """``detail_columns()`` returns the DataFrame-filter list,
        which uses the simulator-native column names. The display
        rename ``bracket_pct`` is translated back to ``marginal``
        here so the filter can find the column on the raw frame.
        """
        cols = detail_columns()
        assert "marginal" in cols
        assert "bracket_pct" not in cols

    def test_total_count_matches_groups(self) -> None:
        """Coverage invariant — every column declared in
        ``_YEARLY_COLUMN_GROUPS`` shows up in ``detail_columns()``.
        """
        group_total = sum(len(cols) for _g, _c, cols in _YEARLY_COLUMN_GROUPS)
        assert len(detail_columns()) == group_total
        # 26 = 3 identity + 5 income + 2 pretax + 3 withdraw +
        #      2 spending + 5 tax + 2 medicare + 4 balances.
        assert group_total == 26

    def test_year_first_balances_last(self) -> None:
        """Identity → income → ... → balances ordering."""
        cols = detail_columns()
        assert cols[0] == "year"
        # The last 4 columns are the EOY balances.
        assert cols[-4:] == [
            "pretax_balance", "roth_balance",
            "taxable_balance", "hsa_balance",
        ]

    def test_unfunded_immediately_after_spending_need(self) -> None:
        """Spending group: target then deficit. The deficit column
        only makes sense relative to the target above it.
        """
        cols = detail_columns()
        assert cols.index("unfunded") == cols.index("spending_need") + 1

    def test_medicare_base_premium_before_irmaa(self) -> None:
        """The IRMAA surcharge is on top of the base premium —
        natural reading order is base then surcharge.
        """
        cols = detail_columns()
        assert cols.index("medicare_base_premium") < cols.index("irmaa")

    def test_agi_before_federal_tax(self) -> None:
        """AGI feeds federal tax — natural reading order. The
        previous layout split them with ``state_tax`` in between.
        """
        cols = detail_columns()
        assert cols.index("agi") < cols.index("federal_tax")

    def test_includes_new_columns(self) -> None:
        """The 4 columns added in this round must be present."""
        cols = detail_columns()
        for new_col in (
            "qualified_dividends", "interest_income",
            "taxable_income", "unfunded",
        ):
            assert new_col in cols, f"{new_col} missing from detail_columns()"


# ---------------------------------------------------------------------
# filter_to_detail_cols renames marginal -> bracket_pct and rounds
# ---------------------------------------------------------------------


class TestFilterToDetailCols:
    @pytest.fixture
    def simulator_like_df(self) -> pd.DataFrame:
        """Synthetic DataFrame with all simulator-native columns
        the dashboard reads, plus extras the dashboard ignores
        (so we also confirm extras are filtered OUT).
        """
        n = 4
        data: dict[str, Any] = {
            # Identity
            "year": [2026, 2027, 2028, 2029],
            "spouse_a_age": [54, 55, 56, 57],
            "filing_status": ["MFJ"] * n,
            # Income
            "wages": np.full(n, 250_000.49),
            "pension": np.zeros(n),
            "ssn": np.zeros(n),
            "qualified_dividends": np.full(n, 4_500.7),
            "interest_income": np.full(n, 1_200.3),
            # Pretax events
            "rmd": np.zeros(n),
            "roth_conversion": np.full(n, 50_000.0),
            # Withdrawals
            "pretax_withdrawal": np.zeros(n),
            "roth_withdrawal": np.zeros(n),
            "taxable_withdrawal": np.full(n, 8_000.6),
            # Spending
            "spending_need": np.full(n, 90_000.0),
            "unfunded": np.zeros(n),
            # Tax
            "agi": np.full(n, 305_700.4),
            "taxable_income": np.full(n, 280_000.0),
            "federal_tax": np.full(n, 60_000.5),
            "state_tax": np.full(n, 12_000.0),
            "marginal": np.full(n, 0.24),
            # Medicare
            "medicare_base_premium": np.zeros(n),
            "irmaa": np.zeros(n),
            # Balances
            "pretax_balance": np.full(n, 1_000_000.0),
            "roth_balance": np.full(n, 200_000.0),
            "taxable_balance": np.full(n, 500_000.0),
            "hsa_balance": np.full(n, 50_000.0),
            # Extras the dashboard ignores
            "fica": np.full(n, 18_000.0),
            "rmd_a": np.zeros(n),
        }
        return pd.DataFrame(data)

    def test_output_columns_match_groups_in_display_order(
        self, simulator_like_df
    ) -> None:
        out = filter_to_detail_cols(simulator_like_df)
        # Column order matches _YEARLY_COLUMN_GROUPS top-to-bottom
        # using DISPLAY names (so `bracket_pct`, not `marginal`).
        expected: list[str] = []
        for _g, _c, cols in _YEARLY_COLUMN_GROUPS:
            expected.extend(cols)
        assert list(out.columns) == expected

    def test_marginal_renamed_to_bracket_pct(self, simulator_like_df) -> None:
        out = filter_to_detail_cols(simulator_like_df)
        assert "bracket_pct" in out.columns
        assert "marginal" not in out.columns
        # 0.24 → 24%
        assert int(out["bracket_pct"].iloc[0]) == 24

    def test_extras_are_dropped(self, simulator_like_df) -> None:
        out = filter_to_detail_cols(simulator_like_df)
        assert "fica" not in out.columns
        assert "rmd_a" not in out.columns

    def test_dollar_columns_rounded(self, simulator_like_df) -> None:
        """Every non-identity, non-bracket dollar column should
        be rounded to whole dollars."""
        out = filter_to_detail_cols(simulator_like_df)
        # `wages` was 250_000.49 → should round to 250_000.
        assert float(out["wages"].iloc[0]) == 250_000.0
        # `qualified_dividends` was 4_500.7 → 4_501.
        assert float(out["qualified_dividends"].iloc[0]) == 4_501.0

    def test_handles_missing_columns_gracefully(self) -> None:
        """If the simulator omits some columns (e.g. older
        versions), `filter_to_detail_cols` should still return
        whatever IS present in the same order.
        """
        df = pd.DataFrame({
            "year": [2026], "spouse_a_age": [54], "filing_status": ["MFJ"],
            "wages": [100_000], "agi": [100_000], "marginal": [0.22],
            "pretax_balance": [500_000],
        })
        out = filter_to_detail_cols(df)
        # Only the present columns, in display order.
        assert list(out.columns) == [
            "year", "spouse_a_age", "filing_status",
            "wages", "agi", "bracket_pct", "pretax_balance",
        ]


# ---------------------------------------------------------------------
# yearly_table_styles() — DataTable conditional styling
# ---------------------------------------------------------------------


class TestYearlyTableStyles:
    @pytest.fixture
    def styles(self) -> dict[str, list[dict[str, Any]]]:
        return yearly_table_styles()

    def test_returns_three_conditional_lists(self, styles) -> None:
        assert set(styles.keys()) == {
            "style_header_conditional",
            "style_data_conditional",
            "style_cell_conditional",
        }

    def test_header_and_data_rules_match_colored_columns(
        self, styles
    ) -> None:
        """One header rule + one data rule per column in every
        non-identity group. Identity columns are NOT styled.
        """
        colored_cols: list[str] = []
        for _g, color_key, cols in _YEARLY_COLUMN_GROUPS:
            if color_key is None:
                continue
            colored_cols.extend(cols)
        # 23 = 26 total - 3 identity.
        assert len(colored_cols) == 23
        assert len(styles["style_header_conditional"]) == 23
        assert len(styles["style_data_conditional"]) == 23

        header_targets = {
            r["if"]["column_id"]
            for r in styles["style_header_conditional"]
        }
        data_targets = {
            r["if"]["column_id"]
            for r in styles["style_data_conditional"]
        }
        assert header_targets == set(colored_cols)
        assert data_targets == set(colored_cols)

    def test_one_left_border_per_non_identity_group(self, styles) -> None:
        """Exactly one ``borderLeft`` rule per non-identity
        group's first column — these create the vertical
        dividers between groups.
        """
        cell_rules = styles["style_cell_conditional"]
        assert len(cell_rules) == 7  # 8 groups - 1 identity
        first_cols = {
            cols[0]
            for _g, ck, cols in _YEARLY_COLUMN_GROUPS
            if ck is not None
        }
        target_cols = {r["if"]["column_id"] for r in cell_rules}
        assert target_cols == first_cols
        # Every cell rule has a borderLeft, not just any styling.
        for r in cell_rules:
            assert "borderLeft" in r
            assert r["borderLeft"].startswith("2px solid rgba(")

    def test_header_uses_stronger_alpha_than_data(self, styles) -> None:
        """Header tint (0.22) is stronger than cell tint (0.08)
        so the group label reads at-a-glance while the data
        background stays subtle."""
        # Pull the alpha from the first rule of each list.
        h = styles["style_header_conditional"][0]["backgroundColor"]
        d = styles["style_data_conditional"][0]["backgroundColor"]
        h_alpha = float(re.search(r",\s*([0-9.]+)\)$", h).group(1))
        d_alpha = float(re.search(r",\s*([0-9.]+)\)$", d).group(1))
        assert h_alpha > d_alpha
        assert h_alpha == pytest.approx(0.22)
        assert d_alpha == pytest.approx(0.08)

    def test_every_color_in_styles_is_okabe_ito(self, styles) -> None:
        """Pin palette membership: every rgba color used in any
        rule must be a tint of an Okabe-Ito hex code. If somebody
        adds a one-off non-CB-safe color, this fails.
        """
        # Pre-compute the set of (r, g, b) triples corresponding
        # to Okabe-Ito hex codes. We'll then check every rgba
        # color produced by `yearly_table_styles()` against this
        # set.
        cud_rgb: set[tuple[int, int, int]] = set()
        for hex_str in _OKABE_ITO.values():
            h = hex_str.lstrip("#")
            cud_rgb.add(
                (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
            )

        rgba_re = re.compile(
            r"rgba\(\s*(\d+),\s*(\d+),\s*(\d+),\s*[0-9.]+\)"
        )
        for bucket in styles.values():
            for rule in bucket:
                # Combine all string values in the rule (background,
                # borderLeft, etc.) and find every rgba(...) inside.
                blob = " ".join(
                    str(v) for k, v in rule.items() if k != "if"
                )
                for r, g, b in rgba_re.findall(blob):
                    triple = (int(r), int(g), int(b))
                    assert triple in cud_rgb, (
                        f"rule {rule} uses non-CUD color rgb{triple}"
                    )

    def test_first_column_of_each_group_uses_its_groups_color(
        self, styles
    ) -> None:
        """The left-border divider on the first column of a group
        must use that group's hue — not, say, the previous
        group's. This was a real bug-prone spot.
        """
        cell_rules_by_col = {
            r["if"]["column_id"]: r["borderLeft"]
            for r in styles["style_cell_conditional"]
        }
        for _g, color_key, cols in _YEARLY_COLUMN_GROUPS:
            if color_key is None:
                continue
            first = cols[0]
            assert first in cell_rules_by_col
            expected_color = _hex_to_rgba(_OKABE_ITO[color_key], 0.6)
            assert expected_color in cell_rules_by_col[first]


# ---------------------------------------------------------------------
# _hex_to_rgba helper
# ---------------------------------------------------------------------


class TestHexToRgba:
    @pytest.mark.parametrize("hex_str, alpha, expected", [
        ("#0072B2", 0.08, "rgba(0, 114, 178, 0.08)"),
        ("#0072b2", 0.08, "rgba(0, 114, 178, 0.08)"),     # case insensitive
        ("0072B2",  0.08, "rgba(0, 114, 178, 0.08)"),     # leading # optional
        ("#000000", 1.0,  "rgba(0, 0, 0, 1.0)"),
        ("#FFFFFF", 0.5,  "rgba(255, 255, 255, 0.5)"),
    ])
    def test_known_inputs(self, hex_str, alpha, expected) -> None:
        assert _hex_to_rgba(hex_str, alpha) == expected

    def test_invalid_hex_raises(self) -> None:
        with pytest.raises(ValueError):
            _hex_to_rgba("#0072B", 0.5)        # 5 digits
        with pytest.raises(ValueError):
            _hex_to_rgba("#0072B2AA", 0.5)     # 8 digits


# ---------------------------------------------------------------------
# DISPLAY → NATIVE rename map
# ---------------------------------------------------------------------


class TestDisplayToNative:
    def test_bracket_pct_maps_to_marginal(self) -> None:
        """The dashboard exposes ``bracket_pct`` (integer percent)
        but the simulator emits ``marginal`` (fraction). Pin the
        rename so a future refactor doesn't decouple them.
        """
        assert _DISPLAY_TO_NATIVE.get("bracket_pct") == "marginal"

    def test_every_renamed_display_col_actually_appears_in_groups(
        self,
    ) -> None:
        """Every display name in `_DISPLAY_TO_NATIVE` must show
        up in some group, otherwise the rename has no effect.
        """
        all_group_cols: set[str] = set()
        for _g, _c, cols in _YEARLY_COLUMN_GROUPS:
            all_group_cols.update(cols)
        for display in _DISPLAY_TO_NATIVE:
            assert display in all_group_cols


# ---------------------------------------------------------------------
# End-to-end: real run → filter_to_detail_cols → column match
# ---------------------------------------------------------------------


class TestEndToEnd:
    def test_real_run_produces_columns_matching_detail_columns(self) -> None:
        """Real ``run_scenario`` → ``filter_to_detail_cols`` →
        verify the rendered DataTable column set is exactly the
        union of `_YEARLY_COLUMN_GROUPS` (in display order).
        Catches drift between the simulator's emitted columns and
        the dashboard's filter list.
        """
        from tax_optimizer import Config, Inputs
        from dash_app.runner import run_scenario, deserialize_strategy_df

        cfg = Config(horizon_age=58)
        rr = run_scenario(cfg, Inputs(), mode="four_strategies", seed=0)
        # Pull a strategy DataFrame back through the same path the
        # Dash callback uses.
        for s in rr.strategies.values():
            df = s.df  # already a DataFrame on the runtime object
            out = filter_to_detail_cols(df)
            expected_cols: list[str] = []
            for _g, _c, cols in _YEARLY_COLUMN_GROUPS:
                expected_cols.extend(cols)
            assert list(out.columns) == expected_cols
            return  # one strategy is enough
        pytest.fail("no strategies in real-run RunResult")
