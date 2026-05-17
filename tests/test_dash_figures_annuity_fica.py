"""Regression tests for the audit's "Dash income figure / FICA-SDI"
findings.

The audit flagged that:

  * the year-by-year column groups in `_YEARLY_COLUMN_GROUPS` listed
    `wages`, `pension`, `ssn`, `qualified_dividends`,
    `interest_income` but omitted `annuity_taxable` even though the
    simulator records it; non-qualified annuity income was therefore
    invisible in the Dash year-by-year table; and
  * the Taxes tab's `taxes_panel` plotted AGI / federal / state /
    IRMAA but not FICA or state SDI, even though the simulator
    emits explicit `fica_oasdi`, `fica_medicare`,
    `fica_additional_medicare`, and `state_sdi` columns.

Both have been fixed; these tests pin the new behavior so it doesn't
regress when the figures module is refactored.
"""

from __future__ import annotations

import pandas as pd

from tax_optimizer import Config, Inputs
from tax_optimizer.inputs import AnnuityInputs
from tax_optimizer.simulator import simulate
from dash_app import figures


# ---------------------------------------------------------------------
# `annuity_taxable` is part of the `income` column group.
# ---------------------------------------------------------------------


class TestAnnuityInDetailColumns:
    def test_detail_columns_contains_annuity_taxable(self):
        cols = figures.detail_columns()
        assert "annuity_taxable" in cols, (
            f"`annuity_taxable` missing from detail_columns(); got {cols}"
        )

    def test_filter_keeps_annuity_taxable_when_present(self):
        # Build a small synthetic frame mimicking the simulator's
        # column shape.
        df = pd.DataFrame({
            "year": [2026, 2027],
            "spouse_a_age": [60, 61],
            "filing_status": ["mfj", "mfj"],
            "wages": [100_000.0, 0.0],
            "pension": [0.0, 0.0],
            "ssn": [0.0, 0.0],
            "annuity_taxable": [12_000.0, 12_500.0],
            "qualified_dividends": [0.0, 0.0],
            "interest_income": [100.0, 100.0],
            "rmd": [0.0, 0.0],
            "roth_conversion": [0.0, 0.0],
            "pretax_withdrawal": [0.0, 0.0],
            "roth_withdrawal": [0.0, 0.0],
            "taxable_withdrawal": [0.0, 0.0],
            "spending_need": [50_000.0, 50_000.0],
            "unfunded": [0.0, 0.0],
            "agi": [100_000.0, 12_500.0],
            "taxable_income": [80_000.0, 0.0],
            "federal_tax": [10_000.0, 0.0],
            "state_tax": [3_000.0, 0.0],
            "marginal": [0.22, 0.10],
            "medicare_base_premium": [0.0, 0.0],
            "irmaa": [0.0, 0.0],
            "pretax_balance": [200_000.0, 200_000.0],
            "roth_balance": [50_000.0, 50_000.0],
            "taxable_balance": [80_000.0, 80_000.0],
            "hsa_balance": [10_000.0, 10_000.0],
        })
        out = figures.filter_to_detail_cols(df)
        assert "annuity_taxable" in out.columns
        assert out["annuity_taxable"].tolist() == [12_000.0, 12_500.0]


# ---------------------------------------------------------------------
# `taxes_panel` adds FICA + SDI traces on the Dollars subplot.
# ---------------------------------------------------------------------


class TestTaxesPanelFICAAndSDI:
    def _df_with_fica_sdi(self) -> pd.DataFrame:
        # Default Config / Inputs has working spouses with W-2
        # wages, so FICA fires every working year. State default is
        # STATELESS (no SDI); we explicitly install CA so SDI is
        # nonzero too.
        from tax_optimizer.tax.state import CA
        cfg = Config(state_regime=CA, horizon_age=60)
        inp = Inputs(
            annuity=AnnuityInputs(
                balance_today=100_000,
                monthly_at_start=1_000,
                start_age=55,
                tax_kind="qualified",
            ),
        )
        return simulate(cfg, inp)

    def test_fica_trace_present(self):
        df = self._df_with_fica_sdi()
        fig = figures.taxes_panel(df)
        names = [t.name for t in fig.data]
        assert any("FICA" in n for n in names), (
            f"taxes_panel should expose a FICA trace; saw {names}"
        )

    def test_sdi_trace_present_when_state_has_sdi(self):
        df = self._df_with_fica_sdi()
        # Sanity: SDI column has some positive values.
        assert (df["state_sdi"] > 0).any(), (
            "expected positive state_sdi rows; setup invalid"
        )
        fig = figures.taxes_panel(df)
        names = [t.name for t in fig.data]
        assert "State SDI" in names, (
            f"taxes_panel should expose a State SDI trace; saw {names}"
        )

    def test_sdi_trace_absent_when_stateless(self):
        # STATELESS regime → state_sdi is identically 0, no trace.
        df = simulate(Config(horizon_age=55), Inputs())
        fig = figures.taxes_panel(df)
        names = [t.name for t in fig.data]
        assert "State SDI" not in names, (
            f"taxes_panel should not expose an SDI trace when SDI is "
            f"zero everywhere; saw {names}"
        )
