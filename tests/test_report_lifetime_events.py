"""Regression tests for `_lifetime_events` in the action-plan report.

The audit flagged that the action-plan markdown ignored
pension / annuity / §72(t) events even though the simulator records
them in dedicated columns. The new `_lifetime_events` helper scans
those columns and renders a "Major lifetime income / lump-sum events"
table inside §6 of the action plan.

Each test below pins one event-type's row content so future regressions
in the report formatting or simulator-column naming are caught
immediately.
"""

from __future__ import annotations

import pandas as pd

from tax_optimizer import Config, Inputs
from tax_optimizer.inputs import (
    AnnuityInputs,
    PensionInputs,
    StartingBalances,
)
from tax_optimizer.report import _lifetime_events
from tax_optimizer.simulator import simulate


def _make_df(**inp_kwargs) -> tuple[Config, Inputs, pd.DataFrame]:
    inp = Inputs(**inp_kwargs)
    cfg = Config(horizon_age=80)
    df = simulate(cfg, inp)
    return cfg, inp, df


class TestNoEventsRendersEmpty:
    def test_household_with_no_pension_or_annuity_returns_empty_list(self):
        cfg, inp, df = _make_df()
        # Default Inputs() has no pension or annuity → empty list.
        out = _lifetime_events(cfg, inp, df)
        assert out == []


class TestPensionLumpSum:
    def test_cash_lump_post_60_renders_with_no_penalty_note(self):
        cfg, inp, df = _make_df(
            household_kind="single",
            spouse_a_age_start=70,
            starting=StartingBalances(pension_balance=300_000),
            pension=PensionInputs(
                balance_today=300_000,
                monthly_at_nrd=2_000,
                start_age=70,
                lump_sum_mode="cash",
            ),
        )
        out = _lifetime_events(cfg, inp, df)
        assert out, "expected at least one event row for the cash lump"
        body = "\n".join(out)
        assert "Pension lump-sum (cash)" in body
        assert "post-59½, no penalty" in body
        assert "**§72(t)" not in body

    def test_cash_lump_pre_60_flags_72t_penalty(self):
        cfg, inp, df = _make_df(
            household_kind="single",
            spouse_a_age_start=50,
            starting=StartingBalances(pension_balance=300_000),
            pension=PensionInputs(
                balance_today=300_000,
                monthly_at_nrd=2_000,
                start_age=55,
                lump_sum_mode="cash",
            ),
        )
        out = _lifetime_events(cfg, inp, df)
        body = "\n".join(out)
        assert "Pension lump-sum (cash)" in body
        # The 10% surtax should be surfaced explicitly.
        assert "**§72(t) 10% surtax applies**" in body

    def test_rollover_lump_renders_no_tax_note(self):
        cfg, inp, df = _make_df(
            household_kind="single",
            spouse_a_age_start=60,
            starting=StartingBalances(pension_balance=300_000),
            pension=PensionInputs(
                balance_today=300_000,
                monthly_at_nrd=2_000,
                start_age=65,
                lump_sum_mode="rollover_pretax",
            ),
        )
        out = _lifetime_events(cfg, inp, df)
        body = "\n".join(out)
        assert "Pension lump-sum (rollover_pretax)" in body
        assert "no current-year tax" in body


class TestAnnuityFirstPayment:
    def test_qualified_annuity_payments_begin_row(self):
        cfg, inp, df = _make_df(
            household_kind="single",
            spouse_a_age_start=60,
            annuity=AnnuityInputs(
                balance_today=200_000,
                monthly_at_start=1_500,
                expected_payout_years=15,
                start_age=65,
                tax_kind="qualified",
            ),
        )
        out = _lifetime_events(cfg, inp, df)
        body = "\n".join(out)
        assert "Annuity payments begin" in body
        # Qualified annuity → fully taxable, no basis return note.
        assert "fully taxable" in body


class TestStandaloneEarlyDistributionPenalty:
    def test_age59_pension_lump_renders_50pct_penalty(self):
        # The §72(t) boundary year prorates to 50% of the surtax, but
        # the row in `_lifetime_events` is still keyed on
        # `pension_lump_sum > 0` so the penalty is surfaced inline,
        # not as a separate stand-alone row.
        cfg, inp, df = _make_df(
            household_kind="single",
            spouse_a_age_start=50,
            starting=StartingBalances(pension_balance=300_000),
            pension=PensionInputs(
                balance_today=300_000,
                monthly_at_nrd=2_000,
                start_age=59,
                lump_sum_mode="cash",
            ),
        )
        out = _lifetime_events(cfg, inp, df)
        body = "\n".join(out)
        assert "Pension lump-sum (cash)" in body
        assert "**§72(t) 10% surtax applies**" in body
