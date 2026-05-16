"""Behavioral tests for the new ``inputs.pension.lump_sum_mode`` knob.

Mirrors the qualified-annuity lump-sum tests in
``tests/test_simulator_annuity.py``: pensions are always qualified
(no basis, no exclusion ratio), so the behavior set is just two
modes — rollover and cash — with the §72(t) 10% surtax under 59½.
"""

from __future__ import annotations

import pytest

from tax_optimizer import Config, Inputs
from tax_optimizer.inputs import PensionInputs, StartingBalances
from tax_optimizer.simulator import simulate


def _row(df, age):
    sub = df[df["spouse_a_age"] == age]
    assert len(sub) == 1, f"expected one row at age {age}; got {len(sub)}"
    return sub.iloc[0]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_bad_lump_sum_mode_raises(self):
        with pytest.raises(ValueError, match="lump_sum_mode"):
            Inputs(pension=PensionInputs(lump_sum_mode="dump"))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Defaults: lump_sum_mode='none' preserves legacy monthly behavior
# ---------------------------------------------------------------------------


class TestDefaultPreservesMonthly:
    def test_default_is_none(self):
        inp = Inputs()
        assert inp.pension.lump_sum_mode == "none"

    def test_existing_monthly_pension_unchanged(self):
        # A scenario with a monthly pension and no lump-sum election
        # should produce the same pension column as the legacy
        # behavior — and certainly no lump-sum row.
        inp = Inputs(
            household_kind="single",
            spouse_a_age_start=60,
            starting=StartingBalances(pension_balance=300_000),
            pension=PensionInputs(
                balance_today=300_000,
                monthly_at_nrd=2_000,
                start_age=65,
                # lump_sum_mode default = 'none'
            ),
        )
        df = simulate(Config(horizon_age=75), inp)
        r65 = _row(df, 65)
        assert r65["pension"] > 0  # monthly payment kicks in
        assert r65["pension_lump_sum"] == 0.0
        assert r65["pension_lump_sum_event"] == ""


# ---------------------------------------------------------------------------
# Lump-sum: rollover_pretax
# ---------------------------------------------------------------------------


class TestRolloverPretax:
    def test_balance_moves_to_pretax_ira_no_tax(self):
        inp = Inputs(
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
        df = simulate(Config(horizon_age=75), inp)
        r64 = _row(df, 64)
        r65 = _row(df, 65)
        assert r65["pension_lump_sum"] > 0
        assert r65["pension_lump_sum_event"] == "rollover_pretax"
        assert r65["pension_balance"] == 0.0
        assert r65["pension"] == 0.0
        assert r65["early_distribution_penalty"] == 0.0
        # Pretax IRA balance jumps substantially (net of any same-year
        # spending withdrawal). We assert a qualitative jump rather
        # than the full lump-sum amount.
        delta = r65["pretax_a_balance"] - r64["pretax_a_balance"]
        assert delta > 100_000

    def test_no_subsequent_pension_income(self):
        inp = Inputs(
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
        df = simulate(Config(horizon_age=75), inp)
        post = df[df["spouse_a_age"] >= 66]
        assert (post["pension"] == 0).all()
        assert (post["pension_balance"] == 0).all()


# ---------------------------------------------------------------------------
# Lump-sum: cash
# ---------------------------------------------------------------------------


class TestCash:
    def test_full_balance_ordinary_no_penalty_post_60(self):
        inp = Inputs(
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
        df = simulate(Config(horizon_age=80), inp)
        r70 = _row(df, 70)
        assert r70["pension_lump_sum"] > 0
        assert r70["pension_lump_sum_event"] == "cash"
        # Pension income for the year includes the cash distribution.
        assert r70["pension"] >= r70["pension_lump_sum"]
        assert r70["early_distribution_penalty"] == 0.0

    def test_pre_60_surtax_is_10_percent_of_lump_sum(self):
        inp = Inputs(
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
        df = simulate(Config(horizon_age=70), inp)
        r55 = _row(df, 55)
        assert r55["pension_lump_sum"] > 0
        # 10% on the full lump-sum amount.
        assert r55["early_distribution_penalty"] == pytest.approx(
            0.10 * r55["pension_lump_sum"], rel=1e-6
        )

    def test_no_subsequent_pension_income(self):
        inp = Inputs(
            household_kind="single",
            spouse_a_age_start=60,
            starting=StartingBalances(pension_balance=300_000),
            pension=PensionInputs(
                balance_today=300_000,
                monthly_at_nrd=2_000,
                start_age=62,
                lump_sum_mode="cash",
            ),
        )
        df = simulate(Config(horizon_age=75), inp)
        post = df[df["spouse_a_age"] > 62]
        assert (post["pension"] == 0).all()
        assert (post["pension_lump_sum"] == 0).all()


# ---------------------------------------------------------------------------
# Lump-sum gate fires only once
# ---------------------------------------------------------------------------


class TestLumpSumLatch:
    def test_rollover_does_not_re_fire(self):
        inp = Inputs(
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
        df = simulate(Config(horizon_age=75), inp)
        # Exactly one year shows a non-empty event string.
        events = df[df["pension_lump_sum_event"] != ""]
        assert len(events) == 1
        assert events.iloc[0]["spouse_a_age"] == 65
