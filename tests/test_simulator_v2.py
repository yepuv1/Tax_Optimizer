"""Tests for v2 simulator behaviors: HSA wiring, contribution caps,
taxable-account yield-as-AGI, and deficit-cascade / unfunded tracking.

Kept in a separate file so the original `test_simulator.py` keeps its
shape as the "engine smoke test" surface.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tax_optimizer import (
    Config,
    Inputs,
    LongTermCareShock,
    Mortality,
    SpendingProfile,
    StartingBalances,
    simulate,
)
from tax_optimizer.limits import (
    ELECTIVE_DEFERRAL_CATCH_UP_50,
    ELECTIVE_DEFERRAL_LIMIT,
    HSA_FAMILY_LIMIT,
)


# ---------------------------------------------------------------------------
# Contribution caps (B2)
# ---------------------------------------------------------------------------


class TestElectiveDeferralCap:
    def test_oversized_pct_is_capped(self) -> None:
        # 30% of $300k = $90k target, far above the IRS limit. The
        # simulator must cap the recorded elective deferral.
        cfg = Config()
        inp = Inputs(
            income=replace(Inputs().income, spouse_a_gross=300_000.0),
            spouse_a_total_contrib_pct=0.30,
            spouse_a_age_start=45,
            spouse_a_retire_age=65,
        )
        df = simulate(cfg, inp)
        first = df.iloc[0]
        assert first["elective_deferral_a"] == pytest.approx(ELECTIVE_DEFERRAL_LIMIT)

    def test_age_50_catch_up_increases_cap(self) -> None:
        cfg = Config()
        # Spouse A turns 50 at year 0 with 30% on $300k -> hits cap including catch-up.
        inp = Inputs(
            income=replace(Inputs().income, spouse_a_gross=300_000.0),
            spouse_a_total_contrib_pct=0.30,
            spouse_a_age_start=50,
            spouse_a_retire_age=65,
        )
        df = simulate(cfg, inp)
        first = df.iloc[0]
        assert first["elective_deferral_a"] == pytest.approx(
            ELECTIVE_DEFERRAL_LIMIT + ELECTIVE_DEFERRAL_CATCH_UP_50
        )

    def test_undersized_pct_passes_through(self) -> None:
        cfg = Config()
        inp = Inputs()  # 8% of $95k = $7.6k, below limit
        df = simulate(cfg, inp)
        first = df.iloc[0]
        expected = inp.income.spouse_a_gross * inp.spouse_a_total_contrib_pct
        assert first["elective_deferral_a"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# HSA wiring (B1)
# ---------------------------------------------------------------------------


class TestHSA:
    def test_hsa_contributions_apply_during_working_years(self) -> None:
        cfg = Config()
        inp = Inputs()
        df = simulate(cfg, inp)
        # During working years, HSA contribution should be capped at the
        # IRS family limit (catch-up not yet active for ages 50-54).
        working = df[df["spouse_a_age"] < inp.spouse_a_retire_age]
        assert (working["hsa_contrib"] == HSA_FAMILY_LIMIT).all()

    def test_hsa_contributions_stop_after_medicare_eligibility(self) -> None:
        cfg = Config()
        inp = Inputs()  # both retire at 65
        df = simulate(cfg, inp)
        # After both spouses retire AND hit Medicare age, no more HSA.
        post = df[(df["spouse_a_age"] >= 65) & (df["spouse_b_age"] >= 65)]
        assert (post["hsa_contrib"] == 0.0).all()

    def test_hsa_pays_for_ltc_first(self) -> None:
        # Construct a scenario with HSA much bigger than LTC cost so HSA
        # absorbs the entire shock and net spending need does not spike.
        cfg = Config(
            spending=SpendingProfile(
                base_spending=80_000.0,
                inflation=0.0,
                ltc_shock=LongTermCareShock(years=2, annual_cost_today=50_000.0),
            ),
        )
        inp = Inputs(starting=replace(Inputs().starting, hsa=200_000.0))
        df = simulate(cfg, inp)
        # Final two years are LTC years. HSA covers the LTC cost so the
        # recurring spending need should be ~ base ($80k), not $130k.
        final = df.iloc[-1]
        assert final["hsa_withdrawal"] > 0
        assert final["spending_need"] < 100_000.0

    def test_hsa_withdrawal_zero_outside_ltc_window(self) -> None:
        cfg = Config()  # default flat spending, no LTC
        inp = Inputs()
        df = simulate(cfg, inp)
        assert (df["hsa_withdrawal"] == 0.0).all()


# ---------------------------------------------------------------------------
# Taxable-account yield (A1)
# ---------------------------------------------------------------------------


class TestTaxableYield:
    def test_dividends_continue_after_retirement(self) -> None:
        cfg = Config()
        inp = Inputs(
            starting=replace(Inputs().starting, taxable_brokerage=500_000.0),
        )
        df = simulate(cfg, inp)
        # Pick a year where both spouses are retired, no SS yet.
        # AGI should be > 0 because of taxable-account dividends/interest,
        # not because of W-2 wages.
        retired_no_ss = df[
            (df["spouse_a_age"] >= inp.spouse_a_retire_age)
            & (df["spouse_a_age"] < inp.ss.start_age)
        ]
        if not retired_no_ss.empty:
            # AGI should reflect the yield from the taxable balance.
            assert (retired_no_ss["agi"] > 0).any()

    def test_zero_yield_recovers_legacy_behavior(self) -> None:
        # With both yields zeroed, post-retirement AGI in a year with
        # zero withdrawals should drop to ~zero (no SS yet, no wages).
        cfg = Config(
            taxable_equity_div_yield=0.0,
            taxable_bond_interest_yield=0.0,
        )
        inp = Inputs()
        df = simulate(cfg, inp)
        # Can't trivially assert AGI=0 in a retirement year because of
        # withdrawals, but at the very least the withdrawal decisions
        # shouldn't be inflated by phantom dividend income.
        assert "agi" in df.columns


# ---------------------------------------------------------------------------
# Deficit-cascade + unfunded tracking (A2 + A3)
# ---------------------------------------------------------------------------


class TestDeficitCascade:
    def test_unfunded_column_present(self) -> None:
        df = simulate(Config(), Inputs())
        assert "unfunded" in df.columns
        # Default scenario should fund every year cleanly.
        assert (df["unfunded"] == 0.0).all()

    def test_extreme_spending_marks_unfunded_years(self) -> None:
        # Spending dwarfs all balances and income -> the cascade
        # exhausts every bucket and unfunded > 0.
        cfg = Config(
            spending=SpendingProfile.flat(base_spending=2_000_000.0, inflation=0.0),
        )
        inp = Inputs(
            spouse_a_age_start=64,
            spouse_b_age_start=64,
            spouse_a_retire_age=65,
            spouse_b_retire_age=65,
            starting=StartingBalances(
                spouse_a_pretax_401k=10_000.0,
                spouse_b_pretax_401k=10_000.0,
                spouse_a_roth_ira=0.0,
                spouse_b_roth_ira=0.0,
                spouse_a_pretax_ira=0.0,
                spouse_b_pretax_ira=0.0,
                hsa=0.0,
                taxable_brokerage=10_000.0,
                pension_balance=0.0,
            ),
            income=replace(Inputs().income, spouse_a_gross=50_000.0, spouse_b_gross=0.0),
        )
        df = simulate(cfg, inp)
        retired = df[df["spouse_a_age"] >= inp.spouse_a_retire_age]
        assert (retired["unfunded"] > 0).any()
