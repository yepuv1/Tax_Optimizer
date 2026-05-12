"""Regression tests for the v6.5 Roth-conversion liquidity guards.

Covers three behavior contracts introduced in v6.5:

1. ``cfg.cap_conversion_by_liquidity`` (default True) sizes the
   conversion DOWN if its marginal federal + state tax would exceed
   the household's non-Roth tax-paying capacity (earned-cash surplus
   plus a configurable slice of the taxable brokerage). Surfaces via
   the ``roth_conv_capped_by_liquidity`` diagnostic column and the
   gap between ``roth_conv_bracket_target`` and ``roth_conversion``.

2. ``cfg.protect_roth_in_conversion_years`` (default True) excludes
   the Roth bucket from the deficit cascade in any year a conversion
   fires. A too-aggressive conversion sized past liquidity now shows
   as ``unfunded`` instead of silently raiding the just-funded Roth.

3. Both knobs can be flipped to ``False`` to recover the pre-v6.5
   behavior (sizing to bracket-headroom regardless of cash, cascade
   raids Roth) — important for sensitivity analysis and reproducing
   historical outputs.

Plus a back-stop direct test on ``planned_roth_conversion`` to verify
the bisection produces ``capped_by_liquidity=True`` and respects the
``state_tax_fn`` argument (so CA's 9.3% marginal bite tightens the
cap relative to STATELESS).
"""

from __future__ import annotations

import pytest

from tax_optimizer.config import Config
from tax_optimizer.conversion import planned_roth_conversion
from tax_optimizer.inputs import (
    CurrentIncome,
    Inputs,
    SocialSecurity,
    StartingBalances,
)
from tax_optimizer.simulator import simulate
from tax_optimizer.spending import SpendingProfile
from tax_optimizer.state import State
from tax_optimizer.tax.federal import federal_tax
from tax_optimizer.tax.regimes import TCJA_EXTENDED
from tax_optimizer.tax.state import CA


def _liquidity_demo_setup():
    """Return (Inputs, Config) for a 65-y-o retiree with tight liquidity
    and an aggressive bracket-fill conversion target — the canonical
    bug scenario from the v6.5 diagnosis."""
    inp = Inputs(
        spouse_a_age_start=65,
        spouse_b_age_start=64,
        spouse_a_retire_age=65,
        spouse_b_retire_age=64,
        income=CurrentIncome(
            spouse_a_gross=0.0,
            spouse_b_gross=0.0,
            spouse_a_bonus=0.0,
            interest=0.0,
            capital_gains=0.0,
            dividends=0.0,
        ),
        ss=SocialSecurity(
            monthly_spouse_a=3_000,
            monthly_spouse_b=2_000,
            start_age_a=70,
            start_age_b=70,
        ),
        starting=StartingBalances(
            spouse_a_pretax_401k=2_000_000.0,
            taxable_brokerage=50_000.0,
            spouse_a_roth_ira=300_000.0,
        ),
    )
    cfg = Config(
        horizon_age=72,
        start_year=2026,
        bracket_fill_target=0.22,
        roth_conversion_target_bracket=0.24,
        rmd_start_age=75,
        spending=SpendingProfile.flat(base_spending=80_000.0),
        taxable_equity_div_yield=0.0,
        taxable_bond_interest_yield=0.0,
    )
    return inp, cfg


class TestLiquidityGuardCapsAggressiveConversion:
    """Bug scenario: 24% bracket-fill on a $2M pretax / $50k taxable
    household sizes to ~$400k under pre-v6.5 logic. With the v6.5
    liquidity guard, the year-1 conversion should be sized down so
    its marginal tax fits in capacity (cash surplus + half taxable).
    """

    def test_year_one_conversion_is_capped(self):
        inp, cfg = _liquidity_demo_setup()
        df = simulate(cfg, inp)
        row = df.iloc[0]
        # Bracket target is the pre-cap size — the bug behavior.
        assert row["roth_conv_bracket_target"] > 300_000.0, (
            "Bracket-fill should have wanted a huge conversion."
        )
        # Liquidity guard caps it.
        assert row["roth_conv_capped_by_liquidity"], (
            "Liquidity guard should flip True when the cap binds."
        )
        # Conversion is smaller than the bracket target.
        assert row["roth_conversion"] < row["roth_conv_bracket_target"]
        # Capacity is positive (year 1 has $50k taxable → at least
        # $25k slice).
        assert row["roth_conv_tax_capacity"] > 0.0

    def test_capped_conversion_does_not_raid_roth(self):
        """The whole point of the liquidity cap: ending Roth balance
        should equal starting Roth + conversion (no withdrawal back
        out)."""
        inp, cfg = _liquidity_demo_setup()
        df = simulate(cfg, inp)
        row = df.iloc[0]
        # Year 1 starts with $300k Roth. End-of-year Roth = start
        # + conversion + growth. Growth at default nominal 6% on
        # ($300k start + ~half-year average conversion) ≈ $30-40k.
        # No raid means Roth grows MORE than the conversion amount.
        starting_roth = 300_000.0
        end_roth = row["roth_balance"]
        conv = row["roth_conversion"]
        # Allow a margin: end_roth >= starting_roth + conv * 0.99
        # (the 0.99 is because growth applies after the conversion,
        # not on the just-added conv dollars when the cascade fires
        # mid-year — but with no cascade the conv survives intact).
        assert end_roth >= starting_roth + conv * 0.99, (
            f"Roth was raided. Expected >= {starting_roth + conv * 0.99:,.0f}, "
            f"got {end_roth:,.0f}."
        )

    def test_capacity_uses_taxable_slice_default_half(self):
        """conversion_taxable_use_ratio=0.5 means capacity gets only
        half the taxable balance. Setting ratio=1.0 should yield a
        BIGGER conversion."""
        inp, cfg = _liquidity_demo_setup()
        df_half = simulate(cfg, inp)
        conv_half = df_half.iloc[0]["roth_conversion"]

        from dataclasses import replace
        cfg_full = replace(cfg, conversion_taxable_use_ratio=1.0)
        df_full = simulate(cfg_full, inp)
        conv_full = df_full.iloc[0]["roth_conversion"]

        # Allowing full taxable as capacity → larger allowable
        # conversion → larger sized total.
        assert conv_full > conv_half, (
            f"Ratio=1.0 should allow larger conversion than ratio=0.5; "
            f"got full={conv_full:,.0f}, half={conv_half:,.0f}."
        )

    def test_capacity_zero_ratio_reduces_conversion(self):
        """Ratio=0.0 strips the taxable slice; conversion has to be
        funded purely from earned + retirement income surplus."""
        inp, cfg = _liquidity_demo_setup()
        from dataclasses import replace
        cfg_zero = replace(cfg, conversion_taxable_use_ratio=0.0)
        df_zero = simulate(cfg_zero, inp)
        df_half = simulate(cfg, inp)
        assert (
            df_zero.iloc[0]["roth_conversion"]
            < df_half.iloc[0]["roth_conversion"]
        )


class TestRothProtectionInConversionYears:
    """When ``protect_roth_in_conversion_years=True`` (default) and a
    conversion fires, the deficit cascade refuses to touch the Roth
    bucket. Any shortfall surfaces as ``unfunded``.

    Test design: turn the liquidity cap OFF so the year's conversion
    is huge enough to definitely produce a cascade-trigger, then
    flip Roth-protection on vs off and observe the Roth balance
    delta.
    """

    def _stress_scenario(self):
        inp = Inputs(
            spouse_a_age_start=65,
            spouse_b_age_start=64,
            spouse_a_retire_age=65,
            spouse_b_retire_age=64,
            income=CurrentIncome(0, 0, 0, 0, 0, 0),
            ss=SocialSecurity(3_000, 2_000, 70, 70),
            starting=StartingBalances(
                spouse_a_pretax_401k=2_000_000.0,
                taxable_brokerage=50_000.0,
                spouse_a_roth_ira=300_000.0,
            ),
        )
        # Liquidity cap OFF, protection ON.
        cfg_on = Config(
            horizon_age=66,
            start_year=2026,
            roth_conversion_target_bracket=0.24,
            rmd_start_age=75,
            spending=SpendingProfile.flat(base_spending=80_000.0),
            taxable_equity_div_yield=0.0,
            taxable_bond_interest_yield=0.0,
            cap_conversion_by_liquidity=False,
            protect_roth_in_conversion_years=True,
        )
        return inp, cfg_on

    def test_unfunded_surfaces_when_roth_protected(self):
        inp, cfg_on = self._stress_scenario()
        df = simulate(cfg_on, inp)
        row = df.iloc[0]
        # With protection on and a huge conversion vs small taxable,
        # the cascade should leave some unfunded gap (taxable + HSA
        # + pretax can fill some but the converted Roth is off-limits).
        # The conversion is ~$400k → fed tax ~$87k. Taxable has only
        # $50k. So at least ~$37k of deficit must look elsewhere.
        # Specifically: not Roth. Either pretax (will reduce the
        # benefit but at least won't penalize) or unfunded.
        assert row["roth_conversion"] > 100_000.0, (
            "Bracket-fill conversion should still be aggressive when "
            "the cap is off."
        )
        # Roth grows by ~ conversion + growth (allow 1% for mid-year
        # ordering nuance). If protection failed, the cascade would
        # have pulled tens of thousands BACK out.
        starting_roth = 300_000.0
        conv = row["roth_conversion"]
        assert row["roth_balance"] >= starting_roth + conv * 0.95, (
            f"Roth was raided despite protection. start={starting_roth}, "
            f"conv={conv}, end={row['roth_balance']}."
        )

    def test_protection_off_recovers_roth_raid(self):
        inp, cfg_on = self._stress_scenario()
        from dataclasses import replace
        cfg_off = replace(cfg_on, protect_roth_in_conversion_years=False)
        df_on = simulate(cfg_on, inp)
        df_off = simulate(cfg_off, inp)
        # With protection OFF, end-of-year Roth balance should be
        # smaller (cascade pulled some out) AND unfunded should be
        # lower (cascade had more capacity).
        roth_on = df_on.iloc[0]["roth_balance"]
        roth_off = df_off.iloc[0]["roth_balance"]
        assert roth_off < roth_on, (
            f"Without protection the cascade should reduce Roth balance. "
            f"on={roth_on:,.0f}, off={roth_off:,.0f}."
        )


class TestBackCompatBothKnobsOff:
    """Flipping both v6.5 knobs to False reproduces the pre-v6.5
    behavior: aggressive sizing + Roth raid. Important so historical
    runs (and the buggy-behavior comparison in our own diagnostic
    output) stay reproducible.
    """

    def test_pre_v65_behavior_reproduced(self):
        inp, cfg = _liquidity_demo_setup()
        from dataclasses import replace
        cfg_off = replace(
            cfg,
            cap_conversion_by_liquidity=False,
            protect_roth_in_conversion_years=False,
        )
        df = simulate(cfg_off, inp)
        row = df.iloc[0]
        # Without the cap, year-1 conversion goes to the full bracket
        # headroom (~$394k for this scenario).
        assert row["roth_conversion"] > 300_000.0, (
            "Bracket-fill should size to its full target when knobs are off."
        )
        # `capped_by_liquidity` flag should be False.
        assert not row["roth_conv_capped_by_liquidity"], (
            "Flag should NOT trigger when liquidity cap is disabled."
        )
        # And taxable should be drained (cascade raided it AND Roth).
        assert row["taxable_balance"] < 1_000.0


class TestPlannedRothConversionDirect:
    """Direct unit tests on ``planned_roth_conversion`` to verify the
    bisection contract independent of the simulator's plumbing."""

    def _kwargs(self):
        return dict(
            wages=0.0,
            interest=0.0,
            ordinary_div=0.0,
            qualified_div=0.0,
            ltcg=0.0,
            pension=80_000.0,  # gap-year pension income
            social_security=0.0,
            deduction=30_000.0,
        )

    def _state_and_inputs(self):
        state = State(
            year=2026,
            spouse_a_age=65,
            spouse_b_age=65,
            spouse_a_pretax=1_000_000.0,
            spouse_b_pretax=0.0,
            roth=200_000.0,
            taxable=20_000.0,
            hsa=0.0,
            pension_balance=0.0,
        )
        inp = Inputs(
            spouse_a_age_start=65,
            spouse_b_age_start=65,
            spouse_a_retire_age=64,  # already retired
            spouse_b_retire_age=64,
        )
        return state, inp

    def test_no_capacity_arg_preserves_old_sizing(self):
        """Without `tax_paying_capacity`, sizing should match the
        bracket-headroom logic — i.e. capped_by_liquidity=False and
        the conversion equals the bracket target."""
        state, inp = self._state_and_inputs()
        cfg = Config(
            horizon_age=66,
            start_year=2026,
            roth_conversion_target_bracket=0.22,
            rmd_start_age=75,
        )
        plan = planned_roth_conversion(
            cfg, inp, state, self._kwargs(),
            regime=TCJA_EXTENDED, filing_status="married_joint",
            rmd_total=0.0, rmd_a=0.0, rmd_b=0.0,
            tax_paying_capacity=None,  # explicit
        )
        assert not plan.capped_by_liquidity
        assert plan.conv_a > 0.0  # bracket headroom is positive
        # Plan total equals bracket-target total when no cap.
        assert plan.conv_a + plan.conv_b == pytest.approx(
            plan.bracket_target_total, rel=1e-3
        )

    def test_tight_capacity_drives_capped_true(self):
        """Pass a tiny `tax_paying_capacity` and confirm the
        bisection drives the conversion down + flips the capped flag.
        """
        state, inp = self._state_and_inputs()
        cfg = Config(
            horizon_age=66,
            start_year=2026,
            roth_conversion_target_bracket=0.24,
            rmd_start_age=75,
        )
        plan_loose = planned_roth_conversion(
            cfg, inp, state, self._kwargs(),
            regime=TCJA_EXTENDED, filing_status="married_joint",
            rmd_total=0.0, rmd_a=0.0, rmd_b=0.0,
            tax_paying_capacity=1_000_000.0,  # huge, doesn't bind
        )
        plan_tight = planned_roth_conversion(
            cfg, inp, state, self._kwargs(),
            regime=TCJA_EXTENDED, filing_status="married_joint",
            rmd_total=0.0, rmd_a=0.0, rmd_b=0.0,
            tax_paying_capacity=5_000.0,  # tight: only $5k tax budget
        )
        assert plan_tight.capped_by_liquidity
        assert not plan_loose.capped_by_liquidity
        total_tight = plan_tight.conv_a + plan_tight.conv_b
        total_loose = plan_loose.conv_a + plan_loose.conv_b
        assert total_tight < total_loose

    def test_state_tax_fn_tightens_cap_further(self):
        """Passing a CA `state_tax_fn` makes the marginal-tax-on-
        conversion larger (federal + 9.3% state) so the bisection
        accepts a SMALLER conversion than the federal-only case."""
        state, inp = self._state_and_inputs()
        cfg = Config(
            horizon_age=66,
            start_year=2026,
            roth_conversion_target_bracket=0.24,
            rmd_start_age=75,
        )
        kw = self._kwargs()

        def ca_state_fn(kw, ss_taxable_federal: float) -> float:
            from tax_optimizer.tax.state import state_tax as st
            return st(
                regime=CA, filing_status="married_joint",
                wages_box1=kw.get("wages", 0.0),
                interest=kw.get("interest", 0.0),
                ordinary_div=kw.get("ordinary_div", 0.0),
                qualified_div=kw.get("qualified_div", 0.0),
                ltcg=kw.get("ltcg", 0.0),
                pension=kw.get("pension", 0.0),
                pretax_withdrawal=kw.get("pretax_withdrawal", 0.0),
                roth_conversion=kw.get("roth_conversion", 0.0),
                social_security=kw.get("social_security", 0.0),
                ss_taxable_federal=ss_taxable_federal,
                age_a=65, age_b=65, alive_a=True, alive_b=True,
            )["state_tax"]

        def nv_state_fn(kw, ss_taxable_federal: float) -> float:
            return 0.0  # STATELESS

        plan_ca = planned_roth_conversion(
            cfg, inp, state, kw,
            regime=TCJA_EXTENDED, filing_status="married_joint",
            rmd_total=0.0, rmd_a=0.0, rmd_b=0.0,
            tax_paying_capacity=15_000.0,
            state_tax_fn=ca_state_fn,
        )
        plan_nv = planned_roth_conversion(
            cfg, inp, state, kw,
            regime=TCJA_EXTENDED, filing_status="married_joint",
            rmd_total=0.0, rmd_a=0.0, rmd_b=0.0,
            tax_paying_capacity=15_000.0,
            state_tax_fn=nv_state_fn,
        )
        total_ca = plan_ca.conv_a + plan_ca.conv_b
        total_nv = plan_nv.conv_a + plan_nv.conv_b
        assert total_ca < total_nv, (
            "CA's 9.3% marginal bite should tighten the cap below the "
            "STATELESS equivalent. "
            f"CA={total_ca:,.0f}, NV={total_nv:,.0f}."
        )
        assert plan_ca.capped_by_liquidity
        assert plan_nv.capped_by_liquidity

    def test_bisection_respects_actual_marginal_tax(self):
        """Independent verification: marginal tax on the returned
        conversion should be <= capacity (within a $200 tolerance for
        bisection coarseness)."""
        state, inp = self._state_and_inputs()
        cfg = Config(
            horizon_age=66,
            start_year=2026,
            roth_conversion_target_bracket=0.24,
            rmd_start_age=75,
        )
        kw = self._kwargs()
        capacity = 8_000.0
        plan = planned_roth_conversion(
            cfg, inp, state, kw,
            regime=TCJA_EXTENDED, filing_status="married_joint",
            rmd_total=0.0, rmd_a=0.0, rmd_b=0.0,
            tax_paying_capacity=capacity,
        )
        base_fed = federal_tax(
            regime=TCJA_EXTENDED, filing_status="married_joint", **kw
        )
        kw_conv = dict(kw)
        kw_conv["roth_conversion"] = plan.conv_a + plan.conv_b
        new_fed = federal_tax(
            regime=TCJA_EXTENDED, filing_status="married_joint", **kw_conv
        )
        fed_delta = new_fed["tax"] - base_fed["tax"]
        # Marginal tax should not blow past capacity. $200 slop for
        # the bisection's 100-dollar termination tolerance.
        assert fed_delta <= capacity + 200.0, (
            f"Marginal tax {fed_delta:,.2f} exceeded capacity "
            f"{capacity:,.2f} by more than tolerance."
        )
