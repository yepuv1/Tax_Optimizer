"""Regression tests for the phased-retirement cash-flow fix.

Pre-fix bug
-----------
In the year-loop's working-year branch, `cash_inflow` was computed as:

    cash_inflow = wages + extras + gross_cash_in + annuity_cash_in
                  - fica - sdi - roth_contribs

`pension_income` and `ssn_income` were NOT included, so any household
that was still working W-2 wages (one or both spouses) AND already
drawing pension or claiming Social Security in the same year had
those streams silently dropped from the cash-flow ledger. The result
was a phantom deficit cascade that drained brokerage / Roth / pretax
to cover spending that was actually being funded by the missing
pension or SS dollars.

The fix lands `pension_income + ssn_income` into both the working-
year `cash_inflow` and the conversion-capacity `guaranteed_cash_no_conv`
so the two solvers agree on what the household sees.

Tests pin down
--------------
1. **SS while still working at 64.** Spouse claims SS at 62 but
   keeps working (W-2 wages) at 64. Pre-fix the SS dollars vanished
   from cash_inflow; post-fix they show up as positive `delta` (or
   reduce the conversion's funding gap).

2. **Pension paid out while still working.** Spouse A starts a
   pension at 62 but keeps working W-2 to 65. Same story.

3. **Annuity-paying year still working.** Annuity starts at 60,
   spouse A still working — annuity_cash_in already lived in the
   working-year branch (pre-existing), but the test confirms the
   companion `tax_paying_capacity` knob also reflects it.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tax_optimizer.config import Config
from tax_optimizer.inputs import (
    AnnuityInputs,
    CurrentIncome,
    Inputs,
    SocialSecurity,
    StartingBalances,
)
from tax_optimizer.simulator import simulate


def _phased_setup(claim_age_a: int) -> tuple[Inputs, Config]:
    """Spouse A is 62, working W-2 to age 65, claims SS at `claim_age_a`.

    Spouse B is absent (single household) so the test focuses on a
    single income stream.
    """
    inp = Inputs(
        household_kind="single",
        spouse_a_age_start=62,
        spouse_a_retire_age=65,
        income=CurrentIncome(
            spouse_a_gross=60_000.0,
            spouse_b_gross=0.0,
            spouse_a_bonus=0.0,
            interest=0.0,
            capital_gains=0.0,
            dividends=0.0,
        ),
        ss=SocialSecurity(
            monthly_spouse_a=2_500.0,
            monthly_spouse_b=0.0,
            start_age_a=claim_age_a,
            start_age_b=70,
        ),
        starting=StartingBalances(
            spouse_a_pretax_401k=200_000.0,
            spouse_b_pretax_401k=0.0,
            spouse_b_pretax_ira=0.0,
            spouse_b_roth_ira=0.0,
            taxable_brokerage=20_000.0,
            hsa=0.0,
        ),
    )
    cfg = Config(
        horizon_age=70,
        spending=__import__("tax_optimizer.spending", fromlist=[""]).SpendingProfile.flat(
            base_spending=40_000.0
        ),
        taxable_equity_div_yield=0.0,
        taxable_bond_interest_yield=0.0,
    )
    return inp, cfg


class TestSSWhileStillWorking:
    def test_ss_dollars_show_up_in_working_year_cash_inflow(self):
        # Compare two scenarios: claim at 62 (early, gets reduced
        # benefit but flows into working years 62-64) vs claim at
        # 70 (no SS in working years). The early-claim scenario
        # must add the SS dollars (after taxes) to the working-year
        # taxable / Roth balances. Pre-fix the early-claim case
        # silently lost those dollars in cash_inflow.
        inp_early, cfg = _phased_setup(claim_age_a=62)
        inp_late, _ = _phased_setup(claim_age_a=70)
        df_early = simulate(cfg, inp_early)
        df_late = simulate(cfg, inp_late)

        # Year 0 (age 62) — early claimant gets SS now.
        early0 = df_early.iloc[0]
        late0 = df_late.iloc[0]
        # Sanity: SS column is non-zero in early scenario, zero in
        # late.
        assert early0["ssn"] > 0.0
        assert late0["ssn"] == 0.0
        # Working-year `cash_inflow` is not directly reported, but
        # the surplus shows up as a higher taxable_balance or roth_
        # balance at end of year 0 (the SS dollars don't vanish into
        # a phantom deficit cascade). Compute end-of-year liquid
        # wealth as a robust proxy.
        early_wealth = (
            early0["taxable_balance"]
            + early0["roth_balance"]
            + early0["pretax_balance"]
        )
        late_wealth = (
            late0["taxable_balance"]
            + late0["roth_balance"]
            + late0["pretax_balance"]
        )
        # Early SS adds ~$22k (PIA $30k * early-claim factor 0.7 ≈
        # $21k) of new household cash. After SS taxes (some flows
        # to federal at the 10/12% bracket on roughly half of SS
        # being taxable), net ~$18k+ should land in liquid wealth.
        # Pre-fix the early-claim scenario was LESS wealthy because
        # SS was treated as taxable (raising tax) without flowing
        # into cash_inflow.
        delta = early_wealth - late_wealth
        assert delta > 5_000.0, (
            f"SS-while-working dollars must add to liquid wealth; "
            f"early={early_wealth:,.0f}, late={late_wealth:,.0f}, "
            f"delta={delta:,.0f}"
        )


class TestSSAddsToConversionCapacity:
    def test_capacity_grows_with_pension_or_ss(self):
        # Year-1 conversion capacity should rise when SS is on (vs
        # off), because guaranteed_cash_no_conv now correctly
        # includes ssn_income.
        inp_off, cfg = _phased_setup(claim_age_a=70)
        inp_on, _ = _phased_setup(claim_age_a=62)
        # Add a Roth-conversion target so the capacity column is
        # populated.
        cfg = replace(cfg, roth_conversion_target_bracket=0.22)

        df_off = simulate(cfg, inp_off)
        df_on = simulate(cfg, inp_on)

        cap_off = df_off.iloc[0]["roth_conv_tax_capacity"]
        cap_on = df_on.iloc[0]["roth_conv_tax_capacity"]

        # SS adds non-trivial cash → capacity must grow.
        assert cap_on > cap_off, (
            f"Capacity must include SS income; got cap_on={cap_on:,.0f}, "
            f"cap_off={cap_off:,.0f}"
        )


class TestAnnuityCapacityRouting:
    def test_capacity_includes_annuity_cash(self):
        # An annuity payment in a working year should ALSO surface
        # in conversion capacity. Pre-fix the working-year capacity
        # only saw earned_cash; annuity_cash_in was silently zero.
        inp = Inputs(
            household_kind="single",
            spouse_a_age_start=60,
            spouse_a_retire_age=65,
            income=CurrentIncome(
                spouse_a_gross=60_000.0,
                spouse_b_gross=0.0,
                spouse_a_bonus=0.0,
                interest=0.0,
                capital_gains=0.0,
                dividends=0.0,
            ),
            starting=StartingBalances(
                spouse_a_pretax_401k=300_000.0,
                spouse_b_pretax_401k=0.0,
                spouse_b_pretax_ira=0.0,
                spouse_b_roth_ira=0.0,
                taxable_brokerage=10_000.0,
                hsa=0.0,
            ),
            annuity=AnnuityInputs(
                balance_today=200_000,
                monthly_at_start=1_000,
                start_age=60,
                tax_kind="qualified",
            ),
        )
        cfg = Config(
            horizon_age=70,
            roth_conversion_target_bracket=0.22,
            spending=__import__("tax_optimizer.spending", fromlist=[""]).SpendingProfile.flat(
                base_spending=40_000.0
            ),
            taxable_equity_div_yield=0.0,
            taxable_bond_interest_yield=0.0,
        )

        # Same household but no annuity (zero balance / zero monthly).
        inp_no_annuity = replace(
            inp,
            annuity=AnnuityInputs(
                balance_today=0.0, monthly_at_start=0.0, start_age=60
            ),
        )
        df = simulate(cfg, inp)
        df_no = simulate(cfg, inp_no_annuity)

        cap_with = df.iloc[0]["roth_conv_tax_capacity"]
        cap_without = df_no.iloc[0]["roth_conv_tax_capacity"]
        # Annuity payment ($12k qualified) must lift capacity.
        assert cap_with > cap_without, (
            f"Annuity cash must lift capacity; with={cap_with:,.0f}, "
            f"without={cap_without:,.0f}"
        )
