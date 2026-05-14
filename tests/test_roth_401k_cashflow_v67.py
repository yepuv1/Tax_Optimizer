"""Regression tests for the v6.7 Roth-401(k) cash-flow fix.

Pre-v6.7 bug
------------
`earned_cash` (used in `tax_paying_capacity`) and `cash_inflow` (used in
the year's final `delta`) both omitted the Roth-401(k) deferral as a
paycheck outflow:

  * Pretax 401(k) deferrals are already reflected because they reduce
    `wages_box1` (lower Box 1 → lower federal tax → naturally lower cash).
  * Mega-backdoor / after-tax 401(k) is explicitly subtracted from
    `delta` via `mega_backdoor_total`.
  * **Roth 401(k) deferrals were NOT subtracted** even though they're
    real paycheck dollars routed into `state.roth` at line 230.

The same `a_roth_contrib + b_roth_contrib` dollars therefore landed in
two places each year: `state.roth` (correctly) AND inflated `delta`
which flowed into `state.taxable`. Net effect: every Roth-401(k)
contribution was silently DOUBLE-COUNTED — and `tax_paying_capacity`
was overstated by the same amount, weakening the conversion-liquidity
guard.

Contract these tests pin down
-----------------------------
1. **Wealth conservation under Roth-401(k)-pct toggle.** Flipping a
   spouse from 100% Roth-401(k) to 0% Roth-401(k) (i.e. all pretax)
   on a constant-return scenario, with everything else held equal,
   should reduce terminal wealth by approximately
   `deferral × marginal_rate` (the extra tax paid TODAY for Roth
   treatment), and **never** create money out of thin air.

2. **`tax_paying_capacity` correctly excludes Roth-401(k).**
   The diagnostic column `roth_conv_tax_capacity` should drop by
   `a_roth_contrib + b_roth_contrib` when Roth-401(k) toggles on.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tax_optimizer.config import Config
from tax_optimizer.inputs import (
    CurrentContrib,
    CurrentIncome,
    HealthPremiums,
    Inputs,
    SocialSecurity,
    StartingBalances,
)
from tax_optimizer.simulator import simulate
from tax_optimizer.tax.state import STATELESS


def _wealth_setup():
    """Two working spouses, generous wages, no growth, no conversion.

    Returns a clean (Inputs, Config) pair we can toggle
    `spouse_*_roth_401k_pct` on. Zeroed growth/drag/dividends so cash
    flows are exact; no IRA / mega-backdoor / backdoor so the only
    contribution moving between buckets is the elective 401(k) deferral
    itself.
    """
    inp = Inputs(
        spouse_a_age_start=45,
        spouse_b_age_start=44,
        spouse_a_retire_age=65,
        spouse_b_retire_age=64,
        spouse_a_total_contrib_pct=0.10,
        spouse_b_total_contrib_pct=0.10,
        spouse_a_employer_match_rate=0.0,
        spouse_b_employer_match_rate=0.0,
        spouse_a_employer_match_max_pct=0.0,
        spouse_b_employer_match_max_pct=0.0,
        spouse_a_backdoor_roth=False,
        spouse_b_backdoor_roth=False,
        spouse_a_mega_backdoor_enabled=False,
        spouse_b_mega_backdoor_enabled=False,
        spouse_a_after_tax_401k_pct=0.0,
        spouse_b_after_tax_401k_pct=0.0,
        spouse_a_traditional_ira_contrib=0,
        spouse_b_traditional_ira_contrib=0,
        spouse_a_roth_ira_contrib=0,
        spouse_b_roth_ira_contrib=0,
        income=CurrentIncome(
            spouse_a_gross=150_000.0,
            spouse_b_gross=80_000.0,
            spouse_a_bonus=0.0,
            interest=0.0,
            capital_gains=0.0,
            dividends=0.0,
        ),
        contrib=CurrentContrib(hsa_family=0.0),
        ss=SocialSecurity(
            monthly_spouse_a=2_000,
            monthly_spouse_b=1_500,
            start_age_a=67,
            start_age_b=67,
        ),
        starting=StartingBalances(
            spouse_a_pretax_401k=500_000.0,
            taxable_brokerage=100_000.0,
            spouse_a_roth_ira=50_000.0,
        ),
        health_premiums=HealthPremiums(),
    )
    cfg = Config(
        horizon_age=46,             # one full year of working-life simulation
        start_year=2026,
        nominal_growth_rate=0.0,
        taxable_drag=0.0,
        taxable_equity_div_yield=0.0,
        taxable_bond_interest_yield=0.0,
        roth_conversion_target_bracket=0.0,
        roth_conversion_amount=0.0,
        state_regime=STATELESS,
    )
    return inp, cfg


def _total_wealth(row) -> float:
    return float(
        row["pretax_balance"]
        + row["roth_balance"]
        + row["taxable_balance"]
        + row["hsa_balance"]
    )


class TestWealthConservationUnderRothToggle:
    """Flipping Roth-401(k) % on/off must not create or destroy money.

    The household pays `deferral × marginal_rate` MORE tax today under
    Roth treatment (post-tax), so terminal wealth should be LOWER by
    exactly that amount vs. the pretax case. Pre-v6.7 this difference
    was OPPOSITE-SIGNED — the simulator silently gifted the household
    the full deferral as bonus taxable cash, on top of correctly
    crediting it to Roth.
    """

    def test_terminal_wealth_drops_by_tax_only_under_roth_election(self):
        inp, cfg = _wealth_setup()

        inputs_roth   = replace(inp, spouse_a_roth_401k_pct=1.0, spouse_b_roth_401k_pct=1.0)
        inputs_pretax = replace(inp, spouse_a_roth_401k_pct=0.0, spouse_b_roth_401k_pct=0.0)
        df_roth   = simulate(cfg, inputs_roth)
        df_pretax = simulate(cfg, inputs_pretax)

        wealth_roth   = _total_wealth(df_roth.iloc[0])
        wealth_pretax = _total_wealth(df_pretax.iloc[0])

        # Extra tax paid in the Roth case = Δ fed_tax + Δ state_tax.
        # State is zero (stateless regime); take it from federal alone.
        extra_tax = float(df_roth.iloc[0]["federal_tax"] - df_pretax.iloc[0]["federal_tax"])
        wealth_delta = wealth_roth - wealth_pretax

        # Total wealth difference must equal -extra_tax (within rounding).
        # Pre-v6.7 wealth_delta was POSITIVE (= +deferral - extra_tax),
        # i.e. wrong sign and wrong magnitude.
        assert wealth_delta == pytest.approx(-extra_tax, abs=1.0), (
            f"Wealth conservation violated: Roth wealth {wealth_roth:,.0f} − "
            f"pretax wealth {wealth_pretax:,.0f} = {wealth_delta:,.0f}, "
            f"expected ≈ -{extra_tax:,.0f} (extra tax paid)."
        )

    def test_pretax_and_roth_bucket_movements_offset(self):
        """The $X moved from pretax bucket to Roth bucket should match.

        This is the orthogonal "money goes to the right buckets" check —
        was already correct pre-v6.7, the bug was in the third leg
        (cash-flow `delta` flowing to taxable).
        """
        inp, cfg = _wealth_setup()
        inputs_roth   = replace(inp, spouse_a_roth_401k_pct=1.0, spouse_b_roth_401k_pct=1.0)
        inputs_pretax = replace(inp, spouse_a_roth_401k_pct=0.0, spouse_b_roth_401k_pct=0.0)
        df_roth   = simulate(cfg, inputs_roth)
        df_pretax = simulate(cfg, inputs_pretax)

        pretax_delta = df_roth.iloc[0]["pretax_balance"] - df_pretax.iloc[0]["pretax_balance"]
        roth_delta   = df_roth.iloc[0]["roth_balance"]   - df_pretax.iloc[0]["roth_balance"]
        # Movement should be equal and opposite (one bucket's loss is the
        # other's gain) and equal to the elective deferral total.
        assert pretax_delta == pytest.approx(-roth_delta, abs=1.0)
        assert roth_delta > 0  # Roth bucket gained dollars under Roth election

    def test_taxable_difference_equals_extra_tax_with_opposite_sign(self):
        """After the v6.7 fix, the only `taxable_balance` Δ between the
        two cases should be exactly -extra_tax (the Roth case had less
        cash surplus this year because it paid more tax). Pre-v6.7 the
        Δ was +deferral instead.
        """
        inp, cfg = _wealth_setup()
        inputs_roth   = replace(inp, spouse_a_roth_401k_pct=1.0, spouse_b_roth_401k_pct=1.0)
        inputs_pretax = replace(inp, spouse_a_roth_401k_pct=0.0, spouse_b_roth_401k_pct=0.0)
        df_roth   = simulate(cfg, inputs_roth)
        df_pretax = simulate(cfg, inputs_pretax)

        extra_tax = float(
            df_roth.iloc[0]["federal_tax"] - df_pretax.iloc[0]["federal_tax"]
        )
        taxable_delta = float(
            df_roth.iloc[0]["taxable_balance"] - df_pretax.iloc[0]["taxable_balance"]
        )
        assert taxable_delta == pytest.approx(-extra_tax, abs=1.0)


class TestTaxPayingCapacityRespectsRothDeferral:
    """`tax_paying_capacity` should NOT spuriously climb when Roth-401(k)-%
    toggles on. Pre-v6.7 capacity gained the full elective deferral
    (≈ $23k in this fixture) because `earned_cash` failed to subtract
    the Roth deferral as a paycheck outflow.

    Direction & magnitude check (not exact equality, because
    `base_tax_no_conv` differs between Roth and pretax variants in any
    year where bracket-fill or a fixed conversion lands on different
    underlying income lines):

      * Roth case must NOT have larger capacity than pretax case
        (the bug's sign).
      * |capacity gap| must be much smaller than the deferral itself.
        Pre-v6.7 the gap was ≈ +$23,000 (the full deferral); post-fix
        it should be in the low single-digit thousands (driven only by
        the small `base_tax_no_conv` difference, since `earned_cash`
        now nets to the same value between the two cases).
    """

    def test_capacity_does_not_climb_under_roth_election(self):
        inp, cfg = _wealth_setup()

        # Turn ON a bracket-fill so `roth_conv_tax_capacity` is populated.
        cfg = replace(cfg, roth_conversion_target_bracket=0.24)

        inputs_roth   = replace(inp, spouse_a_roth_401k_pct=1.0, spouse_b_roth_401k_pct=1.0)
        inputs_pretax = replace(inp, spouse_a_roth_401k_pct=0.0, spouse_b_roth_401k_pct=0.0)
        df_roth   = simulate(cfg, inputs_roth)
        df_pretax = simulate(cfg, inputs_pretax)

        cap_roth   = float(df_roth.iloc[0]["roth_conv_tax_capacity"])
        cap_pretax = float(df_pretax.iloc[0]["roth_conv_tax_capacity"])

        # Elective deferral total: 10% of ($150k + $80k) = $23,000.
        # The deferral is capped against `elective_deferral_cap(age)`
        # which at age 45 is $23,500 → no cap binding, so deferral = $23,000.
        deferral_total = 23_000.0

        # Direction: Roth case must NOT have a larger capacity.
        assert cap_roth <= cap_pretax + 1.0, (
            f"Roth case capacity {cap_roth:,.0f} > pretax case capacity "
            f"{cap_pretax:,.0f}. Pre-v6.7 the gap was ≈ +deferral; this "
            f"assertion catches a regression."
        )

        # Magnitude: gap must be MUCH smaller than the deferral itself.
        # Pre-v6.7 the gap was +$23,000 exactly; post-fix it should be
        # within $7,000 (deferral × top-of-bracket marginal rate buffer)
        # of zero in either direction.
        gap = abs(cap_pretax - cap_roth)
        assert gap < deferral_total * 0.5, (
            f"|capacity gap| {gap:,.0f} >= half the deferral ({deferral_total*0.5:,.0f}). "
            f"Roth case: {cap_roth:,.0f}, pretax case: {cap_pretax:,.0f}. "
            f"Pre-v6.7 this gap was exactly the deferral ($23k); a regression "
            f"would put it back near that level."
        )
