"""Regression tests for v6.6 §125 cafeteria-plan health premiums.

Three behavior contracts:

1. ``inputs.health_premiums`` (medical + dental + vision per spouse)
   reduces **federal Box 1 wages** → lowers federal income tax by
   roughly (marginal rate × premium). Same direction & rough size as
   a traditional 401(k) contribution.

2. With ``cfg.section125_reduces_fica_wages = True`` (the v6.6
   default), the premiums also reduce **FICA wages** (OASDI +
   Medicare) and **SDI wages** — matching real Box 3 / Box 5
   payroll. Households in CA see SDI drop too (FTB conforms).
   The HSA contribution, also §125, gets the same FICA/SDI
   treatment under this flag.

3. Per-spouse gating: a non-working spouse's premium is ignored
   (no W-2 → no §125 deduction). Premiums clamp at the spouse's
   gross wages (you can't pre-tax-deduct more than you earn).
"""

from __future__ import annotations

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
from tax_optimizer.tax.state import CA


def _two_earner_setup(**hp_kwargs):
    """Two-earner household, no HSA, no bonus, large enough wages that
    OASDI cap binding doesn't muddy the §125 math."""
    inp = Inputs(
        spouse_a_age_start=45,
        spouse_b_age_start=45,
        spouse_a_retire_age=65,
        spouse_b_retire_age=65,
        spouse_a_total_contrib_pct=0.0,
        spouse_b_total_contrib_pct=0.0,
        income=CurrentIncome(
            spouse_a_gross=120_000.0,
            spouse_b_gross=80_000.0,
            spouse_a_bonus=0.0,
            interest=0.0,
            capital_gains=0.0,
            dividends=0.0,
        ),
        contrib=CurrentContrib(hsa_family=0.0),
        ss=SocialSecurity(
            monthly_spouse_a=2_500,
            monthly_spouse_b=1_500,
            start_age_a=70,
            start_age_b=70,
        ),
        starting=StartingBalances(taxable_brokerage=50_000.0),
        health_premiums=HealthPremiums(**hp_kwargs),
    )
    cfg = Config(horizon_age=50, start_year=2026)
    return cfg, inp


# ---------------------------------------------------------------------------
# Federal income tax behavior
# ---------------------------------------------------------------------------


class TestFederalTaxReduction:
    """§125 premiums reduce wages_box1 → reduce federal tax by roughly
    marginal_rate × premium."""

    def test_zero_premium_baseline(self) -> None:
        cfg, inp = _two_earner_setup()
        df = simulate(cfg, inp)
        row = df.iloc[0]
        # Sanity: with no premiums, all the §125 diagnostic columns are 0.
        assert row["health_premium_a"] == 0.0
        assert row["health_premium_b"] == 0.0
        assert row["health_premium_total"] == 0.0

    def test_premium_reduces_federal_tax(self) -> None:
        # Spouse A: $6k medical + $1k dental + $400 vision = $7,400.
        # Spouse B: $1.2k vision = $1,200. Total household $8,600.
        cfg, inp_with = _two_earner_setup(
            spouse_a_medical=6_000.0,
            spouse_a_dental=1_000.0,
            spouse_a_vision=400.0,
            spouse_b_vision=1_200.0,
        )
        cfg_base, inp_base = _two_earner_setup()
        df_with = simulate(cfg, inp_with)
        df_base = simulate(cfg_base, inp_base)

        row_with = df_with.iloc[0]
        row_base = df_base.iloc[0]

        # Premium total is $8,600 and surfaces in the diagnostic columns.
        assert row_with["health_premium_a"] == pytest.approx(7_400.0)
        assert row_with["health_premium_b"] == pytest.approx(1_200.0)
        assert row_with["health_premium_total"] == pytest.approx(8_600.0)

        # Federal tax must drop. Household is in the 22% MFJ bracket
        # at this income, so tax reduction ≈ 22% × $8,600 = $1,892.
        # Allow a ±$200 band for QDIV/LTCG interaction and rounding.
        tax_reduction = row_base["federal_tax"] - row_with["federal_tax"]
        assert tax_reduction == pytest.approx(8_600 * 0.22, abs=200.0)

        # The `wages` column is GROSS (Box 3-ish); it doesn't move
        # with §125. Verify that explicitly.
        assert row_base["wages"] == pytest.approx(row_with["wages"], abs=1.0)


# ---------------------------------------------------------------------------
# FICA behavior (v6.6 default: §125 reduces FICA wages)
# ---------------------------------------------------------------------------


class TestFICAReduction:
    """With cfg.section125_reduces_fica_wages = True (the default),
    M/D/V premiums reduce FICA wages — saving 7.65% of premium
    (OASDI + base Medicare) below the OASDI cap."""

    def test_premium_reduces_fica_default_on(self) -> None:
        cfg, inp_with = _two_earner_setup(
            spouse_a_medical=5_000.0,
            spouse_b_medical=3_000.0,
        )
        cfg_base, inp_base = _two_earner_setup()

        df_with = simulate(cfg, inp_with)
        df_base = simulate(cfg_base, inp_base)

        row_with = df_with.iloc[0]
        row_base = df_base.iloc[0]

        # FICA on $8,000 of §125 savings at well-under-OASDI-cap wages
        # = (6.2% + 1.45%) × $8,000 = $612. Allow ±$5 for indexing.
        fica_reduction = row_base["fica"] - row_with["fica"]
        assert fica_reduction == pytest.approx(8_000 * 0.0765, abs=5.0)

    def test_premium_does_NOT_reduce_fica_when_flag_off(self) -> None:
        """Back-compat: setting `section125_reduces_fica_wages=False`
        reproduces the pre-v6.6 approximation where FICA is computed
        on gross wages regardless of §125 deductions."""
        cfg_off, inp = _two_earner_setup(
            spouse_a_medical=5_000.0,
            spouse_b_medical=3_000.0,
        )
        cfg_off = Config(
            horizon_age=50,
            start_year=2026,
            section125_reduces_fica_wages=False,
        )
        cfg_base = Config(
            horizon_age=50,
            start_year=2026,
            section125_reduces_fica_wages=False,
        )
        _, inp_base = _two_earner_setup()

        df_with = simulate(cfg_off, inp)
        df_base = simulate(cfg_base, inp_base)

        row_with = df_with.iloc[0]
        row_base = df_base.iloc[0]

        # FICA delta should be ~0 (premiums don't touch FICA when flag is off).
        assert row_with["fica"] == pytest.approx(row_base["fica"], abs=1.0)

        # But federal tax STILL drops (Box 1 reduction is unaffected
        # by the flag — that's the always-on behavior).
        assert row_with["federal_tax"] < row_base["federal_tax"]


# ---------------------------------------------------------------------------
# State income tax + SDI (CA)
# ---------------------------------------------------------------------------


class TestStateTaxAndSDIReduction:
    """California conforms to §125 federal treatment for both income tax
    AND SDI base wages."""

    def test_premium_reduces_ca_state_tax(self) -> None:
        cfg_base, inp_base = _two_earner_setup()
        cfg_with, inp_with = _two_earner_setup(
            spouse_a_medical=10_000.0,
        )
        # Override state regime to CA for both scenarios.
        cfg_base = Config(horizon_age=50, start_year=2026, state_regime=CA)
        cfg_with = Config(horizon_age=50, start_year=2026, state_regime=CA)
        df_base = simulate(cfg_base, inp_base)
        df_with = simulate(cfg_with, inp_with)

        # CA marginal rate at this income ~9.3% → reduction ≈ $930.
        ca_tax_reduction = df_base.iloc[0]["state_tax"] - df_with.iloc[0]["state_tax"]
        assert ca_tax_reduction > 0
        assert ca_tax_reduction == pytest.approx(10_000 * 0.093, abs=200.0)

    def test_premium_reduces_ca_sdi(self) -> None:
        cfg_base, inp_base = _two_earner_setup()
        cfg_with, inp_with = _two_earner_setup(
            spouse_a_medical=10_000.0,
        )
        # Pin start_year to 2024 (1.1% SDI rate) so the test pins the
        # reduction without depending on the EDD-published year-by-year
        # schedule (1.2% in 2025, 0.9% in 2026).
        cfg_base = Config(horizon_age=50, start_year=2024, state_regime=CA)
        cfg_with = Config(horizon_age=50, start_year=2024, state_regime=CA)
        df_base = simulate(cfg_base, inp_base)
        df_with = simulate(cfg_with, inp_with)

        # 2024 SDI 1.1% × $10k = $110.
        sdi_reduction = df_base.iloc[0]["state_sdi"] - df_with.iloc[0]["state_sdi"]
        assert sdi_reduction == pytest.approx(10_000 * 0.011, abs=2.0)


# ---------------------------------------------------------------------------
# Per-spouse gating + clamp behavior
# ---------------------------------------------------------------------------


class TestGatingAndClamp:
    """A spouse's premium only applies during their own working years
    and is clamped at their gross wages."""

    def test_premium_zero_after_retirement(self) -> None:
        cfg, inp = _two_earner_setup(
            spouse_a_medical=5_000.0,
            spouse_b_medical=3_000.0,
        )
        # Spouse A retires at 50 (year 5), B at 55 (year 10).
        inp = Inputs(
            spouse_a_age_start=45,
            spouse_b_age_start=45,
            spouse_a_retire_age=50,
            spouse_b_retire_age=55,
            spouse_a_total_contrib_pct=0.0,
            spouse_b_total_contrib_pct=0.0,
            income=inp.income,
            contrib=inp.contrib,
            ss=inp.ss,
            starting=inp.starting,
            health_premiums=inp.health_premiums,
        )
        cfg = Config(horizon_age=60, start_year=2026)
        df = simulate(cfg, inp)

        # Year 0 (age 45): both working, both premiums apply.
        row0 = df.iloc[0]
        assert row0["health_premium_a"] == pytest.approx(5_000.0)
        assert row0["health_premium_b"] == pytest.approx(3_000.0)

        # Year 6 (age 51): A retired, B still working.
        row6 = df.iloc[6]
        assert row6["health_premium_a"] == 0.0
        assert row6["health_premium_b"] == pytest.approx(3_000.0)

        # Year 12 (age 57): both retired.
        row12 = df.iloc[12]
        assert row12["health_premium_a"] == 0.0
        assert row12["health_premium_b"] == 0.0

    def test_premium_clamped_at_gross_wages(self) -> None:
        """If a premium exceeds the spouse's W-2 wages, clamp to wages.
        Real-world this can't happen on a paystub (can't deduct more
        than you make); we hard-clamp so the simulator doesn't go
        negative on Box 1."""
        inp = Inputs(
            spouse_a_age_start=45,
            spouse_b_age_start=45,
            spouse_a_retire_age=65,
            spouse_b_retire_age=65,
            income=CurrentIncome(
                spouse_a_gross=5_000.0,   # tiny W-2
                spouse_a_bonus=0.0,       # no bonus
                spouse_b_gross=0.0,
            ),
            contrib=CurrentContrib(hsa_family=0.0),
            health_premiums=HealthPremiums(
                spouse_a_medical=8_000.0,  # exceeds wages
            ),
            ss=SocialSecurity(),
            starting=StartingBalances(taxable_brokerage=50_000.0),
        )
        cfg = Config(horizon_age=50, start_year=2026)
        df = simulate(cfg, inp)
        row = df.iloc[0]
        # Premium clamped to $5,000 (gross wages).
        assert row["health_premium_a"] == pytest.approx(5_000.0)
        # wages_box1 floor of 0 (no negative wages).
        assert row["wages"] >= 0.0

    def test_premium_ignored_for_non_working_spouse(self) -> None:
        """Premium on a $0-wage spouse drops to zero (no W-2 to deduct
        from). Common scenario: one spouse retired, other still working
        and carrying both on their plan."""
        inp = Inputs(
            spouse_a_age_start=45,
            spouse_b_age_start=68,  # already retired
            spouse_a_retire_age=65,
            spouse_b_retire_age=65,
            income=CurrentIncome(
                spouse_a_gross=120_000.0,
                spouse_b_gross=0.0,
            ),
            contrib=CurrentContrib(hsa_family=0.0),
            health_premiums=HealthPremiums(
                spouse_a_medical=6_000.0,
                spouse_b_medical=2_000.0,  # ignored (B not working)
            ),
            ss=SocialSecurity(start_age_a=70, start_age_b=70),
            starting=StartingBalances(taxable_brokerage=50_000.0),
        )
        cfg = Config(horizon_age=50, start_year=2026)
        df = simulate(cfg, inp)
        row = df.iloc[0]
        assert row["health_premium_a"] == pytest.approx(6_000.0)
        assert row["health_premium_b"] == 0.0  # B not working


# ---------------------------------------------------------------------------
# Cash flow consistency: premiums leave the household.
# ---------------------------------------------------------------------------


class TestCashFlowConsistency:
    """The premium leaves the household (paid to the insurer). Cash
    inflow uses wages_box1 which already nets the premium. So the
    end-of-year balances should fall by roughly the premium minus
    the federal/FICA/state tax savings."""

    def test_premium_reduces_taxable_growth(self) -> None:
        cfg_with, inp_with = _two_earner_setup(
            spouse_a_medical=10_000.0,
        )
        cfg_base, inp_base = _two_earner_setup()
        df_with = simulate(cfg_with, inp_with)
        df_base = simulate(cfg_base, inp_base)

        # End of year-0 taxable balance with premium < without premium.
        # Premium $10,000 net of tax savings:
        #   federal: 22% × $10k = $2,200 (MFJ at $200k wages → 22% bracket)
        #   FICA:    7.65% × $10k = $765   (well under OASDI cap, §125 on)
        #   state:   $0 (stateless)
        # Net household cash outflow ≈ $7,035 (modulo portfolio-return
        # interaction on the residual contribution to taxable).
        cash_delta = df_base.iloc[0]["taxable_balance"] - df_with.iloc[0]["taxable_balance"]
        assert cash_delta == pytest.approx(7_035.0, abs=500.0)
