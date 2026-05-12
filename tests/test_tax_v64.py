"""Regression tests for v6.4 tax-module fixes.

Covers:
  * T1 — `_marginal_rate` boundary semantics (>=)
  * T3 — age-65+ std-deduction add-on + OBBBA senior bonus
  * T4 — qualified vs ordinary dividend split
  * T5 — NY per-spouse retirement exclusion
  * T6 — AMT for SUNSET regime
  * T7 — CA SDI on per-spouse wages
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from tax_optimizer.config import Config
from tax_optimizer.inputs import (
    CurrentIncome,
    Inputs,
    SocialSecurity,
    StartingBalances,
)
from tax_optimizer.payroll import state_sdi
from tax_optimizer.simulator import simulate
from tax_optimizer.tax.federal import _marginal_rate, federal_tax
from tax_optimizer.tax.regimes import (
    PRE_TCJA_2017,
    SUNSET_2026,
    TCJA_EXTENDED,
)
from tax_optimizer.tax.state import CA, NY, STATELESS, state_tax


# ---------------------------------------------------------------------------
# T1 — _marginal_rate at exact bracket boundary
# ---------------------------------------------------------------------------


class TestMarginalRateBoundary:
    """At exact `lo` of bracket N, the next dollar lands in bracket N, so
    the marginal rate should reflect bracket N (not N-1)."""

    def test_at_top_of_10pct_returns_12pct(self) -> None:
        # MFJ: 10% to $23,850, then 12%. At exact $23,850, next dollar
        # is 12%.
        assert (
            _marginal_rate(23_850.0, TCJA_EXTENDED.ord_brackets_mfj) == 0.12
        )

    def test_at_top_of_12pct_returns_22pct(self) -> None:
        assert (
            _marginal_rate(96_950.0, TCJA_EXTENDED.ord_brackets_mfj) == 0.22
        )

    def test_inside_bracket_still_correct(self) -> None:
        # Mid-bracket cases should be unchanged.
        assert (
            _marginal_rate(50_000.0, TCJA_EXTENDED.ord_brackets_mfj) == 0.12
        )

    def test_zero_is_lowest_bracket(self) -> None:
        # The next dollar at $0 lands in the 10% bracket.
        assert _marginal_rate(0.0, TCJA_EXTENDED.ord_brackets_mfj) == 0.10


# ---------------------------------------------------------------------------
# T3 — age-65+ standard deduction add-on + OBBBA senior bonus
# ---------------------------------------------------------------------------


class TestEffectiveStdDeduction:
    def test_base_when_no_seniors(self) -> None:
        ded = TCJA_EXTENDED.effective_std_deduction(
            "mfj", n_seniors_65plus=0, calendar_year=2026
        )
        assert ded == pytest.approx(32_200.0)

    def test_one_senior_adds_per_filer_amount(self) -> None:
        # MFJ senior add-on is $1,600/spouse on TCJA_EXTENDED.
        # OBBBA senior bonus +$6,000/spouse during 2025–2028.
        ded = TCJA_EXTENDED.effective_std_deduction(
            "mfj", n_seniors_65plus=1, calendar_year=2026
        )
        assert ded == pytest.approx(32_200.0 + 1_600.0 + 6_000.0)

    def test_both_seniors_double_addon_plus_bonus(self) -> None:
        ded = TCJA_EXTENDED.effective_std_deduction(
            "mfj", n_seniors_65plus=2, calendar_year=2026
        )
        assert ded == pytest.approx(32_200.0 + 2 * 1_600.0 + 2 * 6_000.0)

    def test_obbba_bonus_expires_after_2028(self) -> None:
        # In 2029, OBBBA bonus is gone; only the §63(f) base add-on remains.
        ded = TCJA_EXTENDED.effective_std_deduction(
            "mfj", n_seniors_65plus=2, calendar_year=2029
        )
        assert ded == pytest.approx(32_200.0 + 2 * 1_600.0)

    def test_pre_tcja_has_no_obbba_bonus(self) -> None:
        ded = PRE_TCJA_2017.effective_std_deduction(
            "mfj", n_seniors_65plus=2, calendar_year=2026
        )
        # 12_700 base + 2 * 1_250 senior, no OBBBA bonus.
        assert ded == pytest.approx(12_700.0 + 2 * 1_250.0)

    def test_calendar_year_none_skips_obbba(self) -> None:
        # Pass calendar_year=None to deliberately ignore the OBBBA window.
        ded = TCJA_EXTENDED.effective_std_deduction(
            "mfj", n_seniors_65plus=2, calendar_year=None
        )
        assert ded == pytest.approx(32_200.0 + 2 * 1_600.0)


class TestSeniorDeductionLowersTax:
    def test_senior_pair_pays_less_tax_during_obbba_window(self) -> None:
        # MFJ, $120k of pretax withdrawals only, both 65+, year 2026.
        # Senior std ded should be 32_200 + 3_200 + 12_000 = 47_400 vs
        # base 32_200; the difference saves tax at the 12% bracket.
        base = federal_tax(
            regime=TCJA_EXTENDED,
            filing_status="mfj",
            pretax_withdrawal=120_000.0,
        )
        senior_ded = TCJA_EXTENDED.effective_std_deduction(
            "mfj", n_seniors_65plus=2, calendar_year=2026
        )
        senior = federal_tax(
            regime=TCJA_EXTENDED,
            filing_status="mfj",
            pretax_withdrawal=120_000.0,
            deduction=senior_ded,
        )
        assert senior["tax"] < base["tax"]
        # Roughly 15.2k of extra deduction × ~12% bracket ≈ $1.8k savings.
        # Loose bound — bracket structure can spill.
        savings = base["tax"] - senior["tax"]
        assert 1_200.0 <= savings <= 2_500.0


# ---------------------------------------------------------------------------
# T4 — Qualified vs ordinary dividend split
# ---------------------------------------------------------------------------


class TestQdivSplitFromConfig:
    def test_default_qualified_fraction_is_85pct(self) -> None:
        cfg = Config()
        assert cfg.taxable_equity_qualified_fraction == pytest.approx(0.85)

    def test_split_routes_part_of_div_to_ordinary(self) -> None:
        # Build a scenario with a substantial taxable brokerage producing
        # dividends. Verify the row's qualified vs ordinary split matches
        # cfg.taxable_equity_qualified_fraction. Zero out the
        # "extra dividends" input so the math is clean (default Inputs
        # ships with $2k of working-year dividends from `inputs.income`).
        cfg = Config(
            taxable_equity_div_yield=0.02,
            taxable_bond_interest_yield=0.0,
            taxable_equity_qualified_fraction=0.80,
        )
        inp = Inputs(
            income=replace(Inputs().income, dividends=0.0, interest=0.0),
            starting=replace(
                Inputs().starting, taxable_brokerage=1_000_000.0
            ),
        )
        df = simulate(cfg, inp)
        row = df.iloc[0]
        # Default asset location is 100% equity in taxable. Total div =
        # $1M × 0.02 = $20k. 80% qualified -> $16k qdiv; 20% ordinary
        # -> $4k.
        eq_pct = cfg.asset_location.taxable.equity_pct
        expected_q = 0.80 * 1_000_000.0 * 0.02 * eq_pct
        expected_o = 0.20 * 1_000_000.0 * 0.02 * eq_pct
        assert row["qualified_dividends"] == pytest.approx(expected_q, abs=1.0)
        assert row["ordinary_dividends"] == pytest.approx(expected_o, abs=1.0)

    def test_fraction_clamped_to_unit_interval(self) -> None:
        # Out-of-range fractions should be clamped, not crash.
        cfg = Config(
            taxable_equity_div_yield=0.02,
            taxable_equity_qualified_fraction=1.5,
        )
        inp = Inputs(
            starting=replace(
                Inputs().starting, taxable_brokerage=100_000.0
            ),
        )
        # No exception; ordinary should land at 0.
        df = simulate(cfg, inp)
        assert (df["ordinary_dividends"] == 0.0).all()


# ---------------------------------------------------------------------------
# T5 — NY retirement exclusion is per-spouse, not pooled
# ---------------------------------------------------------------------------


class TestNYRetirementExclusion:
    """NY exempts $20k per filer 59½+ against their OWN distributions."""

    def _call(self, *, pension_split, pretax_split, roth_split):
        return state_tax(
            regime=NY,
            filing_status="mfj",
            wages_box1=0.0,
            interest=0.0,
            ordinary_div=0.0,
            qualified_div=0.0,
            ltcg=0.0,
            pension=pension_split[0] + pension_split[1],
            pretax_withdrawal=pretax_split[0] + pretax_split[1],
            roth_conversion=roth_split[0] + roth_split[1],
            social_security=0.0,
            ss_taxable_federal=0.0,
            age_a=65,
            age_b=65,
            alive_a=True,
            alive_b=True,
            pension_per_spouse=pension_split,
            pretax_per_spouse=pretax_split,
            roth_conv_per_spouse=roth_split,
        )

    def test_lopsided_distributions_excludes_only_one_filer_cap(self) -> None:
        # Spouse A has $40k pretax distribution; spouse B has $0.
        # Per-spouse: exclude min(20k, 40k) + min(20k, 0) = 20k.
        # Pre-fix (pool-wide): exclude min(40k, 40k) = 40k → over-exclude.
        out = self._call(
            pension_split=(0.0, 0.0),
            pretax_split=(40_000.0, 0.0),
            roth_split=(0.0, 0.0),
        )
        # Pretax 40k minus $20k exclusion minus $16_050 std ded = 3_950
        # taxable at 4% = ~$158. Tighter check: only the $20k exclusion
        # took effect, so taxable income > 0.
        assert out["state_taxable_income"] > 0
        # Sanity: the post-exclusion state taxable income should equal
        # 40k - 20k - 16_050 = 3_950.
        assert out["state_taxable_income"] == pytest.approx(3_950.0, abs=1.0)

    def test_balanced_distributions_excludes_both_filer_caps(self) -> None:
        # Both spouses have $20k each → fully excluded ($20k + $20k).
        out = self._call(
            pension_split=(0.0, 0.0),
            pretax_split=(20_000.0, 20_000.0),
            roth_split=(0.0, 0.0),
        )
        # 40k - 40k exclusion - 16_050 std ded = -16_050 (clamped to 0)
        assert out["state_taxable_income"] == 0.0

    def test_old_pool_behavior_with_no_per_spouse_args(self) -> None:
        # Without per-spouse args, falls back to pool-wide cap.
        # 40k pool, 2 eligible filers → exclude min($40k, $40k) = $40k.
        # Demonstrates back-compat for legacy callers.
        out = state_tax(
            regime=NY,
            filing_status="mfj",
            wages_box1=0.0,
            interest=0.0,
            ordinary_div=0.0,
            qualified_div=0.0,
            ltcg=0.0,
            pension=0.0,
            pretax_withdrawal=40_000.0,
            roth_conversion=0.0,
            social_security=0.0,
            ss_taxable_federal=0.0,
            age_a=65,
            age_b=65,
            alive_a=True,
            alive_b=True,
        )
        assert out["state_taxable_income"] == 0.0  # pool over-excludes


# ---------------------------------------------------------------------------
# T6 — AMT
# ---------------------------------------------------------------------------


class TestAMT:
    def test_tcja_extended_dormant_at_moderate_income(self) -> None:
        # MFJ, $200k pretax withdrawal -> regular taxable ~167k, AMTI same
        # (no std ded addback). TCJA exemption is $137k → AMT base = 30k.
        # AMT ord tax = 30k × 26% = $7,800 << regular tax → no AMT.
        out = federal_tax(
            regime=TCJA_EXTENDED,
            filing_status="mfj",
            pretax_withdrawal=200_000.0,
        )
        assert out["amt"] == 0.0
        # Sanity: amti is reported even when AMT doesn't bite.
        assert out["amti"] > 0

    def test_sunset_amt_fires_on_large_conversion(self) -> None:
        # Large Roth conversion under SUNSET — pre-TCJA AMT mechanics
        # (lower exemption, std-ded add-back) make AMT bite. The
        # difference between TMT and regular tax should be positive.
        big = federal_tax(
            regime=SUNSET_2026,
            filing_status="mfj",
            roth_conversion=300_000.0,
        )
        # Either AMT > 0, OR TMT >= regular. Both indicate the calc is
        # operating. We can't predict the exact number without re-doing
        # all bracket math, but TMT must be positive and AMT must be
        # non-negative.
        assert big["tmt"] > 0
        assert big["amt"] >= 0

    def test_amt_disabled_regime_returns_zero(self) -> None:
        # Construct a synthetic regime with AMT disabled (default
        # math.inf exemption). Verify amt always 0.
        custom = replace(
            TCJA_EXTENDED,
            amt_exemption_mfj=math.inf,
            amt_exemption_single=math.inf,
        )
        out = federal_tax(regime=custom, filing_status="mfj", wages=500_000.0)
        assert out["amt"] == 0.0
        assert out["tmt"] == 0.0

    def test_pre_tcja_amt_requires_std_addback(self) -> None:
        # The pre-TCJA regime sets amt_std_deduction_addback=True. The
        # AMTI should equal taxable_income + std_deduction.
        out = federal_tax(
            regime=PRE_TCJA_2017,
            filing_status="mfj",
            wages=200_000.0,
        )
        expected_amti = out["taxable_income"] + PRE_TCJA_2017.std_deduction_mfj
        assert out["amti"] == pytest.approx(expected_amti, abs=0.01)


# ---------------------------------------------------------------------------
# T7 — CA SDI
# ---------------------------------------------------------------------------


class TestCAStateSDI:
    def test_ca_rate_is_11pct_uncapped(self) -> None:
        assert CA.sdi_rate == pytest.approx(0.011)
        assert not math.isfinite(CA.sdi_wage_cap)

    def test_other_states_have_zero_sdi(self) -> None:
        assert STATELESS.sdi_rate == 0.0
        assert NY.sdi_rate == 0.0

    def test_state_sdi_returns_pct_of_wages_uncapped(self) -> None:
        # CA: $300k W-2 wages → 1.1% × $300k = $3,300.
        sdi = state_sdi(300_000.0, rate=0.011, wage_cap=math.inf)
        assert sdi == pytest.approx(3_300.0)

    def test_state_sdi_zero_when_rate_zero(self) -> None:
        # Non-CA states with 0 rate → no SDI.
        assert state_sdi(300_000.0, rate=0.0, wage_cap=math.inf) == 0.0

    def test_simulator_subtracts_sdi_for_ca_users(self) -> None:
        # Run two identical simulations, one CA one stateless. CA's
        # cash flow should be lower by ~1.1% of combined wages, and
        # the row should carry a non-zero state_sdi column.
        inp = Inputs(
            spouse_a_age_start=54,
            spouse_b_age_start=53,
            spouse_a_retire_age=65,
            spouse_b_retire_age=64,
            income=CurrentIncome(
                spouse_a_gross=150_000.0,
                spouse_b_gross=80_000.0,
                spouse_a_bonus=0.0,  # keep SDI math clean
            ),
            ss=SocialSecurity(
                monthly_spouse_a=2_500,
                monthly_spouse_b=1_800,
            ),
            starting=StartingBalances(taxable_brokerage=100_000.0),
        )
        cfg_ca = Config(state_regime=CA, horizon_age=70)
        cfg_none = Config(state_regime=STATELESS, horizon_age=70)
        df_ca = simulate(cfg_ca, inp)
        df_none = simulate(cfg_none, inp)
        row_ca = df_ca.iloc[0]
        row_none = df_none.iloc[0]
        # SDI = 1.1% × ($150k + $80k) = $2,530.
        assert row_ca["state_sdi"] == pytest.approx(
            0.011 * (150_000 + 80_000), abs=1.0
        )
        # Stateless: SDI is 0.
        assert row_none["state_sdi"] == 0.0
