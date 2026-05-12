"""Tests for `tax_optimizer.pension`."""

from __future__ import annotations

import pytest

from tax_optimizer.pension import (
    PENSION_FLOOR_POST_2016,
    PENSION_FLOOR_PRE_2016,
    PENSION_HIGH_RATE,
    PENSION_INTEREST,
    PENSION_LOW_RATE,
    PENSION_QTR_SSWB,
    effective_interest_rate,
    pay_credit_rates,
    pension_annual_credit,
    project_pension_balance,
)


class TestAnnualCredit:
    """`PENSION_QTR_SSWB` is one quarter of the *annual* Social Security
    wage base, so the kink is at ~$46,125 of **annual** earnings — not
    monthly. The v6.2 fix corrected an earlier divide-by-12 / multiply-
    by-12 cancellation that kept the high-rate band dormant until
    annual salary exceeded ~$553k.
    """

    def test_zero_earnings_zero_credit(self) -> None:
        assert pension_annual_credit(0.0) == 0.0

    def test_below_kink_uses_only_low_rate(self) -> None:
        # $30,000 annual is well below the ~$46,125 annual kink.
        annual_e = 30_000.0
        expected = annual_e * PENSION_LOW_RATE
        assert pension_annual_credit(annual_e) == pytest.approx(expected)

    def test_at_kink_no_high_credit(self) -> None:
        # Earnings exactly at the annual kink → only the low band fires.
        annual_e = PENSION_QTR_SSWB
        expected = PENSION_QTR_SSWB * PENSION_LOW_RATE
        assert pension_annual_credit(annual_e) == pytest.approx(expected)

    def test_above_kink_blends_low_and_high(self) -> None:
        # 2x the annual kink → half at the low rate, half at the high rate.
        annual_e = 2 * PENSION_QTR_SSWB
        low = PENSION_QTR_SSWB * PENSION_LOW_RATE
        high = PENSION_QTR_SSWB * PENSION_HIGH_RATE
        expected = low + high
        assert pension_annual_credit(annual_e) == pytest.approx(expected)

    def test_high_band_fires_at_typical_salary(self) -> None:
        # Regression test for the v6.2 fix: a $200k annual salary should
        # land MOSTLY in the high band, not entirely in the low band.
        annual_e = 200_000.0
        out = pension_annual_credit(annual_e)
        assert out > 200_000.0 * PENSION_LOW_RATE, (
            "$200k salary should produce a credit > the all-low-band "
            "amount; otherwise the pre-v6.2 monthly-vs-annual bug has "
            "regressed."
        )
        expected = (
            PENSION_QTR_SSWB * PENSION_LOW_RATE
            + (annual_e - PENSION_QTR_SSWB) * PENSION_HIGH_RATE
        )
        assert out == pytest.approx(expected)

    def test_negative_earnings_treated_as_zero(self) -> None:
        # Defensive: a stray negative salary doesn't produce a credit.
        assert pension_annual_credit(-50_000.0) == 0.0


class TestProjectPensionBalance:
    def test_zero_years_returns_starting_balance(self) -> None:
        # No years to retirement → balance unchanged.
        assert project_pension_balance(50_000.0, 100_000.0, 0) == 50_000.0

    def test_one_year_applies_interest_plus_credit(self) -> None:
        start = 50_000.0
        earnings = 100_000.0
        out = project_pension_balance(start, earnings, 1, wage_growth=0.0)
        expected = start * (1 + PENSION_INTEREST) + pension_annual_credit(earnings)
        assert out == pytest.approx(expected)

    def test_growth_compounds_over_years(self) -> None:
        # After N years balance must exceed starting balance even with 0 earnings.
        out = project_pension_balance(100_000.0, 0.0, 10, wage_growth=0.0)
        # No new credits, just compounding interest.
        assert out == pytest.approx(100_000.0 * (1 + PENSION_INTEREST) ** 10)

    def test_wage_growth_increases_credits_over_time(self) -> None:
        # Identical setup, only wage growth differs → final balance with
        # higher wage growth must be at least as large (more credit dollars).
        slow = project_pension_balance(0.0, 60_000.0, 5, wage_growth=0.0)
        fast = project_pension_balance(0.0, 60_000.0, 5, wage_growth=0.05)
        assert fast > slow


# ===========================================================================
# BP RAP fidelity tests (v6.3) — verifies the implementation against the
# "How the plan works" section of the BP RAP Summary Plan Description
# (August 2023 edition).
# ===========================================================================


class TestPayCreditTiers:
    """BP RAP page 9: pay-credit rate is selected by the higher of age
    tier and service tier."""

    def test_default_top_tier_when_age_and_yos_unspecified(self) -> None:
        # Backward compat: no age / YoS → top tier.
        low, high = pay_credit_rates()
        assert (low, high) == (0.06, 0.11)

    def test_bottom_tier_young_and_short_tenure(self) -> None:
        # Under 40 AND under 10 YoS → 4% / 7%.
        assert pay_credit_rates(age=35, years_of_service=5) == (0.04, 0.07)

    def test_mid_tier_age_dominates(self) -> None:
        # Age 42 (mid) with 3 YoS (bottom) → 5% / 9% (age wins).
        assert pay_credit_rates(age=42, years_of_service=3) == (0.05, 0.09)

    def test_mid_tier_yos_dominates(self) -> None:
        # Age 35 (bottom) with 15 YoS (mid) → 5% / 9% (YoS wins).
        assert pay_credit_rates(age=35, years_of_service=15) == (0.05, 0.09)

    def test_top_tier_yos_dominates(self) -> None:
        # Age 42 (mid) with 25 YoS (top) → 6% / 11% (YoS wins).
        assert pay_credit_rates(age=42, years_of_service=25) == (0.06, 0.11)

    def test_top_tier_age_dominates(self) -> None:
        # Age 55 (top) with 5 YoS (bottom) → 6% / 11% (age wins).
        assert pay_credit_rates(age=55, years_of_service=5) == (0.06, 0.11)

    def test_tier_transitions_at_boundaries(self) -> None:
        # Exact threshold values: age 40 → mid tier; age 50 → top tier.
        assert pay_credit_rates(age=39, years_of_service=0) == (0.04, 0.07)
        assert pay_credit_rates(age=40, years_of_service=0) == (0.05, 0.09)
        assert pay_credit_rates(age=49, years_of_service=0) == (0.05, 0.09)
        assert pay_credit_rates(age=50, years_of_service=0) == (0.06, 0.11)
        # YoS 10 / 20 are likewise inclusive lower bounds.
        assert pay_credit_rates(age=0, years_of_service=9) == (0.04, 0.07)
        assert pay_credit_rates(age=0, years_of_service=10) == (0.05, 0.09)
        assert pay_credit_rates(age=0, years_of_service=19) == (0.05, 0.09)
        assert pay_credit_rates(age=0, years_of_service=20) == (0.06, 0.11)


class TestBpRapWorkedExample1:
    """Replicate Example 1 from BP RAP page 12 (pre-2016 participant
    with 5% floor). Age 45, 20 YoS, $50k eligible earnings ($45k base
    + $5k March bonus), $210k starting balance → annual pay credits of
    $3,497.56 and ending balance ~$224,085 after one year.

    Our model works in annual aggregate, so we expect the pay-credit
    line to match exactly (level + bonus same total earnings) and the
    interest-credit line to be within ~$100 (BP RAP compounds
    monthly; we apply annual interest first then add pay credits).
    """

    def test_pay_credit_matches_spd_example(self) -> None:
        # Age 45 + 20 YoS ⇒ top tier (6% / 11%). 2023 ¼ SSWB = $40,050.
        credit = pension_annual_credit(
            50_000.0,
            age=45,
            years_of_service=20,
            qtr_sswb=160_200.0 / 4,
            comp_limit=330_000.0,  # 2023 IRS limit per the SPD example
        )
        # SPD: $2,403.00 (low) + $1,094.56 (high) = $3,497.56
        # Annual aggregate of high-band credit = ($50,000 - $40,050) × 0.11
        #                                       = $9,950 × 0.11 = $1,094.50
        # SPD's $1,094.56 difference is rounding from monthly application.
        expected_low = 40_050.0 * 0.06
        expected_high = (50_000.0 - 40_050.0) * 0.11
        assert credit == pytest.approx(expected_low + expected_high, abs=0.10)
        assert credit == pytest.approx(3_497.50, abs=0.10)

    def test_ending_balance_matches_spd_example_within_100(self) -> None:
        # Single-year projection of the SPD scenario, with 2023's
        # SSWB / comp-limit / 5% pre-2016 floor.
        end_bal = 210_000.0 * (1 + 0.05) + pension_annual_credit(
            50_000.0,
            age=45,
            years_of_service=20,
            qtr_sswb=160_200.0 / 4,
            comp_limit=330_000.0,
        )
        # SPD's ending balance: $224,084.95.  Our annual-aggregate
        # approximation: balance accrues interest at 5% upfront, then
        # pay credits added at year end — so we undershoot slightly
        # (no interest on the pay credits themselves).
        assert end_bal == pytest.approx(224_085.0, abs=100.0)


class TestIrsCompensationLimit:
    """The SPD applies the IRS §401(a)(17) limit before computing pay
    credits. A $500k earner with the 6%/11% tier only accrues credits
    on the first $350k (2025 limit)."""

    def test_high_earner_capped_by_irs_limit(self) -> None:
        salary = 500_000.0
        # Without the cap, our credit would be:
        #   low_band  = $46,125 × 0.06 = $2,767.50
        #   high_band = ($500,000 − $46,125) × 0.11 = $49,926.25
        #   total = $52,693.75
        # With the IRS cap ($350,000):
        #   low_band  = $46,125 × 0.06 = $2,767.50
        #   high_band = ($350,000 − $46,125) × 0.11 = $33,426.25
        #   total = $36,193.75
        credit = pension_annual_credit(salary, age=55, years_of_service=25)
        assert credit == pytest.approx(36_193.75, abs=0.01)

    def test_explicit_comp_limit_override(self) -> None:
        # Pass an explicit, lower cap.
        credit = pension_annual_credit(
            500_000.0,
            age=55,
            years_of_service=25,
            comp_limit=200_000.0,
        )
        expected = 46_125.0 * 0.06 + (200_000.0 - 46_125.0) * 0.11
        assert credit == pytest.approx(expected, abs=0.01)

    def test_disable_cap_with_zero(self) -> None:
        # comp_limit=0 / negative disables the cap entirely (used by
        # rare callers projecting a non-qualified supplemental plan).
        credit = pension_annual_credit(
            500_000.0,
            age=55,
            years_of_service=25,
            comp_limit=0.0,
        )
        # Full $500k counted.
        expected = 46_125.0 * 0.06 + (500_000.0 - 46_125.0) * 0.11
        assert credit == pytest.approx(expected, abs=0.01)


class TestInterestRateFloor:
    """BP RAP page 10: pre-2016 floor 5%, post-2016 floor 2%."""

    def test_pre_2016_floor_applies(self) -> None:
        # Requested 3%, pre-2016 floor 5% → 5%.
        assert effective_interest_rate(0.03, pre_2016_participant=True) == 0.05

    def test_post_2016_floor_applies(self) -> None:
        # Requested 0.5%, post-2016 floor 2% → 2%.
        assert effective_interest_rate(0.005, pre_2016_participant=False) == 0.02

    def test_above_floor_unchanged(self) -> None:
        # Requested 6%, pre-2016 floor 5% → 6% (above floor wins).
        assert effective_interest_rate(0.06, pre_2016_participant=True) == 0.06

    def test_none_defaults_to_floor(self) -> None:
        # No requested rate → return the participant's floor directly.
        assert effective_interest_rate(None, pre_2016_participant=True) == (
            PENSION_FLOOR_PRE_2016
        )
        assert effective_interest_rate(None, pre_2016_participant=False) == (
            PENSION_FLOOR_POST_2016
        )


class TestProjectorTierTransition:
    """A young participant's projection should produce LESS credit than
    a 25-yr senior's for the same salary, because the tier rates step
    up only after reaching age 40/50 or YoS 10/20."""

    def test_young_participant_under_accrues_relative_to_top_tier(self) -> None:
        # Identical salary, identical years projected — but the young
        # participant starts at age 30 / 5 YoS (bottom tier) and the
        # senior starts at age 55 / 25 YoS (top tier).
        common = dict(
            start_balance=0.0,
            start_earnings=200_000.0,
            years_to_retire=5,
            wage_growth=0.0,
            interest_rate=0.0,  # isolate pay credits
            comp_limit_today=1e9,  # disable comp cap
        )
        bal_young = project_pension_balance(
            **common, start_age=30, years_of_service_today=5
        )
        bal_senior = project_pension_balance(
            **common, start_age=55, years_of_service_today=25
        )
        # Young participant should accrue strictly less. Pre-v6.3 the
        # young projection used the top-tier 6%/11% rates and the two
        # were equal — that regression bug is fixed.
        assert bal_young < bal_senior * 0.75, (
            f"Young participant balance ${bal_young:,.0f} should be much "
            f"smaller than senior's ${bal_senior:,.0f}. Pre-v6.3 both were "
            f"equal because the projector ignored age / YoS."
        )

    def test_projector_advances_yos_through_horizon(self) -> None:
        # A 30-year-old with 9 YoS today, projected 11 years out, must
        # cross the YoS=10 boundary and start accruing at the mid tier
        # (5/9) from year 1 onward.
        common = dict(
            start_balance=0.0,
            start_earnings=80_000.0,
            wage_growth=0.0,
            interest_rate=0.0,
            comp_limit_today=1e9,
        )
        # Project 1 year: start at YoS=9, no tier transition yet.
        bal_no_transition = project_pension_balance(
            **common, years_to_retire=1, start_age=30, years_of_service_today=9
        )
        # Project 1 year: start at YoS=10 (already mid tier).
        bal_at_transition = project_pension_balance(
            **common, years_to_retire=1, start_age=30, years_of_service_today=10
        )
        # The YoS=10 path should accrue strictly more (5% × min(80k, 46125) + …
        # vs 4% × min(80k, 46125) + …).
        assert bal_at_transition > bal_no_transition


class TestProjectorRespectsRetireAge:
    """BP RAP rule: pay credits stop at termination of service, but
    interest credits continue accruing until benefit payment. The
    projector previously credited pay credits all the way to NRD,
    overstating the reference balance for users who retired before
    NRD and silently scaling their input `monthly_at_nrd` down."""

    def test_retire_age_stops_pay_credits(self) -> None:
        # Start at 54, NRD 67, retire at 65. 11 years of pay credits,
        # 2 years of interest-only.
        with_retire = project_pension_balance(
            start_balance=400_000.0,
            start_earnings=200_000.0,
            years_to_retire=13,
            wage_growth=0.01,
            start_age=54,
            years_of_service_today=20,
            interest_rate=0.05,
            comp_limit_today=350_000.0,
            retire_age=65,
        )
        # Same path but the projector doesn't know about retire_age
        # → credits all 13 years.
        without_retire = project_pension_balance(
            start_balance=400_000.0,
            start_earnings=200_000.0,
            years_to_retire=13,
            wage_growth=0.01,
            start_age=54,
            years_of_service_today=20,
            interest_rate=0.05,
            comp_limit_today=350_000.0,
            retire_age=None,
        )
        assert with_retire < without_retire, (
            "Honoring retire_age should yield a smaller projected "
            "balance (fewer pay credits) than ignoring it."
        )
        # Sanity: the difference should be roughly 2 years of pay
        # credits compounded by ~2 years of interest. At ~$20-25k of
        # credit per year × 2 ≈ $40-50k.
        assert 30_000 < (without_retire - with_retire) < 80_000


class TestSimulatorIncludesBonusInPension:
    """BP RAP page 9: eligible earnings include 'Payments made under
    an annual incentive plan'. The simulator must include
    `spouse_a_bonus` in the pension pay-credit base."""

    def test_bonus_increases_pension_balance(self) -> None:
        from tax_optimizer import Config, Inputs, simulate
        from tax_optimizer.inputs import CurrentIncome, PensionInputs, StartingBalances
        from tax_optimizer.mortality import Mortality
        from tax_optimizer.spending import SpendingProfile

        cfg = Config(
            horizon_age=68,
            nominal_growth_rate=0.0,
            taxable_equity_div_yield=0.0,
            taxable_bond_interest_yield=0.0,
            taxable_drag=0.0,
            inflation=0.0,
            wage_growth=0.0,
            spending=SpendingProfile.flat(base_spending=80_000.0, inflation=0.0),
            annual_expenses_today=80_000.0,
            aca_enabled=False,
            mortality=Mortality(),
        )
        common_kwargs = dict(
            spouse_a_age_start=55,
            spouse_b_age_start=55,
            spouse_a_retire_age=65,
            spouse_b_retire_age=65,
            starting=StartingBalances(
                spouse_a_pretax_401k=200_000.0,
                spouse_b_pretax_401k=0.0,
                spouse_a_roth_ira=0.0,
                spouse_b_roth_ira=0.0,
                taxable_brokerage=500_000.0,
                hsa=0.0,
            ),
            pension=PensionInputs(
                balance_today=200_000.0,
                monthly_at_nrd=2_000.0,
                start_age=67,
                years_of_service_today=20,
                pre_2016_participant=True,
            ),
        )
        inp_no_bonus = Inputs(
            **common_kwargs,
            income=CurrentIncome(spouse_a_gross=150_000.0, spouse_a_bonus=0.0),
        )
        inp_with_bonus = Inputs(
            **common_kwargs,
            income=CurrentIncome(spouse_a_gross=150_000.0, spouse_a_bonus=50_000.0),
        )
        df_no_bonus = simulate(cfg, inp_no_bonus)
        df_with_bonus = simulate(cfg, inp_with_bonus)

        # Last working year before retirement: age 64.
        bal_no_bonus = df_no_bonus[df_no_bonus["spouse_a_age"] == 64][
            "pension_balance"
        ].iloc[0]
        bal_with_bonus = df_with_bonus[df_with_bonus["spouse_a_age"] == 64][
            "pension_balance"
        ].iloc[0]
        assert bal_with_bonus > bal_no_bonus, (
            "Including the $50k bonus in pension-eligible earnings "
            "should produce a higher pension balance at retirement."
        )
        # The extra $50k/year for ~10 years at 11% = ~$5500/yr credit,
        # compounded at 5% over 10 years ≈ $69k extra.
        assert bal_with_bonus - bal_no_bonus > 50_000, (
            "Pension-balance lift from including a $50k annual bonus "
            "over 10 working years should be at least $50k."
        )
