"""Tests for `tax_optimizer.pension`."""

from __future__ import annotations

import pytest

from tax_optimizer.pension import (
    PENSION_HIGH_RATE,
    PENSION_INTEREST,
    PENSION_LOW_RATE,
    PENSION_QTR_SSWB,
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
