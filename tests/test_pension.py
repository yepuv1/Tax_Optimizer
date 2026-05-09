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
    def test_zero_earnings_zero_credit(self) -> None:
        assert pension_annual_credit(0.0) == 0.0

    def test_below_kink_uses_only_low_rate(self) -> None:
        # Monthly earnings below the SSWB-quarter kink ($46_125/mo).
        # Annual credit = 12 * (monthly * low_rate).
        annual_e = 60_000.0
        out = pension_annual_credit(annual_e)
        # monthly = 5_000, low_credit = 5_000 * 0.06 = 300/mo, annual = 3_600.
        assert out == pytest.approx(3_600.0)

    def test_at_kink_no_high_credit(self) -> None:
        # Earnings exactly at the kink → high_credit term is zero.
        annual_e = PENSION_QTR_SSWB * 12
        out = pension_annual_credit(annual_e)
        # All credit comes from the low rate at the kink amount.
        expected = PENSION_QTR_SSWB * PENSION_LOW_RATE * 12
        assert out == pytest.approx(expected)

    def test_above_kink_blends_low_and_high(self) -> None:
        # 2x the kink → half low, half high
        monthly_kink = PENSION_QTR_SSWB
        annual_e = monthly_kink * 2 * 12  # double the kink
        low = monthly_kink * PENSION_LOW_RATE
        high = monthly_kink * PENSION_HIGH_RATE
        expected = (low + high) * 12
        assert pension_annual_credit(annual_e) == pytest.approx(expected)


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
