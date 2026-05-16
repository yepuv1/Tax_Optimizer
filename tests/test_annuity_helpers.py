"""Tests for ``tax_optimizer.annuity.exclusion_ratio``.

The exclusion ratio is the percentage of each non-qualified annuity
payment that is treated as a tax-free return of cost basis under
IRC §72(b). The simulator multiplies it by the annual payment to
get the tax-free portion of each year's distribution.

We test:
  * Known closed-form examples.
  * Edge cases (zero basis, zero payment, zero years, over-funded).
  * Clamping into [0, 1].
"""

from __future__ import annotations

import math

import pytest

from tax_optimizer.annuity import exclusion_ratio


class TestExclusionRatioKnownCases:
    def test_classic_example(self):
        # $50k basis / ($12k payment * 20 yrs) = 0.20833...
        ratio = exclusion_ratio(
            cost_basis=50_000,
            annual_payment=12_000,
            expected_payout_years=20,
        )
        assert math.isclose(ratio, 50_000 / (12_000 * 20), rel_tol=1e-9)

    def test_half_basis_recovery(self):
        # Basis exactly half of expected total payments → 50% exclusion.
        ratio = exclusion_ratio(
            cost_basis=120_000,
            annual_payment=12_000,
            expected_payout_years=20,
        )
        assert math.isclose(ratio, 0.5, rel_tol=1e-9)

    def test_full_basis_recovery(self):
        # Basis exactly equals expected total payments → 100% exclusion.
        ratio = exclusion_ratio(
            cost_basis=240_000,
            annual_payment=12_000,
            expected_payout_years=20,
        )
        assert math.isclose(ratio, 1.0, rel_tol=1e-9)


class TestExclusionRatioEdgeCases:
    def test_zero_basis(self):
        # No basis → every dollar is taxable (qualified-like behavior).
        assert exclusion_ratio(0, 12_000, 20) == 0.0

    def test_negative_basis_treated_as_zero(self):
        assert exclusion_ratio(-100, 12_000, 20) == 0.0

    def test_zero_payment(self):
        # No payment → ratio undefined; helper returns 0.0 (no division).
        assert exclusion_ratio(50_000, 0, 20) == 0.0

    def test_negative_payment_treated_as_zero(self):
        assert exclusion_ratio(50_000, -100, 20) == 0.0

    def test_zero_payout_years_returns_one(self):
        # Validation upstream rejects zero, but the helper is defensive:
        # returns 1.0 (full basis recovery this year) instead of dividing
        # by zero. Simulator's basis-remaining decrement still caps the
        # cumulative tax-free amount at the actual basis.
        assert exclusion_ratio(50_000, 12_000, 0) == 1.0

    def test_over_funded_basis_clamps_to_one(self):
        # Basis > expected total payments would yield > 1.0 — clamped
        # to 1.0 so the simulator never multiplies a payment by > 1.
        assert exclusion_ratio(500_000, 12_000, 20) == 1.0


class TestExclusionRatioBoundedness:
    @pytest.mark.parametrize(
        "basis, payment, years",
        [
            (50_000, 12_000, 20),
            (1, 1, 1),
            (100_000_000, 1, 1_000_000),
            (1_000_000, 1_000_000, 100),
        ],
    )
    def test_always_in_zero_one(self, basis, payment, years):
        ratio = exclusion_ratio(basis, payment, years)
        assert 0.0 <= ratio <= 1.0
