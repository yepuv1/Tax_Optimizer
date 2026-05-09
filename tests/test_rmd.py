"""Tests for `tax_optimizer.rmd.rmd_amount`."""

from __future__ import annotations

import pytest

from tax_optimizer.rmd import UNIFORM_LIFETIME, rmd_amount


class TestRmd:
    def test_zero_balance_returns_zero(self) -> None:
        assert rmd_amount(balance=0.0, age=80) == 0.0

    def test_negative_balance_returns_zero(self) -> None:
        # Defensive: simulator sometimes drives a balance to slightly
        # negative on the way to a max(0,...) clamp. RMD on the
        # in-between snapshot must not be a credit.
        assert rmd_amount(balance=-5_000.0, age=80) == 0.0

    def test_pre_start_age_returns_zero(self) -> None:
        assert rmd_amount(balance=500_000.0, age=70) == 0.0
        assert rmd_amount(balance=500_000.0, age=74, rmd_start_age=75) == 0.0

    def test_at_start_age_uses_table_divisor(self) -> None:
        # At age 75 the IRS divisor is 24.6 → RMD = balance / 24.6
        out = rmd_amount(balance=1_000_000.0, age=75)
        assert out == pytest.approx(1_000_000.0 / 24.6)

    def test_arbitrary_table_age(self) -> None:
        out = rmd_amount(balance=500_000.0, age=85)
        assert out == pytest.approx(500_000.0 / 16.0)

    def test_age_beyond_table_uses_oldest_divisor(self) -> None:
        # Table tops out at 110 (divisor 3.5). Age 120 should fall back
        # to the same divisor (the "max key" branch in the implementation).
        oldest_div = UNIFORM_LIFETIME[max(UNIFORM_LIFETIME)]
        out = rmd_amount(balance=100_000.0, age=120)
        assert out == pytest.approx(100_000.0 / oldest_div)

    def test_rmd_grows_as_divisor_shrinks(self) -> None:
        # Same balance, increasing age → strictly larger RMD.
        balance = 750_000.0
        rmds = [rmd_amount(balance=balance, age=age) for age in range(75, 100)]
        for prior, later in zip(rmds, rmds[1:]):
            assert later > prior

    def test_rmd_scales_linearly_with_balance(self) -> None:
        a = rmd_amount(balance=100_000.0, age=80)
        b = rmd_amount(balance=300_000.0, age=80)
        assert b == pytest.approx(3 * a)

    def test_custom_start_age(self) -> None:
        # Bumping rmd_start_age to 73 (SECURE 1.0) makes 73 active.
        assert rmd_amount(balance=1_000_000.0, age=73, rmd_start_age=73) > 0
        assert rmd_amount(balance=1_000_000.0, age=72, rmd_start_age=73) == 0.0
