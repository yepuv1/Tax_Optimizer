"""Tests for `tax_optimizer.limits`."""

from __future__ import annotations

import pytest

from tax_optimizer.limits import (
    ELECTIVE_DEFERRAL_CATCH_UP_50,
    ELECTIVE_DEFERRAL_LIMIT,
    HSA_CATCH_UP_55,
    HSA_FAMILY_LIMIT,
    HSA_INELIGIBLE_AGE,
    elective_deferral_cap,
    hsa_family_cap,
)


class TestElectiveDeferralCap:
    def test_under_50_no_catch_up(self) -> None:
        assert elective_deferral_cap(40) == ELECTIVE_DEFERRAL_LIMIT

    def test_at_50_includes_catch_up(self) -> None:
        assert elective_deferral_cap(50) == (
            ELECTIVE_DEFERRAL_LIMIT + ELECTIVE_DEFERRAL_CATCH_UP_50
        )

    def test_over_50_includes_catch_up(self) -> None:
        assert elective_deferral_cap(67) == (
            ELECTIVE_DEFERRAL_LIMIT + ELECTIVE_DEFERRAL_CATCH_UP_50
        )


class TestHsaFamilyCap:
    def test_zero_when_neither_working(self) -> None:
        assert hsa_family_cap(50, 50, either_working=False) == 0.0

    def test_under_55_no_catch_up(self) -> None:
        assert hsa_family_cap(50, 48, either_working=True) == HSA_FAMILY_LIMIT

    def test_one_over_55_adds_catch_up(self) -> None:
        # Catch-up triggers if either spouse is 55+.
        cap = hsa_family_cap(56, 49, either_working=True)
        assert cap == HSA_FAMILY_LIMIT + HSA_CATCH_UP_55

    def test_both_at_medicare_age_zero(self) -> None:
        # Once both hit 65, no HDHP, no contributions.
        assert (
            hsa_family_cap(HSA_INELIGIBLE_AGE, HSA_INELIGIBLE_AGE, either_working=True)
            == 0.0
        )

    def test_one_under_medicare_age_still_eligible(self) -> None:
        # If one spouse is Medicare-eligible but the other isn't, the
        # younger one can still cover the family on an HDHP.
        cap = hsa_family_cap(67, 60, either_working=True)
        assert cap > 0.0
