"""Tests for `tax_optimizer.tax.irmaa.irmaa_annual_surcharge`."""

from __future__ import annotations

import pytest

from tax_optimizer.tax.irmaa import (
    MEDICARE_ELIGIBLE_AGE,
    irmaa_annual_surcharge,
)
from tax_optimizer.tax.regimes import TCJA_EXTENDED


# Re-derive the tier 1 + 2 tier-defining MFJ thresholds & surcharges from
# the regime so the tests stay in sync if CMS publishes new numbers.
_MFJ_TIERS = TCJA_EXTENDED.irmaa_tiers_mfj
_SINGLE_TIERS = TCJA_EXTENDED.irmaa_tiers_single


class TestIRMAA:
    def test_zero_enrolled_returns_zero(self) -> None:
        out = irmaa_annual_surcharge(magi=500_000, n_enrolled=0, regime=TCJA_EXTENDED)
        assert out == {"partB": 0.0, "partD": 0.0, "total": 0.0, "tier": 0}

    def test_below_first_threshold_is_tier_zero(self) -> None:
        # MFJ tier 0 caps at $212k.
        out = irmaa_annual_surcharge(magi=200_000, n_enrolled=2, regime=TCJA_EXTENDED)
        assert out["tier"] == 0
        assert out["total"] == 0.0

    def test_inside_tier_1_for_one_enrollee(self) -> None:
        # MFJ tier 1: cap=266k, partB=74, partD=13.70
        magi = 250_000  # between 212k and 266k
        out = irmaa_annual_surcharge(magi=magi, n_enrolled=1, regime=TCJA_EXTENDED)
        assert out["tier"] == 1
        assert out["partB"] == pytest.approx(74.0 * 12 * 1)
        assert out["partD"] == pytest.approx(13.70 * 12 * 1)
        assert out["total"] == pytest.approx((74.0 + 13.70) * 12 * 1)

    def test_n_enrolled_scales_total_linearly(self) -> None:
        magi = 250_000  # tier 1
        one = irmaa_annual_surcharge(magi=magi, n_enrolled=1, regime=TCJA_EXTENDED)
        two = irmaa_annual_surcharge(magi=magi, n_enrolled=2, regime=TCJA_EXTENDED)
        assert two["total"] == pytest.approx(one["total"] * 2)
        assert two["partB"] == pytest.approx(one["partB"] * 2)
        assert two["partD"] == pytest.approx(one["partD"] * 2)

    def test_tier_escalation(self) -> None:
        # Walk MFJ MAGI up through every tier and check that the returned
        # tier index matches the published table order.
        midpoints = [200_000, 250_000, 300_000, 370_000, 500_000, 1_000_000]
        expected_tiers = [0, 1, 2, 3, 4, 5]
        for magi, tier in zip(midpoints, expected_tiers):
            out = irmaa_annual_surcharge(magi=magi, n_enrolled=2, regime=TCJA_EXTENDED)
            assert out["tier"] == tier, (magi, out)

    def test_top_tier_uses_max_partB_partD(self) -> None:
        # Anything above the last finite cap (750k MFJ) hits the inf-cap row
        # at partB=443.90, partD=85.80.
        out = irmaa_annual_surcharge(magi=10_000_000, n_enrolled=2, regime=TCJA_EXTENDED)
        assert out["tier"] == len(_MFJ_TIERS) - 1
        assert out["partB"] == pytest.approx(443.90 * 12 * 2)
        assert out["partD"] == pytest.approx(85.80 * 12 * 2)

    def test_single_thresholds_lower_than_mfj(self) -> None:
        # MAGI=$130k → tier 1 single (cap 133k), tier 0 MFJ (cap 212k).
        single = irmaa_annual_surcharge(
            magi=130_000, n_enrolled=1, regime=TCJA_EXTENDED, filing_status="single"
        )
        mfj = irmaa_annual_surcharge(
            magi=130_000, n_enrolled=1, regime=TCJA_EXTENDED, filing_status="mfj"
        )
        assert single["tier"] == 1
        assert mfj["tier"] == 0
        assert single["total"] > 0
        assert mfj["total"] == 0

    def test_medicare_eligible_age_constant(self) -> None:
        # Sanity: simulator hard-codes the Medicare eligibility age.
        assert MEDICARE_ELIGIBLE_AGE == 65
