"""Tests for `tax_optimizer.spending` (SpendingProfile + helpers)."""

from __future__ import annotations

import pytest

from tax_optimizer.spending import (
    LongTermCareShock,
    LumpEvent,
    SpendingPhase,
    SpendingProfile,
)


class TestFlatProfile:
    def test_flat_returns_inflated_amount_each_year(self) -> None:
        profile = SpendingProfile.flat(base_spending=100_000, inflation=0.025)
        # Year 0: no inflation.
        rec0, lumps0 = profile.amount_for(0, age_a=50, years_until_horizon=40)
        assert rec0 == pytest.approx(100_000.0)
        assert lumps0 == []
        # Year 10: inflated by 2.5%/yr.
        rec10, _ = profile.amount_for(10, age_a=60, years_until_horizon=30)
        assert rec10 == pytest.approx(100_000.0 * 1.025**10)

    def test_flat_no_ltc_shock_at_horizon(self) -> None:
        profile = SpendingProfile.flat(base_spending=80_000)
        rec_last, _ = profile.amount_for(year_offset=39, age_a=89, years_until_horizon=0)
        # No LTC shock in flat → just base inflated.
        assert rec_last == pytest.approx(80_000.0 * 1.025**39)


class TestSmileProfile:
    def test_phases_apply_their_multipliers(self) -> None:
        profile = SpendingProfile.retirement_smile(
            base_spending=100_000.0, inflation=0.0, ltc_years=0, ltc_annual_today=0
        )
        # Working: age 50 → mult 1.00
        rec, _ = profile.amount_for(year_offset=0, age_a=50, years_until_horizon=40)
        assert rec == pytest.approx(100_000.0)
        # Go-go: age 65 → 1.15
        rec, _ = profile.amount_for(year_offset=15, age_a=65, years_until_horizon=25)
        assert rec == pytest.approx(115_000.0)
        # Slow-go: age 75 → 0.95
        rec, _ = profile.amount_for(year_offset=25, age_a=75, years_until_horizon=15)
        assert rec == pytest.approx(95_000.0)
        # No-go: age 85 → 1.00
        rec, _ = profile.amount_for(year_offset=35, age_a=85, years_until_horizon=5)
        assert rec == pytest.approx(100_000.0)

    def test_ltc_shock_only_in_last_n_years(self) -> None:
        # ltc_years=3 → shock fires when years_until_horizon < 3 (i.e. last 3 yrs).
        profile = SpendingProfile.retirement_smile(
            base_spending=100_000.0, inflation=0.0,
            ltc_years=3, ltc_annual_today=120_000.0,
        )
        # Outside the LTC window: base * mult only.
        rec, _ = profile.amount_for(year_offset=20, age_a=70, years_until_horizon=20)
        assert rec == pytest.approx(115_000.0)  # go-go phase, no shock
        # Last year of life: shock fires; mult=1.0 (no-go), 0% inflation.
        rec, _ = profile.amount_for(year_offset=39, age_a=89, years_until_horizon=0)
        assert rec == pytest.approx(100_000.0 + 120_000.0)


class TestLumpEvents:
    def test_lump_event_returned_only_in_target_year(self) -> None:
        events = [
            LumpEvent(year_offset=5, amount_today=20_000, label="new car"),
            LumpEvent(year_offset=15, amount_today=50_000, label="kid wedding"),
        ]
        profile = SpendingProfile(
            base_spending=100_000, inflation=0.0,
            phases=[SpendingPhase(0, 200, 1.0, "flat")],
            lump_events=events,
        )
        # Year 5: only the car event.
        _, lumps = profile.amount_for(5, age_a=55, years_until_horizon=35)
        assert len(lumps) == 1
        assert lumps[0].label == "new car"
        # Year 10: nothing.
        _, lumps = profile.amount_for(10, age_a=60, years_until_horizon=30)
        assert lumps == []
        # Year 15: only the wedding.
        _, lumps = profile.amount_for(15, age_a=65, years_until_horizon=25)
        assert len(lumps) == 1
        assert lumps[0].label == "kid wedding"


class TestPhaseFallback:
    def test_age_outside_any_phase_returns_1pt0(self) -> None:
        # No phase covers age 200; helper falls back to 1.0
        profile = SpendingProfile(
            base_spending=100_000.0, inflation=0.0,
            phases=[SpendingPhase(0, 50, 0.5, "kids")],
        )
        rec, _ = profile.amount_for(year_offset=0, age_a=99, years_until_horizon=10)
        assert rec == pytest.approx(100_000.0)


class TestLongTermCareShockDataclass:
    def test_ltc_shock_defaults(self) -> None:
        s = LongTermCareShock()
        assert s.years == 3
        assert s.annual_cost_today == 80_000.0
