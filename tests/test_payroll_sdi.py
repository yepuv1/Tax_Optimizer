"""Regression tests for the year-indexed CA SDI rate schedule.

CA EDD publishes annual SDI / VPDI rates that diverge from the
static 1.1% baseline:

  * 2024: 1.1% (post-SB-951 baseline; the SDI taxable wage cap was
    removed by SB 951 effective 2024)
  * 2025: 1.2% (EDD 2024 announcement)
  * 2026: 0.9% (EDD 2025 announcement)

Pre-fix the simulator hard-coded ``state_regime.sdi_rate=0.011`` for
every year, so a CA scenario with ``cfg.start_year=2025`` understated
SDI by ~9%, and a 2026 scenario overstated it by ~22%. Post-fix the
``StateTaxRegime.sdi_rate_schedule`` lookup by calendar year resolves
the published rate (and falls back to ``sdi_rate`` for years before
the schedule starts).
"""

from __future__ import annotations

import pytest

from tax_optimizer import Config, Inputs, simulate
from tax_optimizer.inputs import (
    CurrentContrib,
    CurrentIncome,
    StartingBalances,
)
from tax_optimizer.tax.state import CA, NY, STATELESS


@pytest.mark.parametrize(
    "calendar_year,expected_rate",
    [
        (2024, 0.011),
        (2025, 0.012),
        (2026, 0.009),
        (2027, 0.009),  # carries forward from most recent published
        (2030, 0.009),
        (2023, 0.011),  # pre-schedule → static fallback (1.1%)
    ],
)
def test_ca_sdi_rate_by_year(calendar_year: int, expected_rate: float) -> None:
    inp = Inputs(
        spouse_a_age_start=50,
        spouse_b_age_start=50,
        spouse_a_retire_age=65,
        spouse_b_retire_age=65,
        income=CurrentIncome(
            spouse_a_gross=100_000.0,
            spouse_b_gross=0.0,
            spouse_a_bonus=0.0,
            interest=0.0,
            capital_gains=0.0,
            dividends=0.0,
        ),
        contrib=CurrentContrib(hsa_family=0.0),
        starting=StartingBalances(taxable_brokerage=0.0, hsa=0.0),
    )
    cfg = Config(state_regime=CA, horizon_age=51, start_year=calendar_year)
    df = simulate(cfg, inp)
    actual_rate = df.iloc[0]["state_sdi"] / 100_000.0
    assert actual_rate == pytest.approx(expected_rate, abs=1e-5), (
        f"CA SDI rate at {calendar_year} expected {expected_rate}, got "
        f"{actual_rate:.4f}"
    )


def test_state_regime_method_directly() -> None:
    # Direct API: `effective_sdi_rate` resolves the schedule entry
    # without needing a full simulation.
    assert CA.effective_sdi_rate(2024) == 0.011
    assert CA.effective_sdi_rate(2025) == 0.012
    assert CA.effective_sdi_rate(2026) == 0.009
    # Carry-forward beyond the schedule.
    assert CA.effective_sdi_rate(2030) == 0.009
    # Pre-schedule fallback.
    assert CA.effective_sdi_rate(2020) == CA.sdi_rate


def test_other_regimes_unaffected() -> None:
    # NY / STATELESS don't ship SDI; year doesn't matter.
    assert NY.effective_sdi_rate(2025) == 0.0
    assert STATELESS.effective_sdi_rate(2026) == 0.0
