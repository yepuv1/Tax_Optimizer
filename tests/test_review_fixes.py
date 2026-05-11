"""Regression tests for the six HIGH-severity findings identified during
the package review (see CHANGELOG entry "v6.1 — review fixes").

Each block corresponds to one finding:

  1. `scenarios/example01.json` rename + hardened smoke skip
  2. Simulator gates `inputs.income.*` on (a_working OR b_working)
  3. CLI threads `--seed` into the optimizer's `mc_seed=`
  4. `Config` rejects `rmd_start_age < 72`
  5. `_market_summary` / cross-model table route through `resolved_market()`
  6. Post-cascade IRMAA respects `irmaa_lookback_years`
"""
from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import pytest

from tax_optimizer import (
    Config,
    Inputs,
    LognormalModel,
    Mortality,
    SocialSecurity,
    SpendingProfile,
    StartingBalances,
    simulate,
)
from tax_optimizer.inputs import CurrentIncome
from tax_optimizer.rmd import UNIFORM_LIFETIME, rmd_amount


# ---------------------------------------------------------------------------
# Fix 1 — example01.json rename
# ---------------------------------------------------------------------------


class TestExampleScenarioRename:
    def test_example01_json_exists(self) -> None:
        assert Path("scenarios/example01.json").exists(), (
            "scenarios/example01.json must be the on-disk filename so the "
            "README / CHANGELOG / scenario_guide references work."
        )

    def test_example_json_legacy_name_is_gone(self) -> None:
        assert not Path("scenarios/example.json").exists(), (
            "Legacy scenarios/example.json must not coexist with the renamed "
            "scenarios/example01.json — pick one."
        )


# ---------------------------------------------------------------------------
# Fix 2 — staggered retirement preserves inputs.income.*
# ---------------------------------------------------------------------------


def _staggered_inputs(*, income_block: CurrentIncome) -> tuple[Config, Inputs]:
    """A household where spouse A retires at 60 and spouse B keeps working
    to 67. We deliberately set portfolio yields and HSA/ACA to zero so the
    `inputs.income.*` contribution to AGI is isolated.
    """
    cfg = Config(
        horizon_age=70,
        nominal_growth_rate=0.0,
        taxable_equity_div_yield=0.0,
        taxable_bond_interest_yield=0.0,
        taxable_drag=0.0,
        inflation=0.0,
        spending=SpendingProfile.flat(base_spending=0.0, inflation=0.0),
        annual_expenses_today=0.0,
        aca_enabled=False,
        mortality=Mortality(),
    )
    inputs = Inputs(
        spouse_a_age_start=58,
        spouse_b_age_start=58,
        spouse_a_retire_age=60,
        spouse_b_retire_age=67,
        starting=StartingBalances(
            spouse_a_pretax_401k=200_000.0,
            spouse_b_pretax_401k=200_000.0,
            spouse_a_roth_ira=0.0,
            spouse_b_roth_ira=0.0,
            taxable_brokerage=0.0,
            hsa=0.0,
        ),
        income=income_block,
        ss=SocialSecurity(monthly_spouse_a=0.0, monthly_spouse_b=0.0, start_age=70),
    )
    return cfg, inputs


class TestSimulatorIncomeGating:
    def test_inputs_income_preserved_when_only_spouse_b_works(self) -> None:
        """Year offset 3 = age 61 for A (retired) but 61 for B (still
        working through age 66). With the previous `a_working`-only gate
        the interest/cap-gains/dividends silently zeroed; the fix should
        keep them flowing while at least one spouse earns wages."""
        income_with = CurrentIncome(
            spouse_a_gross=0.0,           # A no longer earns wages.
            spouse_b_gross=120_000.0,     # B still working.
            spouse_a_bonus=0.0,
            interest=5_000.0,
            capital_gains=3_000.0,
            dividends=2_000.0,
        )
        income_without = CurrentIncome(
            spouse_a_gross=0.0,
            spouse_b_gross=120_000.0,
            spouse_a_bonus=0.0,
            interest=0.0,
            capital_gains=0.0,
            dividends=0.0,
        )
        cfg, inputs_with = _staggered_inputs(income_block=income_with)
        _, inputs_without = _staggered_inputs(income_block=income_without)

        df_with = simulate(cfg, inputs_with)
        df_without = simulate(cfg, inputs_without)

        # Pick a year where A has retired but B is still working. With
        # spouse_a_retire_age=60 and start age 58, year offset 3 = age 61.
        a_retired_b_working = df_with[
            (df_with["spouse_a_age"] > 60) & (df_with["spouse_b_age"] < 67)
        ]
        assert len(a_retired_b_working) > 0
        agi_delta = (
            a_retired_b_working["agi"].iloc[0]
            - df_without[df_without["spouse_a_age"] == a_retired_b_working["spouse_a_age"].iloc[0]][
                "agi"
            ].iloc[0]
        )
        # 5k interest + 3k cap gains + 2k dividends = 10k of extra AGI
        # while spouse B is still working (gated previously to 0 by the
        # bug).
        assert agi_delta == pytest.approx(10_000.0, rel=0.01)


# ---------------------------------------------------------------------------
# Fix 3 — CLI mc_seed thread-through
# ---------------------------------------------------------------------------


class TestObjectiveOptimizerSeedThreading:
    def test_run_objective_optimizer_passes_mc_seed(self) -> None:
        """The CLI's `_run_objective_optimizer` must thread `args.seed`
        into BOTH `seed=` and `mc_seed=` so MC objective curves react to
        the user's seed knob.
        """
        from tax_optimizer.__main__ import _run_objective_optimizer

        src = inspect.getsource(_run_objective_optimizer)
        assert "mc_seed=args.seed" in src, (
            "_run_objective_optimizer must pass `mc_seed=args.seed` to "
            "optimize_household; otherwise the CLI's --seed flag has no "
            "effect on the Monte Carlo draws used by cvar / p_success "
            "objectives."
        )


# ---------------------------------------------------------------------------
# Fix 4 — rmd_start_age guard
# ---------------------------------------------------------------------------


class TestRmdStartAgeGuard:
    def test_config_rejects_rmd_start_age_below_72(self) -> None:
        with pytest.raises(ValueError, match="rmd_start_age must be >= 72"):
            Config(rmd_start_age=71)

    def test_config_accepts_72_and_above(self) -> None:
        # Should not raise.
        Config(rmd_start_age=72)
        Config(rmd_start_age=73)
        Config(rmd_start_age=75)

    def test_rmd_amount_returns_zero_for_under_72(self) -> None:
        """Defensive: even if someone bypasses Config and calls
        rmd_amount directly with an under-72 age (but age >=
        rmd_start_age), we must return 0.0 — not balance / 1.0.
        """
        assert rmd_amount(balance=500_000, age=71, rmd_start_age=71) == 0.0
        assert rmd_amount(balance=500_000, age=70, rmd_start_age=70) == 0.0

    def test_rmd_amount_caps_at_top_of_table(self) -> None:
        """Beyond the published table (age >= 110) pins to the smallest
        divisor (largest withdrawal fraction) rather than running off
        the end."""
        balance = 100_000.0
        top_key = max(UNIFORM_LIFETIME)
        top_divisor = UNIFORM_LIFETIME[top_key]
        expected = balance / top_divisor
        assert rmd_amount(balance, age=120, rmd_start_age=75) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Fix 5 — _market_summary uses resolved_market()
# ---------------------------------------------------------------------------


class TestMarketSummaryResolvesNone:
    def test_default_config_renders_deterministic_not_nonetype(self) -> None:
        from tax_optimizer.report import _market_summary

        cfg = Config()
        assert cfg.market is None
        summary = _market_summary(cfg)
        assert "NoneType" not in summary, (
            "_market_summary must route through cfg.resolved_market() so "
            "the default Config() renders DeterministicModel rather than "
            "the raw NoneType string."
        )
        assert "Deterministic" in summary

    def test_explicit_lognormal_renders_parameters(self) -> None:
        from tax_optimizer.report import _market_summary

        cfg = Config(market=LognormalModel(equity_mu=0.07, equity_sigma=0.18))
        summary = _market_summary(cfg)
        assert "LognormalModel" in summary
        assert "equity μ=7.0%" in summary


# ---------------------------------------------------------------------------
# Fix 6 — post-cascade IRMAA respects irmaa_lookback_years
# ---------------------------------------------------------------------------


def _cascade_triggering_inputs() -> tuple[Config, Inputs]:
    """Both spouses on Medicare with spending high enough to force a
    deficit cascade in retirement years. Pretax balance is sized to
    push current AGI into a high IRMAA tier when cascade draws fire."""
    cfg = Config(
        horizon_age=72,
        nominal_growth_rate=0.0,
        taxable_equity_div_yield=0.0,
        taxable_bond_interest_yield=0.0,
        taxable_drag=0.0,
        inflation=0.0,
        spending=SpendingProfile.flat(base_spending=260_000.0, inflation=0.0),
        annual_expenses_today=260_000.0,
        aca_enabled=False,
        irmaa_lookback_years=2,
        mortality=Mortality(),
    )
    inputs = Inputs(
        spouse_a_age_start=66,
        spouse_b_age_start=66,
        spouse_a_retire_age=66,
        spouse_b_retire_age=66,
        starting=StartingBalances(
            spouse_a_pretax_401k=2_000_000.0,
            spouse_b_pretax_401k=2_000_000.0,
            spouse_a_roth_ira=0.0,
            spouse_b_roth_ira=0.0,
            taxable_brokerage=10_000.0,  # tiny taxable forces the cascade.
            hsa=0.0,
        ),
        income=CurrentIncome(),
        ss=SocialSecurity(
            monthly_spouse_a=1_500.0, monthly_spouse_b=1_500.0, start_age=67
        ),
    )
    return cfg, inputs


class TestPostCascadeIrmaaLookback:
    def test_first_retirement_year_irmaa_zero_under_two_year_lookback(self) -> None:
        """Year 0 (start of retirement) IRMAA must be 0 even if the
        cascade fires, because IRMAA in year T is based on AGI from
        year T-2 (which is pre-retirement and below thresholds).
        Previously the post-cascade recompute clobbered that 0 with the
        current-year, cascade-inflated AGI."""
        cfg, inputs = _cascade_triggering_inputs()
        df = simulate(cfg, inputs)
        first_year = df.iloc[0]
        # The cascade only fires if pretax_withdrawal or roth conversion
        # bumps current AGI above the IRMAA threshold. Confirm the
        # scenario actually exercises the cascade.
        assert first_year["agi"] > 220_000, (
            "Test scenario is supposed to push AGI above the first IRMAA "
            "MFJ threshold ($212k in the regime defaults) via cascade draws; "
            "if AGI stays low the test is meaningless."
        )
        assert first_year["irmaa"] == 0.0, (
            "Year-0 IRMAA under irmaa_lookback_years=2 must be 0 (T-2 AGI "
            "is pre-retirement). The previous post-cascade recompute "
            "incorrectly used current-year AGI."
        )

    def test_zero_lookback_still_recomputes_after_cascade(self) -> None:
        """When the user opts out of the lookback (sets
        `irmaa_lookback_years=0`), the post-cascade IRMAA should still
        be recomputed on the cascade-inflated AGI — that's the only
        knob whose semantics asks for it."""
        cfg, inputs = _cascade_triggering_inputs()
        cfg_zero = replace(cfg, irmaa_lookback_years=0)
        df = simulate(cfg_zero, inputs)
        first_year = df.iloc[0]
        assert first_year["agi"] > 220_000
        # With current-year AGI, this household is firmly into an IRMAA
        # tier — surcharge should be strictly positive.
        assert first_year["irmaa"] > 0.0
