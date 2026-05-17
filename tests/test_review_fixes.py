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


# ===========================================================================
# v6.2 — functional review fixes (F1/F2/F10 + B2 fixes)
# ===========================================================================


# ---------------------------------------------------------------------------
# F1 — pension_annual_credit annual-vs-monthly bug
# ---------------------------------------------------------------------------

class TestPensionCreditHighBandFires:
    """Pre-v6.2 the high-rate band almost never fired for normal salaries
    because the threshold (an *annual* dollar amount) was compared
    against monthly earnings. The fix evaluates both legs in annual
    dollars."""

    def test_high_band_credit_exceeds_low_band_only(self) -> None:
        from tax_optimizer.pension import (
            PENSION_HIGH_RATE,
            PENSION_LOW_RATE,
            PENSION_QTR_SSWB,
            pension_annual_credit,
        )

        # A salary 2x the kink should produce a *materially* larger
        # credit than the all-low-band approximation. With low=6% /
        # high=11%, 2× the kink gives 0.17× kink in credit vs 0.12×
        # kink for all-low — ~42% uplift.
        salary = 2 * PENSION_QTR_SSWB
        credit = pension_annual_credit(salary)
        all_low = salary * PENSION_LOW_RATE
        assert credit > all_low * 1.3, (
            f"Salary at 2× the kink produced credit ${credit:,.0f}; "
            f"all-low-band would be ${all_low:,.0f}. With the high band "
            f"working, the credit should be substantially larger."
        )
        # And the closed form should match.
        expected = (
            PENSION_QTR_SSWB * PENSION_LOW_RATE
            + PENSION_QTR_SSWB * PENSION_HIGH_RATE
        )
        assert credit == pytest.approx(expected)


# ---------------------------------------------------------------------------
# F2 — deficit cascade ignores state income tax
# ---------------------------------------------------------------------------

class TestCascadeIncorporatesStateTax:
    """The `_solve_pretax_for_net` / `_solve_taxable_for_net` helpers
    used to gross up only for federal tax. Households in CA/MA/OR
    came up 9-12% short on every cascade leg until v6.2."""

    def test_pretax_solver_grosses_up_for_state_tax(self) -> None:
        from tax_optimizer.tax.regimes import TCJA_EXTENDED
        from tax_optimizer.withdrawals import _solve_pretax_for_net

        regime = TCJA_EXTENDED
        base_kwargs: dict = dict(
            wages=0.0, interest=0.0, ordinary_div=0.0,
            qualified_div=0.0, ltcg=0.0, pension=0.0,
            pretax_withdrawal=0.0, roth_conversion=0.0,
            social_security=0.0,
        )

        # Federal-only gross.
        gross_fed_only = _solve_pretax_for_net(
            100_000.0, base_kwargs,
            regime=regime, filing_status="married_joint",
        )

        # Add a 10% flat state tax via the closure.
        def flat_state_tax(kw: dict, ss_taxable: float) -> float:
            return 0.10 * (
                kw.get("pretax_withdrawal", 0.0)
                + kw.get("wages", 0.0)
                + kw.get("interest", 0.0)
                + kw.get("ordinary_div", 0.0)
                + kw.get("ltcg", 0.0)
                + kw.get("roth_conversion", 0.0)
            )

        gross_with_state = _solve_pretax_for_net(
            100_000.0, base_kwargs,
            regime=regime, filing_status="married_joint",
            state_tax_fn=flat_state_tax,
        )

        # The state-tax-aware solver should pull a *larger* gross to
        # cover the additional 10% bite.
        assert gross_with_state > gross_fed_only + 5_000.0, (
            f"State-tax-aware gross (${gross_with_state:,.0f}) should "
            f"meaningfully exceed federal-only gross "
            f"(${gross_fed_only:,.0f}) when net target is $100k and "
            f"state rate is 10%."
        )

    def test_cascade_in_high_state_tax_state_pulls_more(self) -> None:
        """End-to-end: same scenario, CA vs NV. CA's cascade should
        produce a larger sum of `extra` draws because the gross-up
        accounts for state tax."""
        from tax_optimizer.withdrawals import cover_deficit
        from tax_optimizer.tax.regimes import TCJA_EXTENDED
        from tax_optimizer.tax.state import CA, STATELESS, state_tax
        from tax_optimizer.state import State

        regime = TCJA_EXTENDED
        base_kwargs: dict = dict(
            wages=0.0, interest=0.0, ordinary_div=0.0,
            qualified_div=0.0, ltcg=0.0, pension=0.0,
            pretax_withdrawal=0.0, roth_conversion=0.0,
            social_security=0.0,
        )

        st = State(
            year=0, spouse_a_age=70, spouse_b_age=70,
            spouse_a_pretax=500_000.0, spouse_b_pretax=500_000.0,
            roth=0.0, taxable=0.0, hsa=0.0, pension_balance=0.0,
        )

        def make_state_tax_fn(state_regime):
            def fn(kw: dict, ss_taxable: float) -> float:
                return state_tax(
                    regime=state_regime,
                    filing_status="married_joint",
                    wages_box1=kw.get("wages", 0.0),
                    interest=kw.get("interest", 0.0),
                    ordinary_div=kw.get("ordinary_div", 0.0),
                    qualified_div=kw.get("qualified_div", 0.0),
                    ltcg=kw.get("ltcg", 0.0),
                    pension=kw.get("pension", 0.0),
                    pretax_withdrawal=kw.get("pretax_withdrawal", 0.0),
                    roth_conversion=kw.get("roth_conversion", 0.0),
                    social_security=kw.get("social_security", 0.0),
                    ss_taxable_federal=ss_taxable,
                    hsa_contrib=0.0,
                    age_a=70, age_b=70, alive_a=True, alive_b=True,
                )["state_tax"]
            return fn

        extra_nv, _ = cover_deficit(
            deficit=100_000.0, state=st, base_kwargs=base_kwargs,
            basis_frac=1.0, regime=regime, filing_status="married_joint",
            state_tax_fn=make_state_tax_fn(STATELESS),
        )
        extra_ca, _ = cover_deficit(
            deficit=100_000.0, state=st, base_kwargs=base_kwargs,
            basis_frac=1.0, regime=regime, filing_status="married_joint",
            state_tax_fn=make_state_tax_fn(CA),
        )
        # CA total gross should exceed NV total gross because the
        # solver had to cover an additional ~9% state-tax bite.
        nv_total = extra_nv["pretax"]
        ca_total = extra_ca["pretax"]
        assert ca_total > nv_total + 3_000.0, (
            f"CA cascade ${ca_total:,.0f} should exceed NV "
            f"${nv_total:,.0f} by enough to cover ~9% CA tax on a "
            f"$100k net target."
        )


# ---------------------------------------------------------------------------
# F10 — pension annuity initialization with `>= start_age`
# ---------------------------------------------------------------------------

class TestPensionAnnuityInitWhenAlreadyAtNRD:
    """If simulation starts at-or-past pension NRD, the annuity must
    still initialize on year 0. Pre-v6.2 the strict `==` check left
    `state.pension_annuity` at 0 forever."""

    def test_simulation_starting_at_nrd_pays_pension(self) -> None:
        from tax_optimizer.inputs import PensionInputs

        cfg = Config(
            horizon_age=70,
            nominal_growth_rate=0.0,
            taxable_equity_div_yield=0.0,
            taxable_bond_interest_yield=0.0,
            taxable_drag=0.0,
            inflation=0.0,
            spending=SpendingProfile.flat(base_spending=80_000.0, inflation=0.0),
            annual_expenses_today=80_000.0,
            aca_enabled=False,
            mortality=Mortality(),
        )
        inputs = Inputs(
            spouse_a_age_start=66,
            spouse_b_age_start=66,
            spouse_a_retire_age=66,
            spouse_b_retire_age=66,
            starting=StartingBalances(
                spouse_a_pretax_401k=100_000.0,
                spouse_b_pretax_401k=100_000.0,
                spouse_a_roth_ira=0.0,
                spouse_b_roth_ira=0.0,
                taxable_brokerage=500_000.0,
                hsa=0.0,
            ),
            income=CurrentIncome(),
            ss=SocialSecurity(monthly_spouse_a=0.0, monthly_spouse_b=0.0, start_age=67),
            # NRD (65) is BEFORE the simulation start age (66) — pre-v6.2
            # the `a_age == start_age` check left the pension dormant.
            pension=PensionInputs(
                start_age=65,
                monthly_at_nrd=2_000.0,
                balance_today=400_000.0,
            ),
        )
        df = simulate(cfg, inputs)
        first_year_pension = df.iloc[0]["pension"]
        assert first_year_pension > 20_000.0, (
            f"Year-0 pension was ${first_year_pension:,.0f}; expected "
            f"~$24k (12 × $2k monthly) since spouse A is at NRD when "
            f"the simulation begins. The pre-v6.2 strict-equality "
            f"check on `a_age == start_age` skipped the initialization."
        )


# ===========================================================================
# v6.2 — batch 2 (MEDIUM-severity correctness fixes)
# ===========================================================================


# ---------------------------------------------------------------------------
# F4 — HSA cap downshifts to self-only when one spouse on Medicare
# ---------------------------------------------------------------------------

class TestHsaCapMedicareStagger:
    def test_both_under_65_full_family_limit(self) -> None:
        from tax_optimizer.limits import HSA_FAMILY_LIMIT, hsa_family_cap

        cap = hsa_family_cap(50, 50, either_working=True)
        assert cap == HSA_FAMILY_LIMIT

    def test_both_under_65_with_catchup(self) -> None:
        from tax_optimizer.limits import (
            HSA_CATCH_UP_55,
            HSA_FAMILY_LIMIT,
            hsa_family_cap,
        )

        cap = hsa_family_cap(56, 50, either_working=True)
        assert cap == HSA_FAMILY_LIMIT + HSA_CATCH_UP_55

    def test_one_spouse_medicare_eligible_downshifts_to_self_only(self) -> None:
        from tax_optimizer.limits import (
            HSA_CATCH_UP_55,
            HSA_FAMILY_LIMIT,
            HSA_SELF_LIMIT,
            hsa_family_cap,
        )

        # Spouse A is 66 (Medicare), Spouse B is 60 (HDHP-eligible).
        cap = hsa_family_cap(66, 60, either_working=True)
        # Self-only + 55+ catch-up for the working spouse.
        assert cap == HSA_SELF_LIMIT + HSA_CATCH_UP_55
        # Cap is materially smaller than the family limit.
        assert cap < HSA_FAMILY_LIMIT

    def test_both_medicare_zero(self) -> None:
        from tax_optimizer.limits import hsa_family_cap

        cap = hsa_family_cap(66, 66, either_working=True)
        assert cap == 0.0


# ---------------------------------------------------------------------------
# F5 — basis_frac clamp prevents negative LTCG
# ---------------------------------------------------------------------------

class TestBasisFracClamp:
    def test_basis_frac_above_one_does_not_fabricate_negative_gain(self) -> None:
        """A `basis_frac=1.1` (loss position) used to flow through to
        `gain = mid * (1 - 1.1) = -0.1*mid`, producing a "negative
        LTCG" line in the federal_tax kwargs and pulling AGI down.
        The solver now clamps to 1.0."""
        from tax_optimizer.tax.regimes import TCJA_EXTENDED
        from tax_optimizer.withdrawals import _solve_taxable_for_net

        base_kwargs: dict = dict(
            wages=0.0, interest=0.0, ordinary_div=0.0,
            qualified_div=0.0, ltcg=0.0, pension=0.0,
            pretax_withdrawal=0.0, roth_conversion=0.0,
            social_security=0.0,
        )
        # basis_frac = 1.0 ⇒ tax-free taxable draw (no gain).
        gross_at_one = _solve_taxable_for_net(
            50_000.0, base_kwargs,
            basis_frac=1.0,
            regime=TCJA_EXTENDED, filing_status="married_joint",
        )
        # basis_frac = 1.5 (would imply a loss) should clamp to 1.0
        # and produce the same gross — NOT a smaller gross from a
        # phantom AGI reduction.
        gross_at_high = _solve_taxable_for_net(
            50_000.0, base_kwargs,
            basis_frac=1.5,
            regime=TCJA_EXTENDED, filing_status="married_joint",
        )
        assert gross_at_high == pytest.approx(gross_at_one, rel=1e-3)


# ---------------------------------------------------------------------------
# F6 — LTC shock anchored to end of life, not horizon
# ---------------------------------------------------------------------------

class TestLtcShockAnchoredToLife:
    def test_years_until_death_overrides_horizon(self) -> None:
        from tax_optimizer.spending import (
            LongTermCareShock,
            SpendingPhase,
            SpendingProfile,
        )

        profile = SpendingProfile(
            base_spending=100_000.0,
            inflation=0.0,
            phases=[SpendingPhase(0, 200, 1.0, "flat")],
            ltc_shock=LongTermCareShock(years=3, annual_cost_today=80_000.0),
        )
        # Far from horizon, far from death → no LTC.
        rec_far, _ = profile.amount_for(
            year_offset=10, age_a=70,
            years_until_horizon=20, years_until_death=20,
        )
        assert rec_far == pytest.approx(100_000.0)
        # 2 years before death (years_until_death=2 < ltc.years=3) →
        # shock fires, REGARDLESS of years_until_horizon being large.
        rec_dying, _ = profile.amount_for(
            year_offset=28, age_a=88,
            years_until_horizon=20, years_until_death=2,
        )
        assert rec_dying == pytest.approx(100_000.0 + 80_000.0)

    def test_legacy_horizon_only_callers_still_work(self) -> None:
        from tax_optimizer.spending import (
            LongTermCareShock,
            SpendingPhase,
            SpendingProfile,
        )

        profile = SpendingProfile(
            base_spending=100_000.0,
            inflation=0.0,
            phases=[SpendingPhase(0, 200, 1.0, "flat")],
            ltc_shock=LongTermCareShock(years=2, annual_cost_today=60_000.0),
        )
        # No `years_until_death` argument → behaves like pre-v6.2 with
        # `years_until_horizon` as the LTC anchor.
        rec, _ = profile.amount_for(
            year_offset=29, age_a=89, years_until_horizon=1,
        )
        assert rec == pytest.approx(160_000.0)

    def test_simulator_uses_horizon_when_only_one_spouse_has_death(self) -> None:
        # Pre-fix the simulator anchored LTC to whichever spouse had a
        # year_of_death set, even when the OTHER spouse was alive to
        # horizon. So in an MFJ household where only spouse A has a
        # death year (B alive to horizon), LTC fired at A's death
        # rather than at the household's actual end-of-life (the
        # horizon, since B never dies). Post-fix the household's last
        # year is the horizon, so LTC fires near horizon.
        from tax_optimizer import Config, Inputs
        from tax_optimizer.inputs import CurrentContrib, StartingBalances
        from tax_optimizer.mortality import Mortality
        from tax_optimizer.simulator import simulate
        from tax_optimizer.spending import (
            LongTermCareShock,
            SpendingPhase,
            SpendingProfile,
        )

        # 2-year LTC shock at $60k/yr (today's dollars). Mortality:
        # spouse A dies at year 5, spouse B is alive to horizon (15
        # years). Horizon - LTC.years = 14. So LTC should fire in
        # the final 2 years (year_offset 14, 15), NOT at year 3, 4
        # (the two years before A's death).
        #
        # We zero the HSA so the LTC cost actually shows up on the
        # `spending_need` row (HSA paydown otherwise zeroes the
        # excess medical-need above $50k base, which would obscure
        # the test signal).
        cfg = Config(
            horizon_age=65,  # spouse_a starts at 50 → 16-yr horizon
            mortality=Mortality(
                year_of_death_a=5,
                # year_of_death_b stays None → alive to horizon.
            ),
            spending=SpendingProfile(
                base_spending=50_000.0,
                inflation=0.0,
                phases=[SpendingPhase(0, 100, 1.0, "flat")],
                ltc_shock=LongTermCareShock(years=2, annual_cost_today=60_000.0),
            ),
            inflation=0.0,
        )
        inp = Inputs(
            starting=StartingBalances(hsa=0.0),
            contrib=CurrentContrib(hsa_family=0.0),
        )
        df = simulate(cfg, inp)
        spend = df["spending_need"].tolist()
        n = len(spend)
        # Years 3, 4 (pre-A-death window): no LTC bump.
        assert max(spend[3], spend[4]) < 90_000.0, (
            f"LTC must NOT fire pre-A's-death; "
            f"yrs 3,4 spending={spend[3]:,.0f},{spend[4]:,.0f}"
        )
        # Final 2 years (horizon-aligned): LTC fires.
        last_two_min = min(spend[n - 2], spend[n - 1])
        assert last_two_min > 100_000.0, (
            f"LTC must fire in the final 2 years near horizon; "
            f"got spend[-2:]={spend[n-2]:,.0f},{spend[n-1]:,.0f}"
        )


# ---------------------------------------------------------------------------
# F7 / F8 — min_balance and MC ruin both include HSA post-65
# ---------------------------------------------------------------------------

class TestMinBalanceIncludesHsaPost65:
    def test_min_balance_counts_hsa_when_either_spouse_65_plus(self) -> None:
        import pandas as pd

        from tax_optimizer.metrics import summarize

        # Build a tiny synthetic frame with one row pre-65 and one
        # row post-65. Pre-65 HSA should NOT count; post-65 it should.
        df = pd.DataFrame({
            "federal_tax": [10_000.0, 10_000.0],
            "irmaa": [0.0, 0.0],
            "irmaa_tier": [0, 0],
            "marginal": [0.22, 0.22],
            "spouse_a_age": [60, 70],
            "spouse_b_age": [60, 70],
            "pretax_balance": [100_000.0, 0.0],
            "roth_balance": [0.0, 0.0],
            "taxable_balance": [0.0, 0.0],
            "hsa_balance": [200_000.0, 200_000.0],
            "agi": [0.0, 0.0],
            "spending_need": [50_000.0, 50_000.0],
        })
        result = summarize(df)
        # Pre-65 row liquid = 100k (HSA omitted). Post-65 = 200k.
        # Minimum = 100k.
        assert result["min_balance"] == pytest.approx(100_000.0)


# F8 covered indirectly above — the same HSA-post-65 logic is in
# monte_carlo._ruin_year_offset. Direct unit test:
class TestMcRuinIncludesHsaPost65:
    def test_no_ruin_when_hsa_covers_post_65(self) -> None:
        import pandas as pd

        from tax_optimizer.monte_carlo import _ruin_year_offset

        df = pd.DataFrame({
            "spouse_a_age": [70],
            "spouse_b_age": [70],
            "pretax_balance": [0.0],
            "roth_balance": [0.0],
            "taxable_balance": [0.0],
            "hsa_balance": [500_000.0],
            "spending_need": [100_000.0],
            "unfunded": [0.0],
        })
        offset = _ruin_year_offset(df)
        assert offset == -1, (
            "Post-65 household with $500k in HSA and $100k need should "
            "NOT count as ruined — HSA is a stealth IRA after 65."
        )

    def test_ruin_when_only_hsa_pre_65(self) -> None:
        import pandas as pd

        from tax_optimizer.monte_carlo import _ruin_year_offset

        df = pd.DataFrame({
            "spouse_a_age": [60],
            "spouse_b_age": [60],
            "pretax_balance": [0.0],
            "roth_balance": [0.0],
            "taxable_balance": [0.0],
            "hsa_balance": [500_000.0],
            "spending_need": [100_000.0],
            "unfunded": [0.0],
        })
        offset = _ruin_year_offset(df)
        assert offset == 0, (
            "Pre-65 household with only HSA can't tap it for general "
            "spending — should register as ruined."
        )


# ===========================================================================
# v6.2 — batch 3 (LOW-severity defensive / API)
# ===========================================================================


# ---------------------------------------------------------------------------
# F9 — BootstrapModel validators
# ---------------------------------------------------------------------------

class TestBootstrapModelValidators:
    def test_zero_block_size_rejected(self) -> None:
        from tax_optimizer.market import BootstrapModel

        with pytest.raises(ValueError, match="block_size must be > 0"):
            BootstrapModel(block_size=0)

    def test_negative_block_size_rejected(self) -> None:
        from tax_optimizer.market import BootstrapModel

        with pytest.raises(ValueError, match="block_size must be > 0"):
            BootstrapModel(block_size=-1)

    def test_history_length_mismatch_rejected(self) -> None:
        from tax_optimizer.market import BootstrapModel

        with pytest.raises(ValueError, match="same length"):
            BootstrapModel(
                block_size=2,
                equity_history=(0.1, 0.2, 0.3),
                bond_history=(0.04, 0.05),
            )

    def test_block_size_larger_than_history_rejected(self) -> None:
        from tax_optimizer.market import BootstrapModel

        with pytest.raises(ValueError, match="<= history length"):
            BootstrapModel(
                block_size=10,
                equity_history=(0.1, 0.2, 0.3),
                bond_history=(0.04, 0.05, 0.06),
            )

    def test_begin_path_zero_years_rejected(self) -> None:
        import numpy as np
        from tax_optimizer.market import BootstrapModel

        m = BootstrapModel(block_size=3)
        with pytest.raises(ValueError, match="n_years must be > 0"):
            m.begin_path(0, np.random.default_rng(0))


# ---------------------------------------------------------------------------
# F13 — simulate_paths rejects n_paths < 1
# ---------------------------------------------------------------------------

class TestSimulatePathsRejectsZero:
    def test_zero_paths_rejected(self) -> None:
        from tax_optimizer.monte_carlo import simulate_paths

        with pytest.raises(ValueError, match="n_paths must be >= 1"):
            simulate_paths(Config(), Inputs(), n_paths=0)

    def test_negative_paths_rejected(self) -> None:
        from tax_optimizer.monte_carlo import simulate_paths

        with pytest.raises(ValueError, match="n_paths must be >= 1"):
            simulate_paths(Config(), Inputs(), n_paths=-5)


# ---------------------------------------------------------------------------
# F18 — pension kink indexes forward with wage growth
# ---------------------------------------------------------------------------

class TestPensionKinkIndexesForward:
    def test_kink_grows_with_wage_growth_across_decades(self) -> None:
        """Project two identical pensions across 20 years, one with
        the (correct) inflation-indexed kink and one with the (old)
        frozen kink. The frozen-kink projection should over-credit
        because more of the salary spills into the high band as the
        salary grows past the un-indexed threshold.
        """
        from tax_optimizer.pension import (
            PENSION_QTR_SSWB,
            pension_annual_credit,
            project_pension_balance,
        )

        salary_today = 100_000.0
        years = 20
        wage_growth = 0.03

        # Projected balance with the indexed kink (current behavior).
        bal_indexed = project_pension_balance(
            start_balance=0.0,
            start_earnings=salary_today,
            years_to_retire=years,
            wage_growth=wage_growth,
        )

        # Reference: a hand-rolled projection with the FROZEN kink
        # (i.e. pre-v6.2 behavior). With wage growth past 30 years,
        # the un-indexed kink lets MORE of the salary into the 11%
        # band each year. So the frozen-kink balance should be
        # strictly LARGER than the indexed one.
        bal_frozen = 0.0
        salary = salary_today
        from tax_optimizer.pension import PENSION_INTEREST
        for _ in range(years):
            bal_frozen = bal_frozen * (1 + PENSION_INTEREST) + pension_annual_credit(
                salary, qtr_sswb=PENSION_QTR_SSWB
            )
            salary *= 1 + wage_growth

        # The fix should produce a SMALLER balance (more of the salary
        # still falls in the 6% band each year because the kink kept up).
        assert bal_indexed < bal_frozen, (
            f"Indexed-kink projection (${bal_indexed:,.0f}) should be "
            f"smaller than frozen-kink (${bal_frozen:,.0f}) — the kink "
            f"should grow alongside the salary, not stay at 2025's "
            f"value indefinitely."
        )


# ---------------------------------------------------------------------------
# F12 — survivor_label docstring matches code
# ---------------------------------------------------------------------------

class TestSurvivorLabelDocstring:
    def test_docstring_lists_neither_explicitly(self) -> None:
        """The docstring should explicitly call out the four return
        values so callers know `'neither'` is a possible string (and
        not just `None`)."""
        from tax_optimizer.mortality import Mortality

        doc = Mortality.survivor_label.__doc__ or ""
        for token in ("None", '"a"', '"b"', '"neither"'):
            assert token in doc, (
                f"survivor_label docstring is missing {token!r}; the "
                f"prior docstring claimed `None` for both-alive and "
                f"both-dead but the code returns the string "
                f"'neither' for both-dead."
            )
