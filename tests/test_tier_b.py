"""Tests for Tier-B modeling-gap fills.

Each `Test*` class targets one fix in the original review:

  * TB-7  — State income tax (regimes, retirement exclusions, etc.)
  * TB-8  — Backdoor Roth IRA (pro-rata aware)
  * TB-9  — Mega-backdoor Roth (after-tax 401(k) → Roth, §415(c) cap)
  * TB-10 — HSA after 65 (deficit-cascade ordinary-tax bucket)
  * TB-11 — Bequest tax in terminal-NW objective (heir_marginal_rate)
  * TB-13 — Spousal IRA contributions (Traditional + direct Roth)
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from tax_optimizer import (
    Config,
    Inputs,
    Mortality,
    SocialSecurity,
    SpendingProfile,
    StartingBalances,
    simulate,
)
from tax_optimizer.inputs import CurrentIncome
from tax_optimizer.ira import allocate_ira_contributions
from tax_optimizer.limits import (
    IRA_CATCH_UP_50,
    IRA_CONTRIBUTION_LIMIT,
    SECTION_415C_LIMIT,
    ira_contribution_cap,
    roth_ira_phaseout_factor,
)
from tax_optimizer.metrics import terminal_after_tax_nw
from tax_optimizer.tax.state import (
    CA,
    IL,
    MA,
    NY,
    STATELESS,
    StateTaxRegime,
    lookup,
    state_tax,
)


# ---------------------------------------------------------------------------
# Test fixtures: minimal scenario shared across tests
# ---------------------------------------------------------------------------


def _short_horizon_inputs(**overrides) -> Inputs:
    """A small, stable scenario: both 60, retire next year, no SS yet,
    moderate balances. Good for testing single-year mechanics without
    multi-decade noise."""
    base = Inputs(
        spouse_a_age_start=60,
        spouse_b_age_start=60,
        spouse_a_retire_age=61,
        spouse_b_retire_age=61,
        spouse_a_total_contrib_pct=0.0,
        spouse_b_total_contrib_pct=0.0,
        spouse_a_roth_401k_pct=0.0,
        spouse_b_roth_401k_pct=0.0,
        income=CurrentIncome(
            spouse_a_gross=120_000.0,
            spouse_b_gross=80_000.0,
            spouse_a_bonus=0.0,
            interest=0.0,
            capital_gains=0.0,
            dividends=0.0,
        ),
        starting=StartingBalances(
            spouse_a_pretax_401k=300_000.0,
            spouse_b_pretax_401k=200_000.0,
            spouse_a_roth_ira=50_000.0,
            spouse_b_roth_ira=0.0,
            spouse_a_pretax_ira=0.0,
            spouse_b_pretax_ira=0.0,
            hsa=20_000.0,
            taxable_brokerage=200_000.0,
        ),
        ss=SocialSecurity(
            monthly_spouse_a=2_400.0,
            monthly_spouse_b=2_000.0,
            start_age=70,
            fra_a=67,
            fra_b=67,
        ),
        # annual_expenses intentionally omitted: simulator reads
        # `cfg.spending.base_spending` from `_short_horizon_cfg`.
    )
    return replace(base, **overrides) if overrides else base


def _short_horizon_cfg(horizon_age: int = 65, **overrides) -> Config:
    base = Config(
        horizon_age=horizon_age,
        inflation=0.025,
        ss_cola_rate=0.025,
        bracket_indexing_rate=0.025,
        spending=SpendingProfile(base_spending=80_000.0, inflation=0.025),
        mortality=Mortality(year_of_death_a=None, year_of_death_b=None),
    )
    return replace(base, **overrides) if overrides else base


# ---------------------------------------------------------------------------
# TB-7: State income tax
# ---------------------------------------------------------------------------


class TestStateRegimeLookup:
    def test_lookup_canonical_aliases(self):
        assert lookup("CA") is CA
        assert lookup("ca") is CA
        assert lookup("NY") is NY
        assert lookup("IL") is IL
        assert lookup("MA") is MA
        assert lookup("stateless") is STATELESS
        assert lookup("none") is STATELESS

    def test_lookup_unknown_raises(self):
        with pytest.raises(KeyError):
            lookup("zz")


class TestStateTaxRegimes:
    """Direct unit tests on the `state_tax` function — bypass the
    simulator and assert the per-regime mechanics."""

    def _kw(self, **overrides):
        base = dict(
            filing_status="mfj",
            wages_box1=200_000.0,
            interest=0.0,
            ordinary_div=0.0,
            qualified_div=0.0,
            ltcg=0.0,
            pension=0.0,
            pretax_withdrawal=0.0,
            roth_conversion=0.0,
            social_security=0.0,
            ss_taxable_federal=0.0,
            hsa_contrib=0.0,
            age_a=55,
            age_b=55,
            alive_a=True,
            alive_b=True,
        )
        base.update(overrides)
        return base

    def test_stateless_zero_tax(self):
        out = state_tax(regime=STATELESS, **self._kw())
        assert out["state_tax"] == 0.0
        assert out["state_taxable_income"] == 0.0
        assert out["state_marginal"] == 0.0

    def test_california_progressive(self):
        out = state_tax(regime=CA, **self._kw())
        # $200k wages MFJ → state ordinary ~9.3% bracket → > $10k
        assert out["state_tax"] > 10_000.0
        assert out["state_marginal"] >= 0.093

    def test_marginal_at_zero_income_returns_first_slab(self):
        # Pre-fix: zero ordinary income returned 0.0 for the marginal
        # rate (loop seeded at 0.0 and `>` excluded `lo == 0.0`),
        # silently mis-pricing the *next* dollar of pretax / Roth
        # conversion in early-retirement gap years. Post-fix: seed
        # with the first slab's rate and use `>=`.
        out = state_tax(regime=CA, **self._kw(wages_box1=0.0))
        # CA's first ordinary slab starts at 1%.
        assert out["state_marginal"] == pytest.approx(0.01, rel=1e-6)
        assert out["state_tax"] == pytest.approx(0.0, abs=1.0)

    def test_marginal_at_exact_bracket_boundary(self):
        # Income that lands exactly on a CA bracket lower bound
        # should report the *next-dollar* rate. The state standard
        # deduction is subtracted before bracket lookup, so we drive
        # ordinary income to (lo + std_deduction) so taxable_ordinary
        # lands exactly on the boundary.
        from tax_optimizer.tax.state import CA as _CA

        brackets = _CA.ord_brackets("mfj")
        # Pick a mid-stack bracket boundary (avoid the open-ended top
        # bracket whose `lo` is the prior bracket's `hi`, and avoid
        # the first slab whose `lo` is 0).
        lo, _hi, rate = brackets[3]
        std = _CA.std_deduction("mfj")
        out = state_tax(regime=CA, **self._kw(wages_box1=lo + std))
        # On exact-boundary, marginal must reflect the slab the next
        # dollar would land in. Pre-fix this returned the slab below
        # (the `>` test excluded the boundary value).
        assert out["state_marginal"] == pytest.approx(rate, rel=1e-9)

    def test_california_hsa_addback(self):
        # CA does NOT conform to HSA. $4k HSA contrib should ADD BACK
        # to state income → higher state tax than without addback.
        no_hsa = state_tax(regime=CA, **self._kw(hsa_contrib=0.0))
        with_hsa = state_tax(regime=CA, **self._kw(hsa_contrib=4_000.0))
        # Because federal Box-1 wages already excluded the HSA, the
        # addback raises CA state taxable income by 4k → tax higher.
        assert with_hsa["state_tax"] > no_hsa["state_tax"]
        # Difference roughly 4000 × marginal rate.
        diff = with_hsa["state_tax"] - no_hsa["state_tax"]
        assert 4_000 * 0.06 < diff < 4_000 * 0.12

    def test_illinois_full_retirement_exclusion(self):
        # IL exempts ALL retirement distributions. A retiree with
        # $80k IRA withdrawal should owe ZERO state tax.
        out = state_tax(
            regime=IL,
            **self._kw(
                wages_box1=0.0,
                pretax_withdrawal=80_000.0,
                age_a=68,
                age_b=68,
            ),
        )
        assert out["state_tax"] == 0.0

    def test_illinois_taxes_wages(self):
        # But IL DOES tax W-2 wages at the flat 4.95% rate.
        out = state_tax(regime=IL, **self._kw(wages_box1=100_000.0))
        # 100k - 5550 std dn → 94,450 × 4.95% ≈ 4,675
        assert 4_500.0 < out["state_tax"] < 5_000.0

    def test_ny_partial_retirement_exclusion(self):
        # NY excludes $20k/filer at 59½+. Two filers → $40k total
        # exclusion on pension/IRA/Roth-conv income. Use $80k IRA
        # withdrawal so the std-deduction floor doesn't mask the
        # exclusion.
        out_old = state_tax(
            regime=NY,
            **self._kw(
                wages_box1=0.0,
                pretax_withdrawal=80_000.0,
                age_a=65,
                age_b=65,
            ),
        )
        out_young = state_tax(
            regime=NY,
            **self._kw(
                wages_box1=0.0,
                pretax_withdrawal=80_000.0,
                age_a=55,
                age_b=55,
            ),
        )
        # Older couple's taxable income is $40k lower (the exclusion).
        assert out_old["state_tax"] < out_young["state_tax"]
        assert (
            out_young["state_taxable_income"]
            - out_old["state_taxable_income"]
            == pytest.approx(40_000.0, abs=1.0)
        )

    def test_ss_exempt_in_bundled_regimes(self):
        # All four bundled regimes (CA, NY, IL, MA) exempt SS at the
        # state level.
        for regime in (CA, NY, IL, MA):
            no_ss = state_tax(regime=regime, **self._kw(wages_box1=80_000.0))
            with_ss = state_tax(
                regime=regime,
                **self._kw(
                    wages_box1=80_000.0,
                    social_security=30_000.0,
                    ss_taxable_federal=25_500.0,  # 85% of 30k
                ),
            )
            assert with_ss["state_tax"] == pytest.approx(no_ss["state_tax"])

    def test_inflation_indexing(self):
        # Inflated regime should have higher std deduction + bracket
        # thresholds → lower tax for the same nominal income.
        ca_2y = CA.inflated(1.05)
        out_now = state_tax(regime=CA, wages_box1=200_000.0, **{
            k: v for k, v in self._kw().items() if k != "wages_box1"
        })
        out_2y = state_tax(regime=ca_2y, wages_box1=200_000.0, **{
            k: v for k, v in self._kw().items() if k != "wages_box1"
        })
        # Same nominal income → indexed regime taxes less.
        assert out_2y["state_tax"] < out_now["state_tax"]


class TestStateTaxInSimulator:
    """End-to-end: state tax shows up in the dataframe and reduces
    terminal NW relative to STATELESS."""

    def test_state_regime_column(self):
        cfg = _short_horizon_cfg()
        cfg = replace(cfg, state_regime=CA)
        df = simulate(cfg, _short_horizon_inputs())
        assert "state_tax" in df.columns
        assert "state_marginal" in df.columns
        assert "state_regime" in df.columns
        assert df["state_regime"].iloc[0] == "CA"
        # CA → first year (working) state tax positive.
        assert df["state_tax"].iloc[0] > 0

    def test_stateless_default(self):
        cfg = _short_horizon_cfg()
        df = simulate(cfg, _short_horizon_inputs())
        # Default is STATELESS → all zeros.
        assert (df["state_tax"] == 0.0).all()
        assert (df["state_regime"] == "stateless").all()

    def test_ca_lowers_terminal_nw(self):
        cfg_none = _short_horizon_cfg()
        cfg_ca = replace(cfg_none, state_regime=CA)
        df_none = simulate(cfg_none, _short_horizon_inputs())
        df_ca = simulate(cfg_ca, _short_horizon_inputs())
        # Adding a 9% bracket on $200k wages should cost real money.
        nw_none = terminal_after_tax_nw(df_none, heir_marginal_rate=0.0)
        nw_ca = terminal_after_tax_nw(df_ca, heir_marginal_rate=0.0)
        assert nw_ca < nw_none

    def test_state_regime_change(self):
        # Move from CA to STATELESS at year 2 (e.g. retire to FL).
        cfg = _short_horizon_cfg(
            horizon_age=64,
            state_regime=CA,
            state_regime_change_year_offset=2,
            state_regime_change_target=STATELESS,
        )
        df = simulate(cfg, _short_horizon_inputs())
        assert df["state_regime"].iloc[0] == "CA"
        assert df["state_regime"].iloc[1] == "CA"
        assert df["state_regime"].iloc[2] == "stateless"
        assert df["state_tax"].iloc[2] == 0.0


# ---------------------------------------------------------------------------
# TB-11: Bequest tax in terminal NW
# ---------------------------------------------------------------------------


class TestBequestTax:
    def _df(self, *, pretax=0.0, roth=0.0, taxable=0.0, hsa=0.0) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "pretax_balance": [pretax],
                "roth_balance": [roth],
                "taxable_balance": [taxable],
                "hsa_balance": [hsa],
            }
        )

    def test_zero_rate_recovers_v1_behavior(self):
        df = self._df(pretax=1_000_000, roth=1_000_000, taxable=500_000, hsa=100_000)
        # heir_marginal_rate=0 ⇒ $1 pretax = $1 Roth.
        nw = terminal_after_tax_nw(df, heir_marginal_rate=0.0)
        assert nw == pytest.approx(2_600_000.0)

    def test_default_rate_haircuts_pretax(self):
        df = self._df(pretax=1_000_000, roth=0, taxable=0, hsa=0)
        # Default 0.22 → $1M pretax = $780k after heir tax.
        nw = terminal_after_tax_nw(df)
        assert nw == pytest.approx(780_000.0)

    def test_high_rate_punishes_pretax(self):
        df_pretax = self._df(pretax=1_000_000)
        df_roth = self._df(roth=1_000_000)
        nw_pretax = terminal_after_tax_nw(df_pretax, heir_marginal_rate=0.32)
        nw_roth = terminal_after_tax_nw(df_roth, heir_marginal_rate=0.32)
        # Roth-heavy bequest preserves $320k more value.
        assert nw_roth - nw_pretax == pytest.approx(320_000.0)

    def test_hsa_treated_like_pretax(self):
        # HSA inherited by non-spouse is ordinary income to heir.
        df = self._df(hsa=1_000_000)
        nw = terminal_after_tax_nw(df, heir_marginal_rate=0.30)
        assert nw == pytest.approx(700_000.0)

    def test_legacy_alias(self):
        df = self._df(pretax=1_000_000)
        # Old `marginal_rate` keyword still works and takes precedence.
        nw_legacy = terminal_after_tax_nw(df, marginal_rate=0.30)
        nw_new = terminal_after_tax_nw(df, heir_marginal_rate=0.30)
        assert nw_legacy == nw_new

    def test_cfg_threading_via_simulate(self):
        # Bequest rate change should not affect the simulation
        # itself, only the post-hoc NW objective.
        cfg = _short_horizon_cfg()
        df = simulate(cfg, _short_horizon_inputs())
        nw_low = terminal_after_tax_nw(df, heir_marginal_rate=0.0)
        nw_high = terminal_after_tax_nw(df, heir_marginal_rate=0.35)
        assert nw_high < nw_low


# ---------------------------------------------------------------------------
# TB-8 + TB-13: IRA contributions (Traditional / Roth direct / backdoor)
# ---------------------------------------------------------------------------


class TestIRACap:
    def test_under_50(self):
        assert ira_contribution_cap(35) == IRA_CONTRIBUTION_LIMIT

    def test_50_plus_catch_up(self):
        assert ira_contribution_cap(50) == IRA_CONTRIBUTION_LIMIT + IRA_CATCH_UP_50

    def test_phaseout_below_lo_full(self):
        assert roth_ira_phaseout_factor(100_000, "mfj") == 1.0

    def test_phaseout_above_hi_zero(self):
        assert roth_ira_phaseout_factor(500_000, "mfj") == 0.0

    def test_phaseout_midpoint(self):
        # Halfway through the $10k MFJ phase-out window.
        f = roth_ira_phaseout_factor(241_000, "mfj")
        assert 0.4 < f < 0.6


class TestIRAAllocation:
    def test_ineligible_returns_zero(self):
        out = allocate_ira_contributions(
            age=40, eligible=False, pretax_existing=0,
            traditional_target=7_000, roth_direct_target=7_000,
            backdoor_enabled=True, magi_estimate=100_000,
            filing_status="mfj",
        )
        assert out.traditional == out.roth_direct == out.backdoor == 0.0

    def test_traditional_only(self):
        out = allocate_ira_contributions(
            age=40, eligible=True, pretax_existing=0,
            traditional_target=7_000, roth_direct_target=0,
            backdoor_enabled=False, magi_estimate=100_000,
            filing_status="mfj",
        )
        assert out.traditional == 7_000
        assert out.roth_direct == 0.0
        assert out.backdoor == 0.0

    def test_cap_priority_traditional_first(self):
        # Both traditional + roth_direct = $7k each but cap is $7k.
        # Traditional fills first, leaving $0 for roth direct.
        out = allocate_ira_contributions(
            age=40, eligible=True, pretax_existing=0,
            traditional_target=7_000, roth_direct_target=7_000,
            backdoor_enabled=False, magi_estimate=100_000,
            filing_status="mfj",
        )
        assert out.traditional == 7_000
        assert out.roth_direct == 0.0

    def test_phaseout_zeros_direct_roth(self):
        # MAGI $260k > $246k upper bound → zero direct Roth.
        out = allocate_ira_contributions(
            age=40, eligible=True, pretax_existing=0,
            traditional_target=0, roth_direct_target=7_000,
            backdoor_enabled=True, magi_estimate=260_000,
            filing_status="mfj",
        )
        assert out.roth_direct == 0.0
        assert out.backdoor == 7_000  # backdoor fills the room

    def test_clean_backdoor_no_pretax(self):
        # No pretax balance → 0% pro-rata-taxable.
        out = allocate_ira_contributions(
            age=40, eligible=True, pretax_existing=0,
            traditional_target=0, roth_direct_target=0,
            backdoor_enabled=True, magi_estimate=300_000,
            filing_status="mfj",
        )
        assert out.backdoor == 7_000
        assert out.backdoor_taxable_conversion == 0.0

    def test_dirty_backdoor_pro_rata(self):
        # $93k pretax + $7k contribution → 93% taxable on conversion.
        out = allocate_ira_contributions(
            age=40, eligible=True, pretax_existing=93_000,
            traditional_target=0, roth_direct_target=0,
            backdoor_enabled=True, magi_estimate=300_000,
            filing_status="mfj",
        )
        assert out.backdoor == 7_000
        assert out.backdoor_taxable_conversion == pytest.approx(7_000 * 0.93, rel=0.01)

    def test_50_plus_catch_up_in_allocation(self):
        # 50+ gets +$1k catch-up.
        out = allocate_ira_contributions(
            age=55, eligible=True, pretax_existing=0,
            traditional_target=8_000, roth_direct_target=0,
            backdoor_enabled=False, magi_estimate=100_000,
            filing_status="mfj",
        )
        assert out.traditional == 8_000

    def test_total_cash_outflow(self):
        # All three lines come out of after-tax cash.
        out = allocate_ira_contributions(
            age=40, eligible=True, pretax_existing=0,
            traditional_target=3_000, roth_direct_target=2_000,
            backdoor_enabled=True, magi_estimate=100_000,
            filing_status="mfj",
        )
        # 3k trad + 2k direct + 2k backdoor (rest of cap) = 7k total
        assert out.total_cash_outflow == 7_000


class TestIRAInSimulator:
    def test_direct_roth_lands_in_roth(self):
        cfg = _short_horizon_cfg(horizon_age=62)
        inputs = _short_horizon_inputs()
        baseline = simulate(cfg, inputs)
        with_roth = simulate(
            cfg,
            replace(
                inputs,
                spouse_a_roth_ira_contrib=7_000.0,
                spouse_b_roth_ira_contrib=7_000.0,
            ),
        )
        # First-year Roth balance grows by ~$14k in the case with
        # contributions; balances grow market-rate, so the difference
        # should be at least the contribution amount.
        diff = with_roth["roth_balance"].iloc[0] - baseline["roth_balance"].iloc[0]
        assert diff >= 13_500  # ~14k minus tiny growth difference

    def test_direct_roth_above_phaseout_blocked(self):
        # Bump salary so MAGI > MFJ phase-out.
        base = _short_horizon_inputs()
        inputs = replace(
            base,
            income=replace(
                base.income, spouse_a_gross=300_000.0, spouse_b_gross=200_000.0
            ),
            spouse_a_roth_ira_contrib=7_000.0,
            spouse_b_roth_ira_contrib=7_000.0,
        )
        df = simulate(_short_horizon_cfg(horizon_age=62), inputs)
        # Direct contribution must be zero in working year.
        assert df["ira_roth_direct_a"].iloc[0] == 0.0
        assert df["ira_roth_direct_b"].iloc[0] == 0.0

    def test_backdoor_works_at_high_income(self):
        base = _short_horizon_inputs()
        inputs = replace(
            base,
            income=replace(
                base.income, spouse_a_gross=300_000.0, spouse_b_gross=200_000.0
            ),
            spouse_a_backdoor_roth=True,
            spouse_b_backdoor_roth=True,
            starting=replace(
                base.starting,
                spouse_a_pretax_401k=0.0,
                spouse_b_pretax_401k=0.0,
                spouse_a_pretax_ira=0.0,
                spouse_b_pretax_ira=0.0,
            ),
        )
        df = simulate(_short_horizon_cfg(horizon_age=62), inputs)
        # Year 0: clean backdoor → cap each (50+ catch-up = $8k),
        # no taxable conversion.
        cap = ira_contribution_cap(60)
        assert df["ira_backdoor_a"].iloc[0] == pytest.approx(cap, abs=1)
        assert df["ira_backdoor_b"].iloc[0] == pytest.approx(cap, abs=1)
        assert df["ira_backdoor_taxable_conv"].iloc[0] == pytest.approx(0.0, abs=1)

    def test_traditional_ira_reduces_agi(self):
        cfg = _short_horizon_cfg(horizon_age=62)
        baseline = simulate(cfg, _short_horizon_inputs())
        with_trad = simulate(
            cfg,
            replace(
                _short_horizon_inputs(),
                spouse_a_traditional_ira_contrib=7_000.0,
            ),
        )
        # AGI in working year drops by ~$7k.
        agi_diff = baseline["agi"].iloc[0] - with_trad["agi"].iloc[0]
        assert 6_500 < agi_diff < 7_500


# ---------------------------------------------------------------------------
# TB-9: Mega-backdoor Roth
# ---------------------------------------------------------------------------


class TestMegaBackdoor:
    def test_disabled_by_default(self):
        df = simulate(_short_horizon_cfg(horizon_age=62), _short_horizon_inputs())
        assert (df["mega_backdoor_a"] == 0.0).all()
        assert (df["mega_backdoor_b"] == 0.0).all()

    def test_enabled_routes_to_roth(self):
        cfg = _short_horizon_cfg(horizon_age=62)
        inputs = replace(
            _short_horizon_inputs(),
            spouse_a_mega_backdoor_enabled=True,
            spouse_a_after_tax_401k_pct=0.20,  # 20% of $120k = $24k
        )
        baseline = simulate(cfg, _short_horizon_inputs())
        with_mb = simulate(cfg, inputs)
        # First-year Roth balance ~$24k higher (+ market growth on top).
        diff = with_mb["roth_balance"].iloc[0] - baseline["roth_balance"].iloc[0]
        assert 23_500 < diff < 26_500
        assert with_mb["mega_backdoor_a"].iloc[0] == pytest.approx(24_000, rel=0.01)

    def test_415c_cap_enforced(self):
        # Try to route 100% of salary → capped at §415(c) - elective.
        cfg = _short_horizon_cfg(horizon_age=62)
        # Max out elective deferrals first.
        inputs = replace(
            _short_horizon_inputs(),
            spouse_a_total_contrib_pct=0.20,  # ~$24k elective at $120k salary
            spouse_a_mega_backdoor_enabled=True,
            spouse_a_after_tax_401k_pct=1.0,
        )
        df = simulate(cfg, inputs)
        # After-tax room = §415(c) - elective ~ 70k - 24k = 46k.
        # But spouse_a_salary=120k, so 100% of salary capped at the
        # §415(c) headroom.
        actual = df["mega_backdoor_a"].iloc[0]
        assert actual <= SECTION_415C_LIMIT  # never exceeds §415(c)
        assert 40_000 < actual < 50_000

    def test_no_after_tax_in_retirement(self):
        cfg = _short_horizon_cfg(horizon_age=63)
        inputs = replace(
            _short_horizon_inputs(),
            spouse_a_retire_age=60,
            spouse_b_retire_age=60,
            spouse_a_mega_backdoor_enabled=True,
            spouse_a_after_tax_401k_pct=0.20,
        )
        df = simulate(cfg, inputs)
        # Already retired (start_age=60, retire_age=60) → no after-tax.
        assert (df["mega_backdoor_a"] == 0.0).all()


# ---------------------------------------------------------------------------
# TB-10: HSA after 65
# ---------------------------------------------------------------------------


class TestHSAAfter65:
    def test_hsa_locked_pre_65(self):
        # Pre-65 deficit cascade should NOT touch HSA (LTC shock
        # is the only HSA usage path < 65).
        cfg = _short_horizon_cfg(horizon_age=63)
        base = _short_horizon_inputs()
        inputs = replace(
            base,
            spouse_a_age_start=58,
            spouse_b_age_start=58,
            spouse_a_retire_age=58,
            spouse_b_retire_age=58,
            starting=replace(
                base.starting,
                hsa=50_000.0,
                taxable_brokerage=0.0,
                spouse_a_roth_ira=0.0,
                spouse_b_roth_ira=0.0,
            ),
        )
        df = simulate(cfg, inputs)
        # The HSA bucket grows in deterministic mode (~5%/yr) and the
        # pre-65 cascade only spends it on LTC shocks (none here).
        # First-year balance must reflect contributions + growth, not
        # ordinary-tax drainage.
        assert df["hsa_balance"].iloc[0] >= 49_000

    def test_hsa_unlocked_post_65(self):
        # Post-65 deficit-cascade should drain HSA before pretax. We
        # construct a scenario where withdraw_for_need can't satisfy
        # the spending need from the normal buckets (taxable / roth /
        # pretax all empty), so cover_deficit fires and HSA — newly
        # unlocked at 65 — gets drained.
        cfg = _short_horizon_cfg(horizon_age=70)
        base = _short_horizon_inputs()
        inputs = replace(
            base,
            spouse_a_age_start=66,
            spouse_b_age_start=66,
            spouse_a_retire_age=66,
            spouse_b_retire_age=66,
            starting=replace(
                base.starting,
                hsa=80_000.0,
                taxable_brokerage=0.0,
                spouse_a_roth_ira=0.0,
                spouse_b_roth_ira=0.0,
                spouse_a_pretax_401k=0.0,
                spouse_b_pretax_401k=0.0,
                spouse_a_pretax_ira=0.0,
                spouse_b_pretax_ira=0.0,
            ),
        )
        df = simulate(cfg, inputs)
        # All other buckets are empty, so cover_deficit must drain the
        # HSA. HSA balance should fall meaningfully below starting +
        # growth.
        assert df["hsa_balance"].iloc[-1] < 80_000.0


# ---------------------------------------------------------------------------
# Cross-cutting: full Tier-B suite still simulates cleanly
# ---------------------------------------------------------------------------


class TestTierBIntegration:
    def test_kitchen_sink_scenario_runs(self):
        """A scenario that exercises every Tier-B knob simultaneously
        should simulate without errors and produce sane balances."""
        cfg = _short_horizon_cfg(
            horizon_age=70, state_regime=CA, heir_marginal_rate=0.28
        )
        base = _short_horizon_inputs()
        inputs = replace(
            base,
            income=replace(
                base.income, spouse_a_gross=250_000.0, spouse_b_gross=180_000.0
            ),
            spouse_a_retire_age=67,
            spouse_b_retire_age=67,
            spouse_a_traditional_ira_contrib=0.0,
            spouse_a_backdoor_roth=True,
            spouse_b_backdoor_roth=True,
            spouse_a_mega_backdoor_enabled=True,
            spouse_a_after_tax_401k_pct=0.10,
            spouse_a_employer_match_rate=0.50,
            spouse_a_employer_match_max_pct=0.06,
        )
        df = simulate(cfg, inputs)
        # Sanity: positive terminal NW, no NaNs, all expected columns.
        cap = ira_contribution_cap(60)
        assert df["pretax_balance"].iloc[-1] >= 0
        assert df["roth_balance"].iloc[-1] > 0
        assert df["state_tax"].iloc[0] > 0
        assert df["mega_backdoor_a"].iloc[0] > 0
        assert df["ira_backdoor_a"].iloc[0] == pytest.approx(cap, abs=1)
        assert df.notna().all().all()
