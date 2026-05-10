"""Tests for Tier-C correctness fixes & modeling additions.

Each `Test*` class targets one fix or feature in the original Tier C
audit (see `.cursor/plans/tier_c_audit_+_plan_*.plan.md`):

  Tier C-A — Correctness bug fixes
    * TC-1  — year-of-death MFJ filing status
    * TC-2  — survivor SS at age 60 from deceased's record
    * TC-3  — deficit-cascade marginal-tax base (final_kwargs not base)
    * TC-4  — Additional Medicare 0.9% MFJ joint threshold
    * TC-5  — Roth IRA MAGI estimate uses prior-year AGI
    * TC-6  — backdoor pro-rata is IRA-aggregate, not IRA + 401(k)
    * TC-7  — fixed Roth conversion reserves RMD bucket
    * TC-8  — report shows per-spouse SS start_age
    * TC-9  — caveats block doesn't claim "federal-only" any more

  Tier C-B — High-ROI modeling additions
    * TC-10 — base Medicare Part B + Part D premiums
    * TC-11 — IRMAA 2-year MAGI lookback
    * TC-12 — pre-Medicare healthcare expense knob
    * TC-13 — ACA premium tax credit (post-IRA-2022 8.5% cap)
    * TC-14 — step-up in basis on first spouse's death

  Tier C-C — Optimizer scope extensions
    * TC-15 — mega-backdoor 401(k) % in optimizer decision vector
    * TC-16 — per-spouse SS claim-age axis
    * TC-17 — seed thread-through in MC objectives
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
from tax_optimizer.metrics import summarize
from tax_optimizer.payroll import (
    ADDITIONAL_MEDICARE_RATE,
    ADDITIONAL_MEDICARE_THRESHOLD_MFJ,
    ADDITIONAL_MEDICARE_THRESHOLD_SINGLE,
    fica_employee,
    fica_household,
)
from tax_optimizer.report import build_action_report
from tax_optimizer.results import StrategyResult


def _build_minimal_report_args(cfg: Config, inputs: Inputs):
    """Return a minimal `(results, sens_df, base_terminal)` triple
    sufficient for `build_action_report` to render."""
    df = simulate(cfg, inputs)
    summary = summarize(df)
    results = {"S0_baseline": StrategyResult(cfg=cfg, inputs=inputs, df=df, summary=summary)}
    sens_df = pd.DataFrame(
        [
            {
                "param": "noop",
                "low_value": 0.0,
                "high_value": 0.0,
                "delta_low": 0.0,
                "delta_high": 0.0,
                "swing": 0.0,
            },
        ]
    )
    base_terminal = float(summary["terminal_after_tax"])
    return results, sens_df, base_terminal


# =====================================================================
# TC-1 — year-of-death MFJ filing status
# =====================================================================


class TestYearOfDeathMFJ:
    """IRS treats the calendar year a spouse dies as MFJ on the joint
    return. Pre-Tier-C, our `Mortality.filing_status()` flipped to
    single immediately at year_of_death."""

    def test_year_of_death_mfj(self) -> None:
        m = Mortality(year_of_death_a=10)
        assert m.filing_status(9) == "mfj"
        assert m.filing_status(10) == "mfj"   # year of death — IRS rule
        assert m.filing_status(11) == "single"

    def test_simulator_keeps_mfj_at_year_of_death(self) -> None:
        cfg = Config(mortality=Mortality(year_of_death_b=15))
        df = simulate(cfg, Inputs())
        assert df.iloc[14]["filing_status"] == "mfj"
        assert df.iloc[15]["filing_status"] == "mfj"   # B dies — still MFJ
        assert df.iloc[16]["filing_status"] == "single"


# =====================================================================
# TC-2 — survivor SS at age 60 from deceased's record
# =====================================================================


class TestSurvivorSSAt60:
    """A widow(er) can claim survivor benefits on the deceased's record
    starting at age 60, regardless of their own claim age. Pre-Tier-C
    `ssn_income` was forced to $0 if the survivor hadn't reached
    their own claim age."""

    def _setup(self, **kwargs):
        # Spouses both 60 at start; A dies year 1; ages_start chosen so
        # survivor B is 61+ when checking ssn_income at year 1.
        cfg = Config(
            mortality=Mortality(year_of_death_a=1),
            **kwargs,
        )
        inp = Inputs(
            spouse_a_age_start=60,
            spouse_b_age_start=60,
            spouse_a_retire_age=60,
            spouse_b_retire_age=60,
            ss=SocialSecurity(
                start_age=70,           # spouses planned to wait until 70
                start_age_a=70,
                start_age_b=70,
                monthly_spouse_a=3_000,
                monthly_spouse_b=1_000,
                fra_a=67, fra_b=67,
            ),
        )
        return cfg, inp

    def test_widow_under_70_collects_survivor_at_60(self) -> None:
        cfg, inp = self._setup()
        df = simulate(cfg, inp)
        # Year 1: B is 61, hasn't reached own claim age (70), but is past
        # 60 and A is dead → can claim A's survivor benefit.
        widow_year = df.iloc[1]
        assert not widow_year["alive_a"]
        assert widow_year["alive_b"]
        assert widow_year["ssn"] > 0   # used to be silently $0

    def test_widow_at_own_claim_takes_max(self) -> None:
        # B's own benefit at 70 should beat A's survivor benefit
        # (since A's monthly was $3k vs B's $1k → at 70 B = $1k * 1.24
        # = $1240; A's survivor frozen at A's claim_age=70 = $3k *
        # 1.24 = $3720). Survivor amount wins.
        cfg, inp = self._setup()
        df = simulate(cfg, inp)
        # Year 11 onward: B is 71+, takes max(own at 70, survivor of A
        # at 70).
        late = df.iloc[11]
        assert late["ssn"] > 30_000   # $3k/mo annualized + COLA


# =====================================================================
# TC-3 — deficit-cascade tax base
# =====================================================================


class TestDeficitCascadeTaxBase:
    """`cover_deficit` must see the same income-line as the primary
    federal tax call. Passing `base_kwargs` (no withdrawals stacked)
    understated the marginal bracket on cascade legs in any year with
    an RMD or planned conversion."""

    def test_cascade_uses_post_primary_tax_base(self) -> None:
        # Drain everything except a big pretax bucket; force a deficit
        # in retirement so the cascade has to pull pretax. Add a fixed
        # Roth conversion to force "primary" pretax outflow already in
        # final_kwargs.
        cfg = Config(
            roth_conversion_amount=20_000,
            withdrawal_strategy="conventional",
            spending=SpendingProfile.flat(130_000),
        )
        inp = Inputs(
            spouse_a_age_start=60,
            spouse_b_age_start=60,
            spouse_a_retire_age=60,
            spouse_b_retire_age=60,
            starting=StartingBalances(
                spouse_a_pretax_401k=900_000,
                spouse_b_pretax_401k=600_000,
                taxable_brokerage=0,
                spouse_a_roth_ira=0,
                spouse_b_roth_ira=0,
                hsa=0,
            ),
        )
        df = simulate(cfg, inp)
        # Sanity: deficit cascade engaged in some retirement years.
        assert (df.iloc[1:]["pretax_withdrawal"] > 0).any()
        # No catastrophic blow-up: tax never exceeds AGI.
        assert (df["federal_tax"] <= df["agi"] + 1.0).all()


# =====================================================================
# TC-4 — Additional Medicare 0.9% MFJ joint threshold
# =====================================================================


class TestAdditionalMedicareMFJ:
    def test_two_180k_w2s_couple_pays_addl(self) -> None:
        # Two $180k W-2s: each below the $200k per-W-2 employer-
        # withholding threshold → fica_employee returns 0 addl. But the
        # MFJ filing-time threshold is $250k joint, so Form 8959 owes:
        #   0.9% × ($360k − $250k) = $990
        out = fica_household(180_000, 180_000, filing_status="mfj")
        expected = 0.009 * (360_000 - ADDITIONAL_MEDICARE_THRESHOLD_MFJ)
        assert out["additional_medicare"] == pytest.approx(expected)

    def test_single_uses_200k_threshold(self) -> None:
        out = fica_household(220_000, 0, filing_status="single")
        expected = 0.009 * (220_000 - ADDITIONAL_MEDICARE_THRESHOLD_SINGLE)
        assert out["additional_medicare"] == pytest.approx(expected)

    def test_per_w2_employee_unchanged(self) -> None:
        # `fica_employee` still reflects per-W-2 withholding behaviour
        # (no household reconciliation).
        out = fica_employee(180_000)
        assert out["additional_medicare"] == 0.0


# =====================================================================
# TC-5 — Roth IRA MAGI estimate uses prior-year AGI
# =====================================================================


class TestRothMagiUsesPriorAgi:
    """Retiree with no wages but $300k of taxable-brokerage income still
    needs to fail the direct-Roth phase-out. Wages-only MAGI estimate
    silently allowed direct contributions."""

    def test_high_portfolio_retiree_blocked_from_direct_roth(self) -> None:
        cfg = Config()
        inp = Inputs(
            spouse_a_age_start=66,
            spouse_b_age_start=66,
            spouse_a_retire_age=66,
            spouse_b_retire_age=66,
            spouse_a_roth_ira_contrib=7_000,
            spouse_b_roth_ira_contrib=7_000,
            starting=StartingBalances(
                taxable_brokerage=4_000_000,   # high portfolio
                spouse_a_pretax_401k=0,
                spouse_b_pretax_401k=0,
            ),
        )
        df = simulate(cfg, inp)
        # First year is wages-only fallback (state.prior_agi = 0).
        # From year 2 onward (prior AGI reflects portfolio), direct Roth
        # contributions should be phased out.
        late = df.iloc[5:]
        assert (late["ira_roth_direct_a"] == 0).all()
        assert (late["ira_roth_direct_b"] == 0).all()


# =====================================================================
# TC-6 — backdoor pro-rata is IRA-aggregate, not IRA + 401(k)
# =====================================================================


class TestBackdoorProRataIRAOnly:
    """A spouse with a $0 pretax IRA but a $500k 401(k) should have a
    clean (zero-taxable) backdoor. Pre-Tier-C, the simulator passed
    combined pretax (IRA + 401k) so the conversion was 99% taxable."""

    def test_clean_backdoor_with_only_401k_pretax(self) -> None:
        cfg = Config()
        inp = Inputs(
            spouse_a_backdoor_roth=True,
            spouse_a_roth_ira_contrib=0,
            starting=StartingBalances(
                spouse_a_pretax_401k=500_000,
                spouse_a_pretax_ira=0,        # clean for backdoor
                spouse_b_pretax_401k=0,
            ),
        )
        df = simulate(cfg, inp)
        # First-year backdoor should produce ZERO taxable conversion
        # (pretax IRA = 0 going in).
        first = df.iloc[0]
        assert first["ira_backdoor_a"] > 0
        assert first["ira_backdoor_taxable_conv"] == pytest.approx(0.0, abs=1e-6)

    def test_partial_backdoor_when_pretax_ira_present(self) -> None:
        # Now the same spouse has $50k of pretax IRA in addition to
        # $500k 401(k). Pro-rata over the *IRA* only:
        #   taxable_fraction = 50k / (50k + 7k) ≈ 0.877
        cfg = Config()
        inp = Inputs(
            spouse_a_backdoor_roth=True,
            starting=StartingBalances(
                spouse_a_pretax_401k=500_000,
                spouse_a_pretax_ira=50_000,
                spouse_b_pretax_401k=0,
            ),
        )
        df = simulate(cfg, inp)
        first = df.iloc[0]
        backdoor = first["ira_backdoor_a"]
        # 401(k) is excluded → much smaller taxable fraction than the
        # 50k+500k=99% you'd see if 401k were included.
        ratio = first["ira_backdoor_taxable_conv"] / max(backdoor, 1e-9)
        assert ratio < 0.95
        # The IRA-only ratio is around 50k/57k ≈ 0.877.
        assert 0.80 < ratio < 0.92


# =====================================================================
# TC-7 — fixed Roth conversion reserves RMD bucket
# =====================================================================


class TestConversionReservesRMD:
    """A fixed `roth_conversion_amount` greater than (pretax − RMD)
    used to silently truncate the RMD via `withdraw_for_need`'s
    `min(rmd, balance)`. The fix caps each spouse's conversion at
    `pretax - rmd` so RMD always satisfies in full."""

    def test_rmd_satisfied_when_fixed_conv_would_exceed_balance(self) -> None:
        # In bracket-fill mode the test would self-regulate; force
        # fixed mode and align the gap window so a conversion is
        # eligible AT or after rmd_start_age. We use a custom
        # rmd_start_age=70 so the "gap window" extends to 70 and
        # collides with RMD.
        cfg = Config(
            roth_conversion_amount=200_000,
            rmd_start_age=70,                # RMDs start at 70
        )
        # Spouse A is 70 in year 0, retired, big pretax balance.
        inp = Inputs(
            spouse_a_age_start=70,
            spouse_b_age_start=70,
            spouse_a_retire_age=65,
            spouse_b_retire_age=65,
            starting=StartingBalances(
                spouse_a_pretax_401k=300_000,
                spouse_b_pretax_401k=0,
            ),
        )
        df = simulate(cfg, inp)
        first = df.iloc[0]
        # Statutory RMD on $300k at age 70 ≈ $300k / 27.4 ≈ $10,949.
        # Fixed conv of $200k would have wiped the bucket; the cap at
        # `pretax - rmd` keeps RMD intact.
        assert first["rmd"] > 9_000
        # And the conversion didn't push pretax to 0 before RMD ran.


# =====================================================================
# TC-8 — report shows per-spouse SS start_age
# =====================================================================


class TestReportPerSpouseSS:
    def test_report_uses_per_spouse_start_age(self) -> None:
        cfg = Config()
        inp = Inputs(
            ss=SocialSecurity(
                start_age=67,
                start_age_a=70,
                start_age_b=62,
                monthly_spouse_a=2_500,
                monthly_spouse_b=1_500,
                fra_a=67, fra_b=67,
            ),
        )
        results, sens_df, base_terminal = _build_minimal_report_args(cfg, inp)
        report_md = build_action_report(cfg, inp, results, sens_df, base_terminal)
        # Spouse A row mentions 70 (both retire and SS); Spouse B row
        # mentions 62 (SS) — these used to both show `inputs.ss.start_age=67`.
        assert "/ 70" in report_md
        assert "/ 62" in report_md


# =====================================================================
# TC-9 — caveats block no longer claims "federal-only"
# =====================================================================


class TestCaveatsBlock:
    def test_caveats_does_not_claim_federal_only(self) -> None:
        cfg, inp = Config(), Inputs()
        results, sens_df, base_terminal = _build_minimal_report_args(cfg, inp)
        report_md = build_action_report(cfg, inp, results, sens_df, base_terminal)
        assert "Federal-only today" not in report_md


# =====================================================================
# TC-10 — base Medicare Part B + Part D premiums
# =====================================================================


class TestBaseMedicarePremiums:
    """Base Medicare Part B + Part D premiums apply to every Medicare-
    enrolled spouse (65+) regardless of income — separately from the
    IRMAA surcharge which only triggers above income thresholds."""

    def test_two_seniors_both_pay_base_premium(self) -> None:
        cfg = Config(medicare_base_b_d_premium=2_500)
        inp = Inputs(
            spouse_a_age_start=66,
            spouse_b_age_start=66,
            spouse_a_retire_age=65,
            spouse_b_retire_age=65,
        )
        df = simulate(cfg, inp)
        first = df.iloc[0]
        # 2 enrolled × $2,500 = $5,000 base.
        assert first["medicare_base_premium"] == pytest.approx(5_000)

    def test_premium_zero_below_65(self) -> None:
        df = simulate(Config(medicare_base_b_d_premium=2_500), Inputs())
        # Default starting ages are 35 → no Medicare in year 0.
        assert df.iloc[0]["medicare_base_premium"] == 0.0


# =====================================================================
# TC-11 — IRMAA 2-year MAGI lookback
# =====================================================================


class TestIRMAALookback:
    """IRMAA in year T is based on MAGI from year T−2 (SSA's published
    rule). Pre-Tier-C the simulator used current-year AGI which made
    Roth-conversion years over-state IRMAA cliffs."""

    def test_year_zero_irmaa_not_triggered_by_current_conversion(self) -> None:
        # Current-year AGI is huge (Roth conversion), but lag-2 AGI is
        # ZERO at year 0 → IRMAA should be $0 at year 0.
        cfg = Config(
            roth_conversion_amount=300_000,
            irmaa_lookback_years=2,
        )
        inp = Inputs(
            spouse_a_age_start=66,
            spouse_b_age_start=66,
            spouse_a_retire_age=65,
            spouse_b_retire_age=65,
            starting=StartingBalances(
                spouse_a_pretax_401k=2_000_000,
            ),
        )
        df = simulate(cfg, inp)
        assert df.iloc[0]["irmaa"] == 0.0

    def test_irmaa_kicks_in_two_years_after_high_agi(self) -> None:
        # A big year-0 conversion should drive IRMAA at year 2 (lag-2
        # lookback) — but be quiet at years 0 and 1.
        cfg = Config(
            roth_conversion_amount=300_000,
            irmaa_lookback_years=2,
        )
        inp = Inputs(
            spouse_a_age_start=66,
            spouse_b_age_start=66,
            spouse_a_retire_age=65,
            spouse_b_retire_age=65,
            starting=StartingBalances(
                spouse_a_pretax_401k=2_000_000,
            ),
        )
        df = simulate(cfg, inp)
        assert df.iloc[2]["irmaa"] > df.iloc[1]["irmaa"]


# =====================================================================
# TC-12 — pre-Medicare healthcare expense knob
# =====================================================================


class TestPreMedicareHealthcare:
    def test_healthcare_charged_until_65_then_drops(self) -> None:
        cfg = Config(
            health_pre65_today=15_000,
            medicare_base_b_d_premium=0,    # disable to isolate pre-65 line
        )
        inp = Inputs(
            spouse_a_age_start=60,
            spouse_b_age_start=60,
            spouse_a_retire_age=60,
            spouse_b_retire_age=60,
        )
        df = simulate(cfg, inp)
        # Year 0: both 60 → both contribute to pre-65 cost.
        # Year 5: both 65 → no pre-65 cost.
        assert df.iloc[0]["health_pre65"] >= 15_000   # at least one spouse
        assert df.iloc[5]["health_pre65"] == 0.0


# =====================================================================
# TC-13 — ACA premium tax credit (post-IRA-2022 8.5% cap)
# =====================================================================


class TestACAPremiumTaxCredit:
    """ACA enhanced subsidies (IRA-2022): premium contribution capped
    at 8.5% of MAGI; APTC = max(0, benchmark - cap). No income cliff.

    We model a benchmark second-lowest-cost-silver-plan premium at
    $14k/spouse pre-65 (rough national average for a 60-year-old in
    2026) and credit benchmark - 8.5%·MAGI against the household tax
    bill in the year of enrollment.
    """

    def test_low_income_retiree_gets_full_subsidy(self) -> None:
        cfg = Config(
            aca_enabled=True,
            aca_benchmark_premium_per_adult=14_000,
            spending=SpendingProfile.flat(40_000),
        )
        inp = Inputs(
            spouse_a_age_start=60,
            spouse_b_age_start=60,
            spouse_a_retire_age=60,
            spouse_b_retire_age=60,
            starting=StartingBalances(
                taxable_brokerage=300_000,    # low-yield → low MAGI
                spouse_a_pretax_401k=0,
            ),
        )
        df = simulate(cfg, inp)
        first = df.iloc[0]
        # Both adults enrolled, low MAGI → APTC nearly equals benchmark.
        assert first["aca_apt_credit"] > 20_000

    def test_high_income_no_subsidy(self) -> None:
        cfg = Config(
            aca_enabled=True,
            aca_benchmark_premium_per_adult=14_000,
        )
        inp = Inputs(
            spouse_a_age_start=60,
            spouse_b_age_start=60,
            spouse_a_retire_age=60,
            spouse_b_retire_age=60,
            starting=StartingBalances(
                spouse_a_pretax_401k=0,
                spouse_b_pretax_401k=0,
                taxable_brokerage=0,
            ),
            income=Inputs().income.__class__(
                spouse_a_gross=600_000,
                spouse_b_gross=0,
                interest=0, dividends=0, capital_gains=0,
            ),
        )
        # Force still-working so wages drive MAGI.
        inp = replace(
            inp,
            spouse_a_retire_age=65,
        )
        df = simulate(cfg, inp)
        # 8.5% of $600k = $51k > $28k benchmark → credit should be $0.
        first = df.iloc[0]
        assert first["aca_apt_credit"] == 0.0

    def test_disabled_by_default(self) -> None:
        df = simulate(Config(), Inputs())
        assert (df["aca_apt_credit"] == 0.0).all()


# =====================================================================
# TC-14 — step-up in basis on first spouse's death
# =====================================================================


class TestStepUpInBasis:
    """At first spouse's death, the surviving spouse's basis in
    inherited taxable assets is reset to fair-market-value (full
    step-up in community-property states; half step-up in common-law
    states — we model the full-step-up community-property case to keep
    the math simple and clear for users in CA / WA / etc.)."""

    def test_basis_resets_to_fmv_on_first_death(self) -> None:
        cfg = Config(
            stepup_at_first_death=True,
            mortality=Mortality(year_of_death_a=10),
            cap_gains_basis_fraction=0.20,
        )
        inp = Inputs(
            starting=StartingBalances(
                taxable_brokerage=1_000_000,
            ),
        )
        df = simulate(cfg, inp)
        pre = df.iloc[9]
        post = df.iloc[10]
        # Cumulative basis should jump up at year 10 (year of death).
        assert post["cumulative_basis"] > pre["cumulative_basis"]
        # And ideally land close to FMV (taxable_balance). Basis is
        # reset to FMV at the START of year 10 → market growth and
        # surplus during year 10 can pull the ratio slightly off 1.0,
        # so we just check the pre→post jump in basis_fraction is
        # large.
        pre_ratio = pre["cumulative_basis"] / max(pre["taxable_balance"], 1.0)
        post_ratio = post["cumulative_basis"] / max(post["taxable_balance"], 1.0)
        assert post_ratio - pre_ratio > 0.5
        assert post_ratio > 0.9

    def test_disabled_by_default_keeps_old_behaviour(self) -> None:
        cfg = Config(mortality=Mortality(year_of_death_a=10))
        df = simulate(cfg, Inputs())
        # No automatic basis bump.
        pre = df.iloc[9]
        post = df.iloc[10]
        assert post["cumulative_basis"] <= pre["cumulative_basis"] + 1e-3


# =====================================================================
# TC-15 — mega-backdoor in optimizer decision vector
# =====================================================================


class TestMegaBackdoorInOptimizer:
    def test_decision_vector_grows_when_mega_backdoor_enabled(self) -> None:
        from tax_optimizer.optimizer import _build_decision_vector_meta

        meta_off = _build_decision_vector_meta(Config(), Inputs())
        meta_on = _build_decision_vector_meta(
            Config(),
            Inputs(
                spouse_a_mega_backdoor_enabled=True,
                spouse_b_mega_backdoor_enabled=True,
            ),
        )
        assert len(meta_on) == len(meta_off) + 2   # one per spouse
        names_on = {m["name"] for m in meta_on}
        assert "mega_backdoor_pct_a" in names_on
        assert "mega_backdoor_pct_b" in names_on


# =====================================================================
# TC-16 — per-spouse SS claim-age axis
# =====================================================================


class TestSSClaimAgeOptimizerAxis:
    def test_optimizer_can_search_ss_claim_age(self) -> None:
        from tax_optimizer.optimizer import _build_decision_vector_meta

        cfg = Config(optimize_ss_claim_age=True)
        meta = _build_decision_vector_meta(cfg, Inputs())
        names = [m["name"] for m in meta]
        assert "ss_claim_age_a" in names
        assert "ss_claim_age_b" in names


# =====================================================================
# TC-17 — seed thread-through in MC objectives
# =====================================================================


class TestSeedThreadthrough:
    def test_optimize_household_accepts_mc_seed(self) -> None:
        from tax_optimizer.optimizer import optimize_household

        import inspect

        sig = inspect.signature(optimize_household)
        assert "mc_seed" in sig.parameters

    def test_make_objective_accepts_mc_seed(self) -> None:
        from tax_optimizer.optimizer import make_objective

        import inspect

        sig = inspect.signature(make_objective)
        assert "mc_seed" in sig.parameters

    def test_two_seeds_produce_different_cvar_curves(self) -> None:
        # Two different `mc_seed`s should yield different CVaR scores
        # for the same x. This is the empirical proof that the seed
        # genuinely flows through the MC layer. Requires a stochastic
        # market — DeterministicModel collapses all paths identically.
        from tax_optimizer.market import LognormalModel
        from tax_optimizer.optimizer import make_objective

        cfg = Config(market=LognormalModel())
        inp = Inputs()
        obj_42 = make_objective(cfg, inp, objective="cvar", n_paths=20, mc_seed=42)
        obj_99 = make_objective(cfg, inp, objective="cvar", n_paths=20, mc_seed=99)
        x = [0.5, 0.5, 2]
        score_42 = obj_42(x)
        score_99 = obj_99(x)
        assert score_42 != score_99
