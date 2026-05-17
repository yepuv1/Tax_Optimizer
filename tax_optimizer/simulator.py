"""Year-by-year simulator.

This is a single-path simulator. The Monte Carlo wrapper in
`monte_carlo.py` runs this many times with different RNG seeds.

Compared to the v1 simulator, this one threads:
  * `cfg.regime_for_year(year_offset)`  - active TaxRegime per year
  * `cfg.mortality.filing_status(...)`  - 'mfj' or 'single' per year
  * `cfg.resolved_market()`             - per-year (equity, bond) returns
  * `cfg.asset_location`                - account-level equity %
  * `cfg.resolved_spending()`           - smile + lump events + LTC
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .annuity import exclusion_ratio
from .config import Config
from .conversion import planned_roth_conversion
from .inputs import Inputs, claim_age_factor
from .ira import allocate_ira_contributions
from .limits import ELECTIVE_DEFERRAL_LIMIT, SECTION_415C_LIMIT, elective_deferral_cap, hsa_family_cap
from .payroll import OASDI_WAGE_BASE_2026, fica_household, state_sdi
from .pension import (
    PENSION_INTEREST,
    PENSION_QTR_SSWB,
    effective_interest_rate,
    pension_annual_credit,
    project_pension_balance,
)
from .rmd import rmd_amount
from .state import initial_state
from .tax.federal import federal_tax
from .tax.irmaa import MEDICARE_ELIGIBLE_AGE, irmaa_annual_surcharge
from .tax.state import state_tax
from .withdrawals import cover_deficit, withdraw_for_need


def simulate(
    cfg: Config,
    inputs: Inputs | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Run one path of the simulator.

    `rng` is only used when `cfg.market` is stochastic. For deterministic
    runs, a default RNG is created but its draws are ignored.
    """
    if inputs is None:
        inputs = Inputs()
    if rng is None:
        rng = np.random.default_rng(0)

    state = initial_state(cfg, inputs)
    rows: list[dict] = []

    spouse_a_salary = inputs.income.spouse_a_gross
    spouse_b_salary = inputs.income.spouse_b_gross
    spouse_a_bonus = inputs.income.spouse_a_bonus

    n_years = cfg.horizon_age - inputs.spouse_a_age_start + 1

    # Track alive-state across years so we can detect the death
    # transition and move the deceased spouse's pretax IRA / 401(k) into
    # the surviving spouse's balance (a spousal rollover under IRC §401
    # / §408). This is what makes the survivor's RMDs continue on the
    # combined balance instead of the deceased's pretax sitting frozen
    # forever — a real-world correctness bug otherwise.
    #
    # For a single-filer household (no spouse B from year 0), seed
    # `prev_alive_b = False` so the rollover-on-death detector
    # (`prev_alive_b and not alive_b and alive_a`) doesn't spuriously
    # fire in year 0 against an empty spouse-B balance.
    prev_alive_a = True
    prev_alive_b = inputs.household_kind != "single"

    market = cfg.resolved_market()
    market.begin_path(n_years, rng)

    spending = cfg.resolved_spending()
    asset_loc = cfg.asset_location

    def _drain_pretax_ira(spouse: str, total_pretax_pre_drain: float, drain: float) -> None:
        """Drain the IRA-only sub-balance pro-rata to its share of the
        total pretax bucket (TC-6 bookkeeping). Used after RMDs,
        Roth conversions, ordinary withdrawals, and deficit-cascade
        pretax draws. In real life the user picks WHICH IRA / 401(k)
        each draw comes from; we average it over the whole pretax
        bucket which is approximately right.
        """
        if drain <= 0 or total_pretax_pre_drain <= 0:
            return
        ira_attr = f"spouse_{spouse}_pretax_ira"
        ira_balance = getattr(state, ira_attr)
        if ira_balance <= 0:
            return
        ira_share = ira_balance / total_pretax_pre_drain
        setattr(state, ira_attr, max(0.0, ira_balance - drain * ira_share))

    # Reference projected pension balance at NRD; used to scale annuity
    # if the actual cash-balance grew faster (or slower) than expected.
    # Pass `retire_age` so the projector stops pay credits at
    # retirement (the simulator does too — without this knob the
    # projector overstates by 4-15% for users who retire before NRD,
    # which scales the user's `monthly_at_nrd` input DOWN by the same
    # margin in the simulator's annuity calculation).
    test_balance = project_pension_balance(
        inputs.pension.balance_today,
        inputs.income.spouse_a_gross + inputs.income.spouse_a_bonus,
        max(0, inputs.pension.start_age - inputs.spouse_a_age_start),
        wage_growth=cfg.wage_growth,
        start_age=inputs.spouse_a_age_start,
        years_of_service_today=inputs.pension.years_of_service_today,
        interest_rate=inputs.pension.interest_rate,
        pre_2016_participant=inputs.pension.pre_2016_participant,
        comp_limit_today=inputs.pension.irs_comp_limit_today,
        retire_age=inputs.spouse_a_retire_age,
    )

    for year_offset in range(n_years):
        year = cfg.start_year + year_offset
        a_age = inputs.spouse_a_age_start + year_offset
        b_age = inputs.spouse_b_age_start + year_offset
        state.spouse_a_age = a_age
        state.spouse_b_age = b_age
        state.year = year

        # Mortality + filing status for this year.
        alive_a = cfg.mortality.alive_a(year_offset)
        alive_b = cfg.mortality.alive_b(year_offset)
        filing_status = cfg.mortality.filing_status(year_offset)

        # Single-filer household override. ``inputs.household_kind ==
        # "single"`` means there is no spouse B from year 0 — not a
        # widowhood transition. We force ``alive_b = False`` (so every
        # downstream "if alive_b" branch correctly skips B's salary,
        # contributions, RMDs, SS, etc.) and ``filing_status =
        # "single"`` (so we bypass the year-of-death MFJ exception in
        # `Mortality.filing_status` — a never-married single filer
        # never gets an MFJ year). Done here once at the top of the
        # year loop so every downstream tax / FICA / IRMAA / SS code
        # path automatically uses the Single tables already present in
        # the regime.
        if inputs.household_kind == "single":
            alive_b = False
            filing_status = "single"

        # Spousal IRA rollover on death transition. When a spouse dies
        # the survivor inherits the IRA / 401(k); IRS rules let them
        # roll it into their own retirement account, after which RMDs
        # are computed on the survivor's age against the combined
        # balance. Without this, a deceased spouse's pretax just sits
        # frozen (still growing tax-deferred but never RMD'd) which
        # wildly under-counts lifetime tax. We trigger the rollover
        # once on the first year `alive_*` flips False, but only when
        # the OTHER spouse is alive (otherwise the balance falls to
        # non-spousal heirs which is out of scope).
        rollover_event = ""
        # ---- Step-up in basis (TC-14) ----
        # Triggered the year a spouse first transitions from alive→dead
        # (community-property full step-up assumed). Reset the surviving
        # spouse's `cumulative_basis` to FMV (= state.taxable) so future
        # withdrawals from the inherited taxable account realize $0
        # gain on the inherited share. Common-law half-step-up is a
        # blind-spot we'd cover in Tier D.
        first_death_this_year = (
            (prev_alive_a and not alive_a and alive_b)
            or (prev_alive_b and not alive_b and alive_a)
        )
        if cfg.stepup_at_first_death and first_death_this_year and state.taxable > 0:
            state.cumulative_basis = state.taxable

        if (
            prev_alive_a
            and not alive_a
            and alive_b
            and state.spouse_a_pretax > 0
        ):
            state.spouse_b_pretax += state.spouse_a_pretax
            state.spouse_b_pretax_ira += state.spouse_a_pretax_ira
            state.spouse_a_pretax = 0.0
            state.spouse_a_pretax_ira = 0.0
            rollover_event = "a_to_b"
        if (
            prev_alive_b
            and not alive_b
            and alive_a
            and state.spouse_b_pretax > 0
        ):
            state.spouse_a_pretax += state.spouse_b_pretax
            state.spouse_a_pretax_ira += state.spouse_b_pretax_ira
            state.spouse_b_pretax = 0.0
            state.spouse_b_pretax_ira = 0.0
            rollover_event = "b_to_a"

        regime = cfg.effective_regime(year_offset)
        state_regime = cfg.effective_state_regime(year_offset)

        a_working = alive_a and a_age < inputs.spouse_a_retire_age
        b_working = alive_b and b_age < inputs.spouse_b_retire_age

        # ---- Contributions (only while alive AND working) ----
        # IRS-cap each spouse's elective deferral (with age-50+ catch-up).
        # Without this guard, a user setting `spouse_a_total_contrib_pct
        # = 0.30` on a $300k salary would silently route $90k into a
        # vehicle whose real-world cap is ~$31k.
        a_target = (
            spouse_a_salary * inputs.spouse_a_total_contrib_pct if a_working else 0.0
        )
        b_target = (
            spouse_b_salary * inputs.spouse_b_total_contrib_pct if b_working else 0.0
        )
        a_total_contrib = min(a_target, elective_deferral_cap(a_age)) if a_working else 0.0
        b_total_contrib = min(b_target, elective_deferral_cap(b_age)) if b_working else 0.0
        a_pretax_contrib = a_total_contrib * (1 - inputs.spouse_a_roth_401k_pct)
        a_roth_contrib = a_total_contrib * inputs.spouse_a_roth_401k_pct
        b_pretax_contrib = b_total_contrib * (1 - inputs.spouse_b_roth_401k_pct)
        b_roth_contrib = b_total_contrib * inputs.spouse_b_roth_401k_pct

        # Employer 401(k) match. Pays in pre-tax (per IRS rules) on top
        # of the employee's elective deferral, regardless of whether
        # the employee chose Roth. Sized on the actually-deferred
        # percentage (already capped by elective_deferral_cap above)
        # rather than the user's pre-cap target.
        a_effective_pct = (
            (a_total_contrib / spouse_a_salary) if (a_working and spouse_a_salary > 0) else 0.0
        )
        b_effective_pct = (
            (b_total_contrib / spouse_b_salary) if (b_working and spouse_b_salary > 0) else 0.0
        )
        a_employer_match = (
            spouse_a_salary
            * min(a_effective_pct, inputs.spouse_a_employer_match_max_pct)
            * inputs.spouse_a_employer_match_rate
            if a_working else 0.0
        )
        b_employer_match = (
            spouse_b_salary
            * min(b_effective_pct, inputs.spouse_b_employer_match_max_pct)
            * inputs.spouse_b_employer_match_rate
            if b_working else 0.0
        )

        state.spouse_a_pretax += a_pretax_contrib + a_employer_match
        state.spouse_b_pretax += b_pretax_contrib + b_employer_match
        state.roth += a_roth_contrib + b_roth_contrib

        # Mega-backdoor Roth (after-tax 401(k) → in-plan Roth conversion).
        # Available room each year is §415(c) overall cap minus
        # employee elective deferrals and employer match; catch-up
        # contributions sit *outside* §415(c) per IRS so we use the
        # raw deferral total (not deferral + catch-up) here. After-tax
        # dollars come from after-tax paycheck cash → Roth balance,
        # with no income-tax impact.
        #
        # v6.6 auto-spillover: when `*_mega_backdoor_enabled` is True,
        # any **excess elective-deferral target** (target above the
        # §402(g) cap that pre-v6.6 silently leaked to taxable cash)
        # is auto-routed into the after-tax bucket first, up to the
        # §415(c) ceiling. The explicit `*_after_tax_401k_pct` then
        # stacks on top, capped at whatever §415(c) room remains.
        # Real-world analog: Vanguard "Spillover After-Tax" and
        # similar Fidelity / Schwab record-keeper features.
        a_excess_deferral = (
            max(0.0, a_target - elective_deferral_cap(a_age)) if a_working else 0.0
        )
        b_excess_deferral = (
            max(0.0, b_target - elective_deferral_cap(b_age)) if b_working else 0.0
        )
        # §415(c) room: catch-up is excluded from the §415(c) limit per
        # IRS rules, so subtract only the base elective deferral here.
        a_base_deferral = min(a_target, ELECTIVE_DEFERRAL_LIMIT) if a_working else 0.0
        b_base_deferral = min(b_target, ELECTIVE_DEFERRAL_LIMIT) if b_working else 0.0
        a_after_tax_room = max(
            0.0,
            SECTION_415C_LIMIT - a_base_deferral - a_employer_match,
        )
        b_after_tax_room = max(
            0.0,
            SECTION_415C_LIMIT - b_base_deferral - b_employer_match,
        )
        # Auto-spillover (consumes §415(c) room first).
        a_after_tax_auto = (
            min(a_excess_deferral, a_after_tax_room)
            if (a_working and inputs.spouse_a_mega_backdoor_enabled)
            else 0.0
        )
        b_after_tax_auto = (
            min(b_excess_deferral, b_after_tax_room)
            if (b_working and inputs.spouse_b_mega_backdoor_enabled)
            else 0.0
        )
        a_room_left = max(0.0, a_after_tax_room - a_after_tax_auto)
        b_room_left = max(0.0, b_after_tax_room - b_after_tax_auto)
        # Explicit after-tax pct (existing knob), capped at the room
        # REMAINING after the auto-spillover. Any uncovered target
        # dollars stay as taxable cash (same fate as elective-deferral
        # excess above the §415(c) ceiling).
        a_after_tax_pct = (
            inputs.spouse_a_after_tax_401k_pct
            if (a_working and inputs.spouse_a_mega_backdoor_enabled)
            else 0.0
        )
        b_after_tax_pct = (
            inputs.spouse_b_after_tax_401k_pct
            if (b_working and inputs.spouse_b_mega_backdoor_enabled)
            else 0.0
        )
        a_after_tax_explicit_target = (
            spouse_a_salary * a_after_tax_pct if a_working else 0.0
        )
        b_after_tax_explicit_target = (
            spouse_b_salary * b_after_tax_pct if b_working else 0.0
        )
        a_after_tax_explicit = min(a_after_tax_explicit_target, a_room_left)
        b_after_tax_explicit = min(b_after_tax_explicit_target, b_room_left)
        a_after_tax_target_uncovered = max(
            0.0, a_after_tax_explicit_target - a_after_tax_explicit
        )
        b_after_tax_target_uncovered = max(
            0.0, b_after_tax_explicit_target - b_after_tax_explicit
        )
        a_after_tax = a_after_tax_auto + a_after_tax_explicit
        b_after_tax = b_after_tax_auto + b_after_tax_explicit
        state.roth += a_after_tax + b_after_tax
        mega_backdoor_total = a_after_tax + b_after_tax

        # HSA family contribution: capped at IRS family limit + 55+
        # catch-up, only valid while at least one spouse works AND is
        # Medicare-ineligible. Pre-tax: reduces wages_box1 below.
        hsa_cap = hsa_family_cap(
            a_age,
            b_age,
            either_working=(a_working or b_working),
            b_alive=alive_b,
        )
        hsa_contrib = min(max(0.0, inputs.contrib.hsa_family), hsa_cap)
        state.hsa += hsa_contrib

        # IRA contributions (Traditional / direct Roth / backdoor Roth).
        # Eligibility: alive AND either spouse working (spousal IRA
        # rule). MAGI for the phase-out: prior-year AGI is the best
        # forward-looking estimate we have at this point in the loop
        # (current-year AGI isn't computed yet; current-year wages alone
        # would miss interest, dividends, SS, pension, and capital
        # gains, which can push a high-portfolio retiree across the
        # ~$246k MFJ direct-Roth phase-out and silently keep direct
        # contributions in scope when they shouldn't be — TC-5).
        wages_estimate = (
            (spouse_a_salary + spouse_a_bonus if a_working else 0.0)
            + (spouse_b_salary if b_working else 0.0)
        )
        magi_estimate = max(state.prior_agi, wages_estimate)
        ira_eligible_either = (a_working or b_working)
        ira_a = allocate_ira_contributions(
            age=a_age,
            eligible=alive_a and ira_eligible_either,
            pretax_existing=state.spouse_a_pretax_ira,
            traditional_target=inputs.spouse_a_traditional_ira_contrib,
            roth_direct_target=inputs.spouse_a_roth_ira_contrib,
            backdoor_enabled=inputs.spouse_a_backdoor_roth,
            magi_estimate=magi_estimate,
            filing_status=filing_status,
        )
        ira_b = allocate_ira_contributions(
            age=b_age,
            eligible=alive_b and ira_eligible_either,
            pretax_existing=state.spouse_b_pretax_ira,
            traditional_target=inputs.spouse_b_traditional_ira_contrib,
            roth_direct_target=inputs.spouse_b_roth_ira_contrib,
            backdoor_enabled=inputs.spouse_b_backdoor_roth,
            magi_estimate=magi_estimate,
            filing_status=filing_status,
        )

        # Apply IRA contributions to balances. Cash outflow (subtracted
        # from working-year cash_inflow below) is the sum of all three
        # paths — the Traditional deduction comes back as a smaller
        # federal tax bill, not as cash that didn't leave the household.
        # Traditional IRA contributions land in BOTH the combined
        # `spouse_*_pretax` bucket and the IRA-only sub-balance used by
        # backdoor pro-rata math (TC-6).
        state.spouse_a_pretax += ira_a.traditional
        state.spouse_b_pretax += ira_b.traditional
        state.spouse_a_pretax_ira += ira_a.traditional
        state.spouse_b_pretax_ira += ira_b.traditional
        state.roth += ira_a.roth_direct + ira_b.roth_direct
        # Backdoor: contribute to non-deductible Traditional → convert
        # in same year. Net pretax balance change is zero (in then out);
        # the taxable-fraction lands as ordinary income via roth_conversion.
        state.roth += ira_a.backdoor + ira_b.backdoor

        ira_total_outflow = ira_a.total_cash_outflow + ira_b.total_cash_outflow
        ira_traditional_deduction = ira_a.traditional + ira_b.traditional
        ira_backdoor_taxable_conv = (
            ira_a.backdoor_taxable_conversion + ira_b.backdoor_taxable_conversion
        )

        # ---- Income items ----
        a_w2_wages = (spouse_a_salary + spouse_a_bonus) if a_working else 0.0
        b_w2_wages = spouse_b_salary if b_working else 0.0
        wages = a_w2_wages + b_w2_wages

        # ---- §125 cafeteria-plan deductions (v6.6) ----
        # Medical / dental / vision premiums are paid through a §125
        # cafeteria plan: they reduce Box 1 federal wages, Box 3/5
        # FICA wages, and state wages. HSA (already accounted for
        # in wages_box1 below) is also §125; we fold it into the
        # FICA reduction here when `section125_reduces_fica_wages`
        # is on. Per-spouse premiums are gated on each spouse's own
        # working status (no W-2 → no §125 deduction) and clamped
        # at that spouse's gross wages.
        hp = inputs.health_premiums
        hp_a = min(max(0.0, hp.total_a), a_w2_wages) if a_working else 0.0
        hp_b = min(max(0.0, hp.total_b), b_w2_wages) if b_working else 0.0
        hp_total = hp_a + hp_b

        # HSA is family-coverage in our model. Allocate it across
        # spouses for FICA-base purposes by wage-share so OASDI's
        # per-spouse wage cap is handled correctly when one spouse
        # is near the cap. The total FICA reduction is invariant to
        # the allocation when both spouses are well under the cap.
        #
        # When `section125_reduces_fica_wages` is False (the pre-v6.6
        # back-compat path), FICA uses GROSS W-2 wages — premiums and
        # HSA both ignored for FICA purposes (still reduce Box 1).
        if cfg.section125_reduces_fica_wages and wages > 0:
            hsa_a_share = hsa_contrib * (a_w2_wages / wages)
            hsa_b_share = hsa_contrib - hsa_a_share
            a_w2_fica = max(0.0, a_w2_wages - hp_a - hsa_a_share)
            b_w2_fica = max(0.0, b_w2_wages - hp_b - hsa_b_share)
        else:
            a_w2_fica = a_w2_wages
            b_w2_fica = b_w2_wages

        # Traditional 401(k) and deductible Traditional IRA reduce
        # Box 1 but NOT FICA. HSA and M/D/V (§125) reduce both —
        # already netted out of Box 1 here. Clamp at 0 to handle
        # edge cases where total pre-tax deductions exceed wages.
        wages_box1 = max(0.0, (
            wages
            - a_pretax_contrib - b_pretax_contrib
            - hsa_contrib
            - hp_total
            - ira_traditional_deduction
        ))

        # FICA on per-spouse W-2 wages. Applies to GROSS wages (Box 3) —
        # traditional 401(k) does NOT reduce FICA wages, but §125
        # cafeteria deductions (HSA + M/D/V) DO when
        # `cfg.section125_reduces_fica_wages` is on (default since
        # v6.6). Each spouse has their own OASDI wage base; the wage
        # base is indexed forward annually so real FICA stays roughly
        # constant. FICA does not touch federal income tax — it just
        # removes cash from the household's working-year inflow.
        #
        # Additional Medicare 0.9% is reconciled at the **household**
        # level (Form 8959): MFJ threshold is $250k on combined wages,
        # not $200k per W-2. `fica_household` handles that; per-W-2
        # OASDI and base Medicare still come out of `fica_employee`
        # internally.
        wage_base_y = OASDI_WAGE_BASE_2026 * (1 + cfg.wage_growth) ** year_offset
        fica_combined = fica_household(
            a_w2_fica, b_w2_fica,
            filing_status=filing_status,
            wage_base=wage_base_y,
        )
        fica_total = fica_combined["total"]

        # State disability insurance (CA SDI / NJ TDI / etc.). Per-spouse
        # rate × wages, indexed to the active state regime. Inflation-
        # indexed wage cap matches FICA's so CA's uncapped rule stays
        # truly uncapped over the horizon. §125 cafeteria deductions
        # reduce SDI wages in CA (FTB conforms); we use the same
        # post-§125 base as FICA.
        sdi_wage_cap_y = (
            state_regime.sdi_wage_cap
            if not math.isfinite(state_regime.sdi_wage_cap)
            else state_regime.sdi_wage_cap * (1 + cfg.inflation) ** year_offset
        )
        # Year-indexed SDI rate (CA: 1.1% / 1.2% / 0.9% for
        # 2024/25/26). Falls back to `state_regime.sdi_rate` for any
        # regime without a published schedule.
        sdi_rate_y = state_regime.effective_sdi_rate(
            cfg.start_year + year_offset
        )
        sdi_a = state_sdi(
            a_w2_fica, rate=sdi_rate_y, wage_cap=sdi_wage_cap_y
        )
        sdi_b = state_sdi(
            b_w2_fica, rate=sdi_rate_y, wage_cap=sdi_wage_cap_y
        )
        sdi_total = sdi_a + sdi_b

        # ---- Pension lump-sum gate (fires once at NRD) -----------
        # If the user elected `lump_sum_mode != "none"`, the entire
        # pension cash-balance is liquidated AT `start_age` instead
        # of starting the monthly annuity. Two flavors:
        #
        #   * "rollover_pretax" — direct rollover into the
        #     participant's pretax IRA (IRC §402(c)). No current-
        #     year tax. Future RMDs / ordinary withdrawals apply
        #     automatically via the existing pretax pipeline.
        #   * "cash" — full balance distributed as ordinary income
        #     in the start_age year. Adds to `pension_income` for
        #     this single year so the income flows through the
        #     standard tax + cash-surplus path. Below, we also
        #     inject the 10% IRC §72(t) additional tax via
        #     `early_distribution_taxable` if `a_age < 60`.
        #
        # The `pension_lump_sum_done` flag latches the event so
        # later years see `pension_balance == 0` and don't re-run
        # the monthly-annuity initializer below.
        pension_lump_sum_taxable = 0.0
        pension_lump_sum_amount = 0.0
        pension_lump_sum_event = ""
        if (
            a_age >= inputs.pension.start_age
            and alive_a
            and inputs.pension.lump_sum_mode != "none"
            and not state.pension_lump_sum_done
            and state.pension_balance > 0.0
        ):
            lump = state.pension_balance
            if inputs.pension.lump_sum_mode == "rollover_pretax":
                state.spouse_a_pretax += lump
                state.spouse_a_pretax_ira += lump
                pension_lump_sum_event = "rollover_pretax"
            elif inputs.pension.lump_sum_mode == "cash":
                pension_lump_sum_taxable = lump
                pension_lump_sum_event = "cash"
            state.pension_balance = 0.0
            state.pension_annuity = 0.0
            state.pension_lump_sum_done = True
            pension_lump_sum_amount = lump

        # Initialize the pension annuity the first time we see spouse A
        # at-or-above NRD. The guard used to be a strict `==`, which
        # silently dropped the pension for households whose simulation
        # starts *at* or *past* NRD (already-retired-with-pension
        # scenarios — `state.pension_annuity` stayed 0 forever).
        # Skipped entirely once a lump-sum has already fired
        # (`pension_lump_sum_done`) — otherwise the floor below would
        # re-create the monthly annuity from the user's `monthly_at_nrd`.
        if (
            a_age >= inputs.pension.start_age
            and state.pension_annuity == 0.0
            and alive_a
            and inputs.pension.annual_at_nrd > 0
            and not state.pension_lump_sum_done
        ):
            scale = state.pension_balance / max(test_balance, 1.0)
            # If the simulation starts at-or-after NRD, the cash-balance
            # accumulator hasn't run any credit years, so `scale` may be
            # near zero. Honor the user's `monthly_at_nrd` input as the
            # baseline by flooring the scale at 1.0 in that case.
            if a_age >= inputs.pension.start_age and state.pension_balance == 0.0:
                scale = 1.0
            state.pension_annuity = inputs.pension.annual_at_nrd * scale
        # Surviving spouse pension election applied if A has died.
        if a_age >= inputs.pension.start_age and alive_a:
            pension_income = state.pension_annuity
        elif a_age >= inputs.pension.start_age and not alive_a and alive_b:
            pension_income = state.pension_annuity * cfg.mortality.pension_survivor_pct
        else:
            pension_income = 0.0
        # Lump-sum cash distribution joins ordinary income via the
        # `pension` kwarg (taxed at marginal) and the cash side via
        # `cash_inflow` further down. Roll-over has no current-year
        # income side — the rollover itself is tax-free.
        pension_income += pension_lump_sum_taxable

        # ---- Annuity contract (separate bucket from pension) -----
        # Pre-payout: balance grows at `inputs.annuity.growth_rate`
        # (defaults to `cfg.inflation` so the contract grows in real
        # terms at zero). No cash-balance / pay-credit accrual —
        # this is a passive contract the user already owns.
        if (
            a_age < inputs.annuity.start_age
            and not state.annuity_lump_sum_done
            and state.annuity_balance > 0
        ):
            ann_growth = (
                inputs.annuity.growth_rate
                if inputs.annuity.growth_rate is not None
                else cfg.inflation
            )
            state.annuity_balance *= 1.0 + ann_growth

        # Annuity lump-sum gate (parallels pension above).
        annuity_lump_sum_taxable = 0.0
        annuity_lump_sum_basis_returned = 0.0
        annuity_lump_sum_amount = 0.0
        annuity_lump_sum_event = ""
        if (
            a_age >= inputs.annuity.start_age
            and alive_a
            and inputs.annuity.lump_sum_mode != "none"
            and not state.annuity_lump_sum_done
            and state.annuity_balance > 0.0
        ):
            bal = state.annuity_balance
            if inputs.annuity.lump_sum_mode == "rollover_pretax":
                # Validation forbids non_qualified + rollover_pretax,
                # so this branch is only reachable for qualified.
                state.spouse_a_pretax += bal
                state.spouse_a_pretax_ira += bal
                annuity_lump_sum_event = "rollover_pretax"
            elif inputs.annuity.lump_sum_mode == "cash":
                if inputs.annuity.tax_kind == "non_qualified":
                    # IRC §72: basis returns tax-free, gain is
                    # ordinary. Surrender exhausts whatever basis
                    # remains; if some was already recovered via
                    # exclusion-ratio payments, only the residue
                    # is tax-free here.
                    tax_free = min(bal, state.annuity_basis_remaining)
                    taxable_part = bal - tax_free
                    state.annuity_basis_remaining -= tax_free
                    annuity_lump_sum_taxable = taxable_part
                    annuity_lump_sum_basis_returned = tax_free
                else:
                    annuity_lump_sum_taxable = bal
                annuity_lump_sum_event = "cash"
            state.annuity_balance = 0.0
            state.annuity_payment = 0.0
            state.annuity_lump_sum_done = True
            annuity_lump_sum_amount = bal

        # Monthly annuity income (only when no lump-sum elected and
        # the contract hasn't already been liquidated).
        annuity_taxable_inc = 0.0
        annuity_tax_free_inc = 0.0
        if (
            a_age >= inputs.annuity.start_age
            and inputs.annuity.lump_sum_mode == "none"
            and not state.annuity_lump_sum_done
            and inputs.annuity.annual_at_start > 0.0
            and state.annuity_balance > 0.0
        ):
            # Initialize once; payments stay nominal-fixed (matches
            # pension convention — no COLA built in).
            if state.annuity_payment == 0.0:
                state.annuity_payment = inputs.annuity.annual_at_start
            payment = min(state.annuity_payment, state.annuity_balance)

            # Survivor scaling matches the pension: surviving spouse
            # keeps `pension_survivor_pct` of the contract payment.
            # Scale the payment BEFORE draining the balance — pre-fix
            # the contract balance dropped by the full `payment` while
            # the household only received the survivor-fraction share,
            # so a 50% J&S election silently halved the contract's
            # economic life.
            if not alive_a and alive_b:
                payment *= cfg.mortality.pension_survivor_pct
            elif not alive_a and not alive_b:
                payment = 0.0
            state.annuity_balance = max(0.0, state.annuity_balance - payment)

            if inputs.annuity.tax_kind == "non_qualified":
                ratio = exclusion_ratio(
                    inputs.annuity.cost_basis,
                    inputs.annuity.annual_at_start,
                    inputs.annuity.expected_payout_years,
                )
                tax_free = min(payment * ratio, state.annuity_basis_remaining)
                state.annuity_basis_remaining -= tax_free
                annuity_taxable_inc = payment - tax_free
                annuity_tax_free_inc = tax_free
            else:
                annuity_taxable_inc = payment
        annuity_cash_in = (
            annuity_taxable_inc + annuity_tax_free_inc
            + annuity_lump_sum_taxable + annuity_lump_sum_basis_returned
        )

        # Social Security: per-spouse claim age, actuarial scaling on
        # FRA-PIA, and an annual COLA. The household-level `monthly_*`
        # inputs are quoted at FRA in today's dollars; we scale by
        # (a) claim-age factor (early-claim haircut or delayed-retirement
        # credits) and (b) (1+ss_cola_rate)**year_offset for COLA.
        ss_cola = cfg.ss_cola_rate if cfg.ss_cola_rate is not None else cfg.inflation
        ss_inflator = (1 + ss_cola) ** year_offset

        a_claim_age = inputs.ss.effective_start_age_a
        b_claim_age = inputs.ss.effective_start_age_b
        a_factor = claim_age_factor(a_claim_age, inputs.ss.fra_a)
        b_factor = claim_age_factor(b_claim_age, inputs.ss.fra_b)
        a_annual_at_claim = inputs.ss.monthly_spouse_a * 12 * a_factor
        b_annual_at_claim = inputs.ss.monthly_spouse_b * 12 * b_factor

        a_ss_eligible = alive_a and a_age >= a_claim_age
        b_ss_eligible = alive_b and b_age >= b_claim_age

        # SSA survivor-benefit eligibility: a widow(er) can claim a
        # survivor benefit on the deceased's record starting at age 60
        # (50 if disabled, or any age caring for a dependent under 16
        # — neither modeled). Reduced for early claim, but for the
        # household-planner level we treat the deceased's claim-age-
        # frozen benefit as the survivor amount (TC-2). Pre-Tier-C,
        # `ssn_income` was forced to $0 if the survivor hadn't reached
        # their *own* claim age, even after the spouse died — which
        # silently dropped 5–10 years of survivor benefits in any
        # mortality scenario where the higher-earner died first.
        SS_SURVIVOR_MIN_AGE = 60

        ssn_income = 0.0
        if alive_a and alive_b:
            if a_ss_eligible:
                ssn_income += a_annual_at_claim * ss_inflator
            if b_ss_eligible:
                ssn_income += b_annual_at_claim * ss_inflator
        elif inputs.household_kind == "single":
            # Never-married single filer. ``alive_b`` is False from
            # year 0 because the household has no spouse B at all —
            # NOT because of widowhood. Skip the survivor branch
            # entirely: a never-married filer has no deceased
            # spouse's record to claim a survivor benefit on.
            if alive_a and a_ss_eligible:
                ssn_income = a_annual_at_claim * ss_inflator
        else:
            # One spouse dead (true widowhood). Survivor takes
            # max(own, survivor-of-deceased) — but each piece is
            # gated by its own age rule:
            #   * own benefit: survivor must have reached own claim_age
            #   * survivor benefit (deceased's record): survivor must
            #     be at least SS_SURVIVOR_MIN_AGE (60).
            if alive_a:
                own = a_annual_at_claim if a_ss_eligible else 0.0
                survivor = (
                    b_annual_at_claim
                    if (a_age >= SS_SURVIVOR_MIN_AGE and not alive_b)
                    else 0.0
                )
                ssn_income = max(own, survivor) * ss_inflator
            elif alive_b:
                own = b_annual_at_claim if b_ss_eligible else 0.0
                survivor = (
                    a_annual_at_claim
                    if (b_age >= SS_SURVIVOR_MIN_AGE and not alive_a)
                    else 0.0
                )
                ssn_income = max(own, survivor) * ss_inflator

        # The "extra" non-portfolio income items (savings interest, a
        # one-off cap gain from a side hustle, an employer dividend on
        # ESPP shares, etc.) live on `inputs.income.*` and only apply
        # while at least one spouse is still working. Once both retire,
        # those streams stop -- but the taxable brokerage keeps
        # producing dividends and interest forever. That portfolio
        # yield is computed below from the year-start balance.
        # NOTE: gate on (a_working OR b_working). Previously this was
        # `a_working` only, which silently zeroed the extras during the
        # staggered-retirement window where A retires first but B is
        # still earning.
        anyone_working = a_working or b_working
        extra_interest = inputs.income.interest if anyone_working else 0.0
        extra_ltcg = inputs.income.capital_gains if anyone_working else 0.0
        extra_qdiv = inputs.income.dividends if anyone_working else 0.0

        # Portfolio yield from the taxable brokerage. Applies in every
        # year (working OR retired), proportional to year-start balance
        # and split via asset_location. This is what makes provisional
        # income / NIIT / IRMAA actually track post-retirement reality.
        taxable_eq_balance = max(0.0, state.taxable) * asset_loc.taxable.equity_pct
        taxable_bd_balance = max(0.0, state.taxable) * asset_loc.taxable.bond_pct
        portfolio_div_total = taxable_eq_balance * cfg.taxable_equity_div_yield
        # Split equity dividends into qualified (LTCG-rate) vs
        # non-qualified (ordinary-rate) per `cfg.taxable_equity_qualified_fraction`.
        # Defaults to 0.85 — closer to broad-market reality than
        # treating 100% as qualified.
        qfrac = min(1.0, max(0.0, cfg.taxable_equity_qualified_fraction))
        portfolio_qdiv = portfolio_div_total * qfrac
        portfolio_ord_div = portfolio_div_total * (1.0 - qfrac)
        portfolio_interest = taxable_bd_balance * cfg.taxable_bond_interest_yield

        interest_inc = extra_interest + portfolio_interest
        qdiv_inc = extra_qdiv + portfolio_qdiv
        ord_div_inc = portfolio_ord_div
        ltcg_inc = extra_ltcg

        # Age-aware effective standard deduction. Includes the §63(f)
        # 65+ add-on per eligible spouse plus the 2025 OBBBA senior
        # bonus during its 2025–2028 window. We compute it once per year
        # and inject into `base_kwargs` so every downstream `federal_tax`
        # call (cascade solvers, conversion sizer, post-cascade recompute)
        # automatically uses the correct figure — no separate threading.
        n_seniors_65plus = int(alive_a and a_age >= 65) + int(
            alive_b and b_age >= 65
        )
        calendar_year_y = cfg.start_year + year_offset
        deduction_y = regime.effective_std_deduction(
            filing_status,
            n_seniors_65plus=n_seniors_65plus,
            calendar_year=calendar_year_y,
        )

        # Annuity income: taxable portion routes through its own
        # `annuity_taxable` kwarg (joins the ordinary stack) so the
        # output DataFrame can show pension and annuity separately
        # even though both tax at marginal ordinary rates.
        annuity_taxable_total = annuity_taxable_inc + annuity_lump_sum_taxable

        # IRC §72(t) (qualified plans) / §72(q) (non-qualified
        # annuities): a 10% additional tax on the *taxable* portion
        # of an early distribution, on top of regular ordinary tax.
        # We use `a_age < 60` as the integer-age proxy for "before
        # 59½" — both match in practice for annual mid-year ages.
        # Rollovers are exempt by statute (no current-year tax at
        # all), as is the basis-return portion of a non-qualified
        # surrender (IRC §72(q) only hits the gain).
        early_distribution_taxable = 0.0
        if a_age < 60:
            if pension_lump_sum_event == "cash":
                early_distribution_taxable += pension_lump_sum_taxable
            if annuity_lump_sum_event == "cash":
                early_distribution_taxable += annuity_lump_sum_taxable

        base_kwargs = dict(
            wages=wages_box1,
            interest=interest_inc,
            ordinary_div=ord_div_inc,
            qualified_div=qdiv_inc,
            ltcg=ltcg_inc,
            pension=pension_income,
            annuity_taxable=annuity_taxable_total,
            social_security=ssn_income,
            deduction=deduction_y,
            early_distribution_taxable=early_distribution_taxable,
        )
        # Backdoor's taxable conversion (the pro-rata-taxable portion
        # of the same-year Roth conversion) lands as ordinary income.
        # We seed it onto base_kwargs so it shows up before
        # `planned_roth_conversion` adds any *additional* conversion.
        if ira_backdoor_taxable_conv > 0:
            base_kwargs["roth_conversion"] = (
                base_kwargs.get("roth_conversion", 0.0) + ira_backdoor_taxable_conv
            )

        # ---- RMDs first (required income; eats bracket headroom) ----
        # IRS rule: at age ≥ rmd_start_age, satisfy RMD BEFORE any
        # optional conversion. Computed up here so the conversion sizer
        # below sees the true post-RMD taxable-income line. After a
        # spousal-rollover-on-death event, the deceased's pretax is in
        # the survivor's bucket, so RMDs land on the right person.
        a_rmd = rmd_amount(state.spouse_a_pretax, a_age, cfg.rmd_start_age) if alive_a else 0.0
        b_rmd = rmd_amount(state.spouse_b_pretax, b_age, cfg.rmd_start_age) if alive_b else 0.0
        rmd_total = a_rmd + b_rmd

        # ---- Live cost-basis fraction (hoisted v6.5) ----
        current_basis_frac = (
            min(1.0, max(0.0, state.cumulative_basis / state.taxable))
            if state.taxable > 0 else 1.0
        )

        # State-tax-aware solver closure (F2). Without this, the gross-up
        # in `_solve_pretax_for_net` / `_solve_taxable_for_net` only
        # covers federal tax, so high-state-tax households (CA/MA/OR)
        # came up ~9-12% short on every cascade leg and quietly carried
        # the gap as "unfunded" or post-hoc rebalanced. Closure captures
        # this year's state regime / ages / HSA-contrib so the solver
        # can compute marginal state tax incrementally.
        # Hoisted in v6.5 so the conversion-liquidity sizer can use it
        # too (a CA conversion's 9.3% marginal state bite shouldn't
        # silently slip past the capacity check).
        _state_regime_local = state_regime
        _filing_local = filing_status
        _hsa_contrib_local = hsa_contrib
        _age_a_local = a_age
        _age_b_local = b_age
        _alive_a_local = alive_a
        _alive_b_local = alive_b

        def _state_tax_fn(kw: dict, ss_taxable_federal: float) -> float:
            # Pull `annuity_taxable` from the kwargs dict so the
            # conversion-liquidity sizer and deficit-cascade solvers
            # can see annuity income at the state level too. Without
            # this, the marginal state tax on a Roth conversion in a
            # year with a non-qualified annuity payment was off by
            # `marginal_state_rate × annuity_taxable_inc` — a real
            # dollar amount in CA / NY scenarios.
            ann_a = kw.get("annuity_taxable", 0.0)
            return state_tax(
                regime=_state_regime_local,
                filing_status=_filing_local,
                wages_box1=kw.get("wages", 0.0),
                interest=kw.get("interest", 0.0),
                ordinary_div=kw.get("ordinary_div", 0.0),
                qualified_div=kw.get("qualified_div", 0.0),
                ltcg=kw.get("ltcg", 0.0),
                pension=kw.get("pension", 0.0),
                pretax_withdrawal=kw.get("pretax_withdrawal", 0.0),
                roth_conversion=kw.get("roth_conversion", 0.0),
                social_security=kw.get("social_security", 0.0),
                ss_taxable_federal=ss_taxable_federal,
                annuity_taxable=ann_a,
                # Annuity is currently a single contract owned by
                # spouse A; the per-spouse split below routes the
                # whole amount to spouse A's NY $20k exclusion pool.
                annuity_per_spouse=(ann_a, 0.0),
                hsa_contrib=_hsa_contrib_local,
                age_a=_age_a_local,
                age_b=_age_b_local,
                alive_a=_alive_a_local,
                alive_b=_alive_b_local,
            )["state_tax"]

        # ---- Spending need + HSA pay-down (hoisted v6.5) ----
        # Lump events always add to net_need; the strategy picks where
        # the cash comes from. (Future enhancement: honor
        # `preferred_source` inside the strategy solvers.)
        years_until_horizon = (n_years - 1) - year_offset
        # F6: anchor the LTC shock to the household's actual end-of-
        # life (latest spouse-death year) rather than to the simulation
        # horizon. The household's "last year alive" is the LATEST of
        # any spouse's known year_of_death; for a single-filer household
        # we fall back to spouse A's death (spouse B never existed) and
        # for an MFJ household where ONE spouse has a death set and the
        # other is None (alive-to-horizon), the OTHER spouse defines
        # the household's last year — which is the horizon, not the
        # known death. Pre-fix the LTC shock fired against the dying
        # spouse's lifeline in the latter case, anchoring nursing-home
        # care to a year when the surviving spouse was still healthy.
        death_a = cfg.mortality.year_of_death_a
        death_b = cfg.mortality.year_of_death_b
        is_single = inputs.household_kind == "single"
        end_of_life_offset: int | None
        if is_single:
            # Single filer: only spouse A's death matters; spouse B is
            # absent, not "alive to horizon".
            if death_a is not None:
                end_of_life_offset = death_a - 1
            else:
                end_of_life_offset = None
        elif death_a is not None and death_b is not None:
            end_of_life_offset = max(death_a, death_b) - 1
        elif death_a is not None or death_b is not None:
            # MFJ household with one death set, the other alive-to-
            # horizon: the household's last year IS the horizon.
            end_of_life_offset = n_years - 1
        else:
            end_of_life_offset = None
        years_until_death = (
            end_of_life_offset - year_offset
            if end_of_life_offset is not None
            else None
        )
        net_need, lump_events = spending.amount_for(
            year_offset, a_age,
            years_until_horizon=years_until_horizon,
            years_until_death=years_until_death,
        )
        lump_total = 0.0
        for event in lump_events:
            inflated = event.amount_today * (1 + spending.inflation) ** year_offset
            net_need += inflated
            lump_total += inflated

        # Estate / heir-mode years: with both spouses dead the household
        # has no living member to spend, so the age-driven smile (and any
        # scheduled lump events) must collapse to zero. Without this guard
        # the deficit cascade keeps phantom-draining the taxable account
        # to fund a `spending_need` of $0 worth of consumption, which
        # silently understates the inheritance balance at horizon.
        # Tax on portfolio yield can still produce a tiny `delta < 0`
        # downstream and is funded by the deficit cascade as before.
        if not (alive_a or alive_b):
            net_need = 0.0
            lump_total = 0.0

        # HSA tax-free pay-down of qualified medical expense.
        # The LTC shock is the only explicit medical-expense line in
        # the model; HSA dollars are spent against it first (the whole
        # point of the triple-tax-advantaged shelter). Net_need drops
        # by the HSA draw, taxes on it stay zero (no AGI hit).
        ltc_today = 0.0
        ltc_anchor = years_until_death if years_until_death is not None else years_until_horizon
        if spending.ltc_shock and ltc_anchor >= 0 and ltc_anchor < spending.ltc_shock.years:
            ltc_today = (
                spending.ltc_shock.annual_cost_today
                * (1 + spending.inflation) ** year_offset
            )
        hsa_withdrawal = min(state.hsa, ltc_today) if ltc_today > 0 else 0.0
        if hsa_withdrawal > 0:
            state.hsa = max(0.0, state.hsa - hsa_withdrawal)
            net_need = max(0.0, net_need - hsa_withdrawal)

        # ---- Healthcare costs (hoisted v6.5) ----
        # Three pieces, all separate from federal/state income tax:
        #   * Pre-Medicare: `health_pre65` (anyone alive AND <65)
        #   * Medicare base: B+D premium per Medicare-enrolled spouse
        #   * IRMAA surcharge: looked up on AGI from year T-2 (TC-11)
        # Hoisted ahead of the conversion sizer so the liquidity guard
        # can subtract these obligations from non-conversion cash flow
        # to derive `tax_paying_capacity`. IRMAA uses lagged AGI so it
        # doesn't depend on this year's conversion; pre-65 / Medicare
        # premiums are statutory + age-driven, also conversion-free.
        n_pre_medicare = (
            int(alive_a and a_age < MEDICARE_ELIGIBLE_AGE)
            + int(alive_b and b_age < MEDICARE_ELIGIBLE_AGE)
        )
        health_pre65_today = cfg.health_pre65_today
        health_pre65 = (
            health_pre65_today
            * (n_pre_medicare / 2)
            * (1 + cfg.inflation) ** year_offset
        )

        n_medicare = (
            int(alive_a and a_age >= MEDICARE_ELIGIBLE_AGE)
            + int(alive_b and b_age >= MEDICARE_ELIGIBLE_AGE)
        )
        medicare_base_premium = (
            n_medicare
            * cfg.medicare_base_b_d_premium
            * (1 + cfg.inflation) ** year_offset
        )

        # IRMAA AGI lookback: year T uses year T-2 AGI under SSA.
        # For irmaa_lookback_years == 0 we fall back to current-year
        # AGI -- but the conversion adds to current-year AGI, so we
        # can't fully resolve IRMAA until after the conversion. Use
        # the *pre-conversion* AGI estimate here (federal_tax on
        # base_kwargs + RMD); for lookback>0 it doesn't matter
        # because IRMAA references lagged AGI which is fixed.
        if cfg.irmaa_lookback_years <= 0:
            preconv_kw = dict(base_kwargs)
            if rmd_total > 0:
                preconv_kw["pretax_withdrawal"] = (
                    preconv_kw.get("pretax_withdrawal", 0.0) + rmd_total
                )
            preconv_fed = federal_tax(
                regime=regime, filing_status=filing_status, **preconv_kw
            )
            irmaa_agi = float(preconv_fed["agi"])
        elif cfg.irmaa_lookback_years == 1:
            irmaa_agi = state.prior_agi
        else:
            irmaa_agi = state.agi_lag_2
        irmaa = irmaa_annual_surcharge(
            irmaa_agi, n_medicare, regime=regime, filing_status=filing_status
        )
        irmaa_cost = irmaa["total"]
        irmaa_tier = irmaa["tier"]

        # ---- Tax-paying capacity for Roth conversion (v6.5) ----
        # Estimate the household's non-Roth cash available to pay the
        # marginal federal + state tax on this year's conversion.
        # Used to bisect the conversion size down so the simulator
        # never converts more than the household could realistically
        # pay tax on. Before v6.5 an aggressive bracket-fill target
        # (or any fixed amount > liquid cash on hand) silently
        # triggered the deficit cascade and raided the just-funded
        # Roth bucket — defeating the strategy.
        if a_working or b_working:
            # Working years: wages (Box 1) + extras, net of FICA + SDI
            # and post-tax Roth-401(k) deferrals. Portfolio dividends /
            # interest hit the tax line but are notional (reinvested
            # into `state.taxable`), so they aren't real cash this year.
            # v6.7: subtract `a_roth_contrib + b_roth_contrib`. Roth
            # 401(k) deferrals don't reduce Box 1 (post-tax) but are
            # paycheck dollars routed into `state.roth`, so they are
            # NOT available cash for paying conversion tax. Pre-v6.7
            # this gap inflated `tax_paying_capacity` and the deficit
            # `delta` (see `cash_inflow` below) by the same dollars,
            # silently double-counting the Roth contribution.
            earned_cash = (
                wages_box1 + extra_interest + extra_qdiv + extra_ltcg
                - fica_total - sdi_total
                - a_roth_contrib - b_roth_contrib
            )
        else:
            earned_cash = 0.0
        # Annuity cash flow (taxable + tax-free portions, plus any
        # lump-sum cash this year) is real money in either working or
        # retired years — pre-fix the capacity solver omitted it,
        # silently shrinking the conversion sizer's headroom in any
        # year a non-qualified annuity was paying out.
        guaranteed_cash_no_conv = (
            earned_cash
            + pension_income
            + ssn_income
            + annuity_cash_in
            + rmd_total
        )
        # Tax on (income line + RMD) but BEFORE any conversion. This is
        # the "background" tax the household owes regardless of the
        # conversion; subtracting it leaves the cash that can be
        # earmarked specifically for conversion-marginal tax.
        capacity_kw = dict(base_kwargs)
        if rmd_total > 0:
            capacity_kw["pretax_withdrawal"] = (
                capacity_kw.get("pretax_withdrawal", 0.0) + rmd_total
            )
        capacity_fed = federal_tax(
            regime=regime, filing_status=filing_status, **capacity_kw
        )
        base_tax_no_conv = (
            capacity_fed["tax"]
            + _state_tax_fn(capacity_kw, capacity_fed["ss_taxable"])
        )
        committed_obligations = (
            base_tax_no_conv
            + net_need
            + medicare_base_premium + health_pre65 + irmaa_cost
            + ira_total_outflow + mega_backdoor_total
        )
        cash_surplus = max(0.0, guaranteed_cash_no_conv - committed_obligations)
        # Plus the share of taxable brokerage the user is willing to
        # spend on conversion tax (default 50% leaves a runway).
        # `taxable_slice` is intentionally additive on top of the
        # clamped cash_surplus: it's the user's preference knob for
        # how much of brokerage they're willing to allocate to
        # conversion tax, not a strict liquidity-net accounting (any
        # operational shortfall flows through the deficit cascade
        # which has its own bucket-priority logic).
        ratio = min(1.0, max(0.0, cfg.conversion_taxable_use_ratio))
        taxable_slice = max(0.0, state.taxable) * ratio
        tax_paying_capacity = cash_surplus + taxable_slice

        # ---- Roth conversion (RMD-aware, regime-aware, liquidity-capped) ----
        conversion_plan = planned_roth_conversion(
            cfg, inputs, state, base_kwargs,
            regime=regime, filing_status=filing_status,
            rmd_total=rmd_total,
            rmd_a=a_rmd,
            rmd_b=b_rmd,
            tax_paying_capacity=tax_paying_capacity,
            state_tax_fn=_state_tax_fn,
        )
        conv_a = conversion_plan.conv_a
        conv_b = conversion_plan.conv_b
        conv_capped_by_liquidity = conversion_plan.capped_by_liquidity
        conv_bracket_target = conversion_plan.bracket_target_total
        # If a spouse is dead, zero out their conversion.
        if not alive_a:
            conv_a = 0.0
        if not alive_b:
            conv_b = 0.0
        conv = conv_a + conv_b
        if conv > 0:
            base_kwargs["roth_conversion"] = base_kwargs.get("roth_conversion", 0.0) + conv
            pre_a = state.spouse_a_pretax
            pre_b = state.spouse_b_pretax
            state.spouse_a_pretax = max(0.0, pre_a - conv_a)
            state.spouse_b_pretax = max(0.0, pre_b - conv_b)
            _drain_pretax_ira("a", pre_a, conv_a)
            _drain_pretax_ira("b", pre_b, conv_b)
            state.roth += conv

        if a_working or b_working:
            withdraws = {
                "pretax_a": a_rmd,
                "pretax_b": b_rmd,
                "pretax": rmd_total,
                "roth": 0.0,
                "taxable": 0.0,
            }
        else:
            withdraws = withdraw_for_need(
                net_need, state, cfg, base_kwargs, a_rmd, b_rmd,
                cfg.withdrawal_strategy, current_basis_frac,
                regime=regime, filing_status=filing_status,
                state_tax_fn=_state_tax_fn,
            )

        # ---- Final tax kwargs ----
        final_kwargs = dict(base_kwargs)
        final_kwargs["pretax_withdrawal"] = (
            final_kwargs.get("pretax_withdrawal", 0.0) + withdraws["pretax"]
        )
        final_kwargs["ltcg"] = (
            final_kwargs.get("ltcg", 0.0)
            + withdraws["taxable"] * (1 - current_basis_frac)
        )

        tax_result = federal_tax(regime=regime, filing_status=filing_status, **final_kwargs)
        federal = tax_result["tax"]

        # ---- Per-spouse distribution breakdowns ----
        # For NY-style per-filer retirement exclusion we need to know
        # which spouse's pretax / Roth-conv income to apply the cap
        # against. In our model the pension belongs to spouse A; pretax
        # withdrawals and conversions are already tracked per-spouse.
        pension_split = (pension_income, 0.0)
        pretax_split = (withdraws["pretax_a"], withdraws["pretax_b"])
        roth_split = (conv_a, conv_b)

        # ---- State income tax ----
        # Computed on the same income line as federal; differences are
        # handled inside `state_tax` (HSA add-back for CA, retirement
        # exclusions for IL / NY, SS taxability fraction, etc.).
        # Annuity is a single-contract household input (no per-spouse
        # split modeled), so we route 100% of `annuity_taxable_total`
        # to spouse A — matching the convention used for pension
        # (`pension_split = (pension_income, 0.0)`). NY's $20k per-
        # filer exclusion in §612(c)(3-a) applies the cap against
        # spouse A's pool only.
        annuity_split = (final_kwargs.get("annuity_taxable", 0.0), 0.0)
        state_result = state_tax(
            regime=state_regime,
            filing_status=filing_status,
            wages_box1=final_kwargs.get("wages", 0.0),
            interest=final_kwargs.get("interest", 0.0),
            ordinary_div=final_kwargs.get("ordinary_div", 0.0),
            qualified_div=final_kwargs.get("qualified_div", 0.0),
            ltcg=final_kwargs.get("ltcg", 0.0),
            pension=final_kwargs.get("pension", 0.0),
            annuity_taxable=final_kwargs.get("annuity_taxable", 0.0),
            pretax_withdrawal=final_kwargs.get("pretax_withdrawal", 0.0),
            roth_conversion=final_kwargs.get("roth_conversion", 0.0),
            social_security=final_kwargs.get("social_security", 0.0),
            ss_taxable_federal=tax_result["ss_taxable"],
            hsa_contrib=hsa_contrib,
            age_a=a_age,
            age_b=b_age,
            alive_a=alive_a,
            alive_b=alive_b,
            pension_per_spouse=pension_split,
            pretax_per_spouse=pretax_split,
            roth_conv_per_spouse=roth_split,
            annuity_per_spouse=annuity_split,
        )
        state_income_tax = state_result["state_tax"]

        # Healthcare costs (n_pre_medicare, n_medicare, health_pre65,
        # medicare_base_premium, irmaa_*) were hoisted in v6.5 to
        # before the conversion sizer so the liquidity guard could
        # see them. When `cfg.irmaa_lookback_years <= 0` the hoisted
        # irmaa_agi used the pre-conversion AGI estimate; refresh it
        # here now that we have the post-conversion final tax_result.
        if cfg.irmaa_lookback_years <= 0:
            irmaa_agi = float(tax_result["agi"])
            irmaa = irmaa_annual_surcharge(
                irmaa_agi, n_medicare, regime=regime, filing_status=filing_status
            )
            irmaa_cost = irmaa["total"]
            irmaa_tier = irmaa["tier"]

        # ---- ACA premium tax credit (TC-13) ----
        # Post-IRA-2022 enhanced subsidies: if the household includes
        # any pre-65 adult and `cfg.aca_enabled=True`, the household's
        # premium contribution is capped at `cfg.aca_max_contrib_pct`
        # of MAGI. The credit equals max(0, benchmark - cap).
        # Modeling notes:
        #   * MAGI ≈ AGI for almost every household (untaxed SS adds
        #     are <1% in scope of typical ACA filers); we use AGI here.
        #   * No FPL × household-size lookup (Tier D); the 8.5% cap
        #     applies cliff-free at all incomes >= 150% FPL post-2022.
        #   * The credit is treated as cash (offsets cash outflow) —
        #     not a federal-tax line item, since most households take
        #     it as advance APTC paid directly to the insurer.
        aca_benchmark_total = 0.0
        aca_apt_credit = 0.0
        if cfg.aca_enabled and n_pre_medicare > 0:
            benchmark_per_adult_y = (
                cfg.aca_benchmark_premium_per_adult
                * (1 + cfg.inflation) ** year_offset
            )
            aca_benchmark_total = benchmark_per_adult_y * n_pre_medicare
            magi_proxy = float(tax_result["agi"])
            applicable_contrib = max(0.0, magi_proxy) * cfg.aca_max_contrib_pct
            aca_apt_credit = max(0.0, aca_benchmark_total - applicable_contrib)
        # Cash-flow offset: APTC reduces the household's premium
        # outlay. We bundle the benchmark premium into pre-65 health
        # cost and credit APTC against it (net of the credit becomes
        # the actual cash outflow). Capped at benchmark to avoid the
        # credit itself producing income.
        aca_apt_credit_offset = -aca_apt_credit + aca_benchmark_total
        # Subtract pre-65 health cost line entirely if ACA is enabled
        # (the benchmark covers it). When ACA is off, fall back to the
        # `health_pre65_today` knob.
        if cfg.aca_enabled and n_pre_medicare > 0:
            health_pre65 = 0.0   # absorbed into the benchmark/credit math

        # ---- Apply withdrawals to balances (single coherent pass) ----
        pre_w_a = state.spouse_a_pretax
        pre_w_b = state.spouse_b_pretax
        state.spouse_a_pretax = max(0.0, pre_w_a - withdraws["pretax_a"])
        state.spouse_b_pretax = max(0.0, pre_w_b - withdraws["pretax_b"])
        _drain_pretax_ira("a", pre_w_a, withdraws["pretax_a"])
        _drain_pretax_ira("b", pre_w_b, withdraws["pretax_b"])
        # Clamp post-withdrawal balances to >= 0. Rounding errors in
        # the cascade gross-up can leave a small negative residue that
        # the rest of the year-loop assumes away (e.g. portfolio yield
        # multiplies `state.taxable`, deficit cascade clamps but
        # subsequent code reads the value).
        state.roth = max(0.0, state.roth - withdraws["roth"])
        state.taxable = max(0.0, state.taxable - withdraws["taxable"])
        state.cumulative_basis = max(
            state.cumulative_basis - withdraws["taxable"] * current_basis_frac, 0.0
        )

        gross_cash_in = withdraws["pretax"] + withdraws["roth"] + withdraws["taxable"]

        # Cash actually flowing into the household this year (extra
        # interest / div / cap-gains on `inputs.income.*` are real cash
        # in working years; portfolio dividends/interest above are
        # *notional* -- already reinvested in the taxable balance via
        # market returns, so they are *not* added to cash_inflow even
        # though they hit the tax line).
        if a_working or b_working:
            # Working years: subtract employee-side FICA and state SDI.
            # Both are withholdings that don't show up in federal/state
            # income tax, so we deduct them explicitly from cash inflow.
            # v6.7: also subtract `a_roth_contrib + b_roth_contrib`.
            # Roth-401(k) deferrals are paycheck dollars routed to
            # `state.roth` (line 230). Pre-v6.7 they were missing from
            # this outflow, so `delta` carried that money a second time
            # into `state.taxable` — silently double-counting every Roth
            # 401(k) dollar (≈ deferral × marginal rate of free wealth
            # per year). Mirrors the same fix applied to `earned_cash`
            # in the tax-paying-capacity block above.
            #
            # `pension_income` and `ssn_income` are added explicitly
            # because a household can be still working (one or both
            # spouses earning W-2 wages) AND already drawing pension
            # or claiming SS in the same year — the working-year
            # branch had pre-fix omitted those streams entirely,
            # silently understating cash inflow (and forcing a
            # phantom deficit cascade) for any "phased retirement"
            # scenario. `pension_income` includes the current-year
            # cash component of any pension lump-sum already; the
            # capacity solver above also receives the same streams,
            # so the two stay in sync.
            cash_inflow = (
                wages_box1
                + extra_interest + extra_qdiv + extra_ltcg
                + pension_income
                + ssn_income
                + gross_cash_in
                + annuity_cash_in
                - fica_total
                - sdi_total
                - a_roth_contrib - b_roth_contrib
            )
        else:
            # `pension_income` already includes any current-year cash
            # lump-sum amount (added above); rollover-mode contributes
            # zero current-year cash, which is correct.
            cash_inflow = (
                pension_income + ssn_income + gross_cash_in + annuity_cash_in
            )

        delta = (
            cash_inflow
            - federal - state_income_tax - irmaa_cost
            - medicare_base_premium - health_pre65
            - aca_apt_credit_offset
            - net_need
            - ira_total_outflow
            - mega_backdoor_total
        )
        unfunded = 0.0

        if delta > 0:
            # Surplus: lands in taxable as new basis.
            state.taxable += delta
            state.cumulative_basis += delta
        elif delta < 0:
            # Deficit (lump-event shortfall, conversion-tax shortfall,
            # gap-year tax overrun, ...). Cascade through tax-efficient
            # buckets and gross up properly. Tracks any genuinely
            # unfunded gap so monte-carlo can see it.
            # HSA acts as a tax-deferred bucket only after either
            # spouse hits 65 (the IRS no-penalty age). Used for general
            # spending it grosses up at ordinary rate just like pretax.
            hsa_unlocked = max(a_age, b_age) >= 65
            # NOTE (TC-3): pass `final_kwargs`, NOT `base_kwargs`. The
            # cascade's `_solve_pretax_for_net` / `_solve_taxable_for_net`
            # use a base-tax delta to compute the marginal tax on the
            # next dollar drawn. That math is only correct if the base
            # tax already reflects this year's primary withdrawals and
            # conversions — i.e. the `final_kwargs` line. Passing
            # `base_kwargs` understated the marginal bracket on every
            # cascade leg in any year with an RMD or planned conversion.
            # v6.5: In any year a Roth conversion fires, exclude the
            # Roth bucket from the cascade. Otherwise an under-sized
            # liquidity check (or a fixed-dollar conversion that
            # overshoots capacity) silently withdraws from the just-
            # converted Roth — which under IRS rules can trigger a
            # 10% penalty on conversion principal if the holder is
            # < 59½ or the 5-year clock hasn't matured (neither
            # tracked by the model). The leftover shows as `unfunded`.
            roth_cascade_ok = not (
                conv > 0 and cfg.protect_roth_in_conversion_years
            )
            extra, unfunded = cover_deficit(
                deficit=-delta,
                state=state,
                base_kwargs=final_kwargs,
                basis_frac=current_basis_frac,
                pretax_a_already=withdraws["pretax_a"],
                pretax_b_already=withdraws["pretax_b"],
                taxable_already=withdraws["taxable"],
                roth_already=withdraws["roth"],
                hsa_already=hsa_withdrawal,
                hsa_unlocked=hsa_unlocked,
                roth_available=roth_cascade_ok,
                regime=regime,
                filing_status=filing_status,
                state_tax_fn=_state_tax_fn,
            )
            if extra["taxable"] > 0:
                state.taxable = max(0.0, state.taxable - extra["taxable"])
                state.cumulative_basis = max(
                    state.cumulative_basis - extra["taxable"] * current_basis_frac, 0.0
                )
                final_kwargs["ltcg"] = (
                    final_kwargs.get("ltcg", 0.0)
                    + extra["taxable"] * (1 - current_basis_frac)
                )
                withdraws["taxable"] += extra["taxable"]
            if extra["roth"] > 0:
                state.roth = max(0.0, state.roth - extra["roth"])
                withdraws["roth"] += extra["roth"]
            if extra.get("hsa", 0.0) > 0:
                state.hsa = max(0.0, state.hsa - extra["hsa"])
                final_kwargs["pretax_withdrawal"] = (
                    final_kwargs.get("pretax_withdrawal", 0.0) + extra["hsa"]
                )
                hsa_withdrawal += extra["hsa"]
            if extra["pretax"] > 0:
                pre_x_a = state.spouse_a_pretax
                pre_x_b = state.spouse_b_pretax
                state.spouse_a_pretax = max(0.0, pre_x_a - extra["pretax_a"])
                state.spouse_b_pretax = max(0.0, pre_x_b - extra["pretax_b"])
                _drain_pretax_ira("a", pre_x_a, extra["pretax_a"])
                _drain_pretax_ira("b", pre_x_b, extra["pretax_b"])
                final_kwargs["pretax_withdrawal"] = (
                    final_kwargs.get("pretax_withdrawal", 0.0) + extra["pretax"]
                )
                withdraws["pretax_a"] += extra["pretax_a"]
                withdraws["pretax_b"] += extra["pretax_b"]
                withdraws["pretax"] = withdraws["pretax_a"] + withdraws["pretax_b"]

            # Recompute federal / state against the post-cascade kwargs
            # so the recorded numbers reflect the additional draws.
            if extra["pretax"] > 0 or extra["taxable"] > 0 or extra.get("hsa", 0.0) > 0:
                tax_result = federal_tax(
                    regime=regime, filing_status=filing_status, **final_kwargs
                )
                federal = tax_result["tax"]
                state_result = state_tax(
                    regime=state_regime,
                    filing_status=filing_status,
                    wages_box1=final_kwargs.get("wages", 0.0),
                    interest=final_kwargs.get("interest", 0.0),
                    ordinary_div=final_kwargs.get("ordinary_div", 0.0),
                    qualified_div=final_kwargs.get("qualified_div", 0.0),
                    ltcg=final_kwargs.get("ltcg", 0.0),
                    pension=final_kwargs.get("pension", 0.0),
                    annuity_taxable=final_kwargs.get("annuity_taxable", 0.0),
                    pretax_withdrawal=final_kwargs.get("pretax_withdrawal", 0.0),
                    roth_conversion=final_kwargs.get("roth_conversion", 0.0),
                    social_security=final_kwargs.get("social_security", 0.0),
                    ss_taxable_federal=tax_result["ss_taxable"],
                    hsa_contrib=hsa_contrib,
                    age_a=a_age,
                    age_b=b_age,
                    alive_a=alive_a,
                    alive_b=alive_b,
                    pension_per_spouse=pension_split,
                    pretax_per_spouse=(
                        withdraws["pretax_a"],
                        withdraws["pretax_b"],
                    ),
                    roth_conv_per_spouse=roth_split,
                    annuity_per_spouse=annuity_split,
                )
                state_income_tax = state_result["state_tax"]
                # IRMAA in year T is based on MAGI from year T-2 under
                # the SSA two-year lookback (or year T-1 if the user
                # set irmaa_lookback_years=1, or year T for
                # irmaa_lookback_years=0). The cascade adds *current
                # year* AGI, so under any positive lookback this
                # year's IRMAA does NOT change — only T+lookback does.
                # The cascade's effect on future-year IRMAA is
                # captured by `state.prior_agi` / `state.agi_lag_2`,
                # which get refreshed from the post-cascade
                # `tax_result["agi"]` at the bottom of the loop.
                if cfg.irmaa_lookback_years <= 0:
                    irmaa = irmaa_annual_surcharge(
                        tax_result["agi"], n_medicare,
                        regime=regime, filing_status=filing_status,
                    )
                    irmaa_cost = irmaa["total"]
                    irmaa_tier = irmaa["tier"]

        # ---- Growth via MarketModel + AssetLocation ----
        eq_r, bd_r = market.returns(year_offset)
        pretax_return = 1 + asset_loc.pretax.annual_return(eq_r, bd_r)
        state.spouse_a_pretax *= pretax_return
        state.spouse_b_pretax *= pretax_return
        state.spouse_a_pretax_ira *= pretax_return
        state.spouse_b_pretax_ira *= pretax_return
        state.roth *= 1 + asset_loc.roth.annual_return(eq_r, bd_r)
        state.taxable *= 1 + asset_loc.taxable.annual_return(eq_r, bd_r) - cfg.taxable_drag
        state.hsa *= 1 + asset_loc.hsa.annual_return(eq_r, bd_r)
        if a_age < inputs.pension.start_age:
            # BP RAP interest credit: honor the participant's
            # configured rate (with the appropriate pre-/post-2016
            # floor). Falls back to PENSION_INTEREST if neither was
            # set on PensionInputs.
            yearly_rate = effective_interest_rate(
                inputs.pension.interest_rate
                if inputs.pension.interest_rate is not None
                else PENSION_INTEREST,
                pre_2016_participant=inputs.pension.pre_2016_participant,
            )
            state.pension_balance *= 1 + yearly_rate
            # Index the SSWB-quarter and IRS comp limit forward to
            # the current year. Both real-world figures track wage
            # growth; freezing them silently pushed more earnings
            # into the high band each year and ignored the cap on
            # high earners.
            current_kink = PENSION_QTR_SSWB * (1 + cfg.wage_growth) ** year_offset
            current_cap = inputs.pension.irs_comp_limit_today * (
                1 + cfg.wage_growth
            ) ** year_offset
            current_yos = (
                inputs.pension.years_of_service_today + year_offset
                if inputs.pension.years_of_service_today is not None
                else None
            )
            # BP RAP eligible earnings include base salary AND annual
            # incentive payments (SPD page 9). The IRS §401(a)(17)
            # comp cap is applied inside `pension_annual_credit`.
            eligible_earnings = (
                (spouse_a_salary + spouse_a_bonus) if a_working else 0.0
            )
            state.pension_balance += pension_annual_credit(
                eligible_earnings,
                age=a_age,
                years_of_service=current_yos,
                qtr_sswb=current_kink,
                comp_limit=current_cap,
            )

        # Salary growth only for living working spouses.
        if alive_a:
            spouse_a_salary *= 1 + cfg.wage_growth
            spouse_a_bonus *= 1 + cfg.wage_growth
        if alive_b:
            spouse_b_salary *= 1 + cfg.wage_growth

        prev_alive_a = alive_a
        prev_alive_b = alive_b

        # Defensive end-of-year clamp on liquid balances. The
        # individual cascade legs already clamp at the call site, but
        # a tiny rounding residue (e.g. `state.taxable = -0.0001`
        # after a basis-fraction multiplication) can survive into the
        # next year's portfolio-yield / cumulative-basis computation
        # and pollute downstream tests (and surface as confusing
        # negative values in CSV exports).
        state.taxable = max(0.0, state.taxable)
        state.cumulative_basis = max(0.0, state.cumulative_basis)
        state.roth = max(0.0, state.roth)
        state.hsa = max(0.0, state.hsa)

        # Roll the AGI lag chain forward (TC-5 + TC-11). At the start
        # of year T+1: `prior_agi` will equal AGI[T] and `agi_lag_2`
        # will equal AGI[T-1]. See `State` docstring for the
        # convention.
        state.agi_lag_2 = state.prior_agi
        state.prior_agi = float(tax_result["agi"])

        rows.append(
            {
                "year": year,
                "spouse_a_age": a_age,
                "spouse_b_age": b_age,
                "alive_a": alive_a,
                "alive_b": alive_b,
                "filing_status": filing_status,
                "regime": regime.name,
                "spousal_rollover": rollover_event,
                "wages": wages,
                "pension": pension_income,
                "pension_lump_sum": pension_lump_sum_amount,
                "pension_lump_sum_event": pension_lump_sum_event,
                "annuity_taxable": annuity_taxable_total,
                "annuity_tax_free": annuity_tax_free_inc + annuity_lump_sum_basis_returned,
                "annuity_payment": state.annuity_payment,
                "annuity_lump_sum": annuity_lump_sum_amount,
                "annuity_lump_sum_event": annuity_lump_sum_event,
                "early_distribution_penalty": tax_result.get(
                    "early_distribution_penalty", 0.0
                ),
                "ssn": ssn_income,
                "rmd": rmd_total,
                "rmd_a": a_rmd,
                "rmd_b": b_rmd,
                "roth_conversion": conv,
                "roth_conversion_a": conv_a,
                "roth_conversion_b": conv_b,
                "roth_conv_capped_by_liquidity": conv_capped_by_liquidity,
                "roth_conv_bracket_target": conv_bracket_target,
                "roth_conv_tax_capacity": tax_paying_capacity,
                "pretax_withdrawal": withdraws["pretax"],
                "pretax_withdrawal_a": withdraws["pretax_a"],
                "pretax_withdrawal_b": withdraws["pretax_b"],
                "roth_withdrawal": withdraws["roth"],
                "taxable_withdrawal": withdraws["taxable"],
                "agi": tax_result["agi"],
                "taxable_income": tax_result["taxable_income"],
                "federal_tax": federal,
                "marginal": tax_result["marginal"],
                "state_tax": state_income_tax,
                "state_marginal": state_result["state_marginal"],
                "state_regime": state_regime.name,
                "irmaa": irmaa_cost,
                "irmaa_tier": irmaa_tier,
                "irmaa_lookback_agi": irmaa_agi,
                "medicare_base_premium": medicare_base_premium,
                "health_pre65": health_pre65,
                "aca_benchmark_premium": aca_benchmark_total,
                "aca_apt_credit": aca_apt_credit,
                "spending_need": net_need,
                "unfunded": unfunded,
                "hsa_contrib": hsa_contrib,
                "hsa_withdrawal": hsa_withdrawal,
                "elective_deferral_a": a_total_contrib,
                "elective_deferral_b": b_total_contrib,
                "ira_traditional_a": ira_a.traditional,
                "ira_traditional_b": ira_b.traditional,
                "ira_roth_direct_a": ira_a.roth_direct,
                "ira_roth_direct_b": ira_b.roth_direct,
                "ira_backdoor_a": ira_a.backdoor,
                "ira_backdoor_b": ira_b.backdoor,
                "ira_backdoor_taxable_conv": ira_backdoor_taxable_conv,
                "mega_backdoor_a": a_after_tax,
                "mega_backdoor_b": b_after_tax,
                "excess_deferral_a": a_excess_deferral,
                "excess_deferral_b": b_excess_deferral,
                "mega_backdoor_spillover_a": a_after_tax_auto,
                "mega_backdoor_spillover_b": b_after_tax_auto,
                "after_tax_target_uncovered_a": a_after_tax_target_uncovered,
                "after_tax_target_uncovered_b": b_after_tax_target_uncovered,
                "employer_match_a": a_employer_match,
                "employer_match_b": b_employer_match,
                "fica": fica_total,
                "fica_oasdi": fica_combined["oasdi"],
                "fica_medicare": fica_combined["medicare"],
                "fica_additional_medicare": fica_combined["additional_medicare"],
                "state_sdi": sdi_total,
                "health_premium_a": hp_a,
                "health_premium_b": hp_b,
                "health_premium_total": hp_total,
                "qualified_dividends": qdiv_inc,
                "ordinary_dividends": ord_div_inc,
                "interest_income": interest_inc,
                "equity_return": eq_r,
                "bond_return": bd_r,
                "pretax_balance": state.spouse_a_pretax + state.spouse_b_pretax,
                "pretax_a_balance": state.spouse_a_pretax,
                "pretax_b_balance": state.spouse_b_pretax,
                "roth_balance": state.roth,
                "taxable_balance": state.taxable,
                "cumulative_basis": state.cumulative_basis,
                "hsa_balance": state.hsa,
                "pension_balance": state.pension_balance,
                "annuity_balance": state.annuity_balance,
                "annuity_basis_remaining": state.annuity_basis_remaining,
            }
        )
    return pd.DataFrame(rows)
