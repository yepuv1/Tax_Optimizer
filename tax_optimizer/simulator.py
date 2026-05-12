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

import numpy as np
import pandas as pd

from .config import Config
from .conversion import planned_roth_conversion
from .inputs import Inputs, claim_age_factor
from .ira import allocate_ira_contributions
from .limits import SECTION_415C_LIMIT, elective_deferral_cap, hsa_family_cap
from .payroll import OASDI_WAGE_BASE_2026, fica_household
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
    prev_alive_a = True
    prev_alive_b = True

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
    test_balance = project_pension_balance(
        inputs.pension.balance_today,
        inputs.income.spouse_a_gross,
        max(0, inputs.pension.start_age - inputs.spouse_a_age_start),
        wage_growth=cfg.wage_growth,
        start_age=inputs.spouse_a_age_start,
        years_of_service_today=inputs.pension.years_of_service_today,
        interest_rate=inputs.pension.interest_rate,
        pre_2016_participant=inputs.pension.pre_2016_participant,
        comp_limit_today=inputs.pension.irs_comp_limit_today,
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
        a_after_tax_room = max(
            0.0,
            SECTION_415C_LIMIT - a_total_contrib - a_employer_match,
        )
        b_after_tax_room = max(
            0.0,
            SECTION_415C_LIMIT - b_total_contrib - b_employer_match,
        )
        a_after_tax = (
            min(spouse_a_salary * a_after_tax_pct, a_after_tax_room)
            if a_working else 0.0
        )
        b_after_tax = (
            min(spouse_b_salary * b_after_tax_pct, b_after_tax_room)
            if b_working else 0.0
        )
        state.roth += a_after_tax + b_after_tax
        mega_backdoor_total = a_after_tax + b_after_tax

        # HSA family contribution: capped at IRS family limit + 55+
        # catch-up, only valid while at least one spouse works AND is
        # Medicare-ineligible. Pre-tax: reduces wages_box1 below.
        hsa_cap = hsa_family_cap(a_age, b_age, either_working=(a_working or b_working))
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
        # HSA contributions are pre-tax (Box 1 wage reducer alongside
        # traditional 401(k) deferrals). Deductible Traditional IRA
        # contributions reduce AGI via Schedule 1 (not Box 1 directly,
        # but mathematically equivalent for our wages-driven AGI).
        wages_box1 = (
            wages
            - a_pretax_contrib - b_pretax_contrib
            - hsa_contrib
            - ira_traditional_deduction
        )

        # FICA on per-spouse W-2 wages. Applies to GROSS wages (Box 3) —
        # traditional 401(k) does NOT reduce FICA wages. Each spouse has
        # their own OASDI wage base; the wage base is indexed forward
        # annually so real FICA stays roughly constant. FICA does not
        # touch federal income tax — it just removes cash from the
        # household's working-year inflow.
        #
        # Additional Medicare 0.9% is reconciled at the **household**
        # level (Form 8959): MFJ threshold is $250k on combined wages,
        # not $200k per W-2. `fica_household` handles that; per-W-2
        # OASDI and base Medicare still come out of `fica_employee`
        # internally.
        wage_base_y = OASDI_WAGE_BASE_2026 * (1 + cfg.inflation) ** year_offset
        fica_combined = fica_household(
            a_w2_wages, b_w2_wages,
            filing_status=filing_status,
            wage_base=wage_base_y,
        )
        fica_total = fica_combined["total"]

        # Initialize the pension annuity the first time we see spouse A
        # at-or-above NRD. The guard used to be a strict `==`, which
        # silently dropped the pension for households whose simulation
        # starts *at* or *past* NRD (already-retired-with-pension
        # scenarios — `state.pension_annuity` stayed 0 forever).
        if (
            a_age >= inputs.pension.start_age
            and state.pension_annuity == 0.0
            and alive_a
            and inputs.pension.annual_at_nrd > 0
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
        else:
            # One spouse dead. Survivor takes max(own, survivor-of-
            # deceased) — but each piece is gated by its own age rule:
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
        portfolio_qdiv = taxable_eq_balance * cfg.taxable_equity_div_yield
        portfolio_interest = taxable_bd_balance * cfg.taxable_bond_interest_yield

        interest_inc = extra_interest + portfolio_interest
        qdiv_inc = extra_qdiv + portfolio_qdiv
        ltcg_inc = extra_ltcg

        base_kwargs = dict(
            wages=wages_box1,
            interest=interest_inc,
            qualified_div=qdiv_inc,
            ltcg=ltcg_inc,
            pension=pension_income,
            social_security=ssn_income,
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

        # ---- Roth conversion (RMD-aware, regime-aware) ----
        conv_a, conv_b = planned_roth_conversion(
            cfg, inputs, state, base_kwargs,
            regime=regime, filing_status=filing_status,
            rmd_total=rmd_total,
            rmd_a=a_rmd,
            rmd_b=b_rmd,
        )
        # If a spouse is dead, zero out their conversion.
        if not alive_a:
            conv_a = 0.0
        if not alive_b:
            conv_b = 0.0
        conv = conv_a + conv_b
        if conv > 0:
            base_kwargs["roth_conversion"] = conv
            pre_a = state.spouse_a_pretax
            pre_b = state.spouse_b_pretax
            state.spouse_a_pretax = max(0.0, pre_a - conv_a)
            state.spouse_b_pretax = max(0.0, pre_b - conv_b)
            _drain_pretax_ira("a", pre_a, conv_a)
            _drain_pretax_ira("b", pre_b, conv_b)
            state.roth += conv

        # ---- Spending need (smile + lump events + LTC) ----
        # Lump events always add to net_need; the strategy picks where
        # the cash comes from. (Future enhancement: honor
        # `preferred_source` inside the strategy solvers.)
        years_until_horizon = (n_years - 1) - year_offset
        # F6: anchor the LTC shock to the household's actual end-of-
        # life (latest spouse-death year) rather than to the simulation
        # horizon. Falls back to horizon when neither spouse has a
        # mortality date set.
        death_a = cfg.mortality.year_of_death_a
        death_b = cfg.mortality.year_of_death_b
        end_of_life_offset: int | None
        if death_a is not None and death_b is not None:
            end_of_life_offset = max(death_a, death_b) - 1  # last year alive
        elif death_a is not None:
            end_of_life_offset = death_a - 1
        elif death_b is not None:
            end_of_life_offset = death_b - 1
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

        # ---- HSA tax-free pay-down of qualified medical expense ----
        # The LTC shock is the only explicit medical-expense line in
        # the model; HSA dollars are spent against it first (the whole
        # point of the triple-tax-advantaged shelter). Net_need drops
        # by the HSA draw, taxes on it stay zero (no AGI hit).
        ltc_today = 0.0
        if spending.ltc_shock and years_until_horizon < spending.ltc_shock.years:
            ltc_today = (
                spending.ltc_shock.annual_cost_today
                * (1 + spending.inflation) ** year_offset
            )
        hsa_withdrawal = min(state.hsa, ltc_today) if ltc_today > 0 else 0.0
        if hsa_withdrawal > 0:
            state.hsa = max(0.0, state.hsa - hsa_withdrawal)
            net_need = max(0.0, net_need - hsa_withdrawal)

        # ---- Live cost-basis fraction ----
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
        _state_regime_local = state_regime
        _filing_local = filing_status
        _hsa_contrib_local = hsa_contrib
        _age_a_local = a_age
        _age_b_local = b_age
        _alive_a_local = alive_a
        _alive_b_local = alive_b

        def _state_tax_fn(kw: dict, ss_taxable_federal: float) -> float:
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
                hsa_contrib=_hsa_contrib_local,
                age_a=_age_a_local,
                age_b=_age_b_local,
                alive_a=_alive_a_local,
                alive_b=_alive_b_local,
            )["state_tax"]

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

        # ---- State income tax ----
        # Computed on the same income line as federal; differences are
        # handled inside `state_tax` (HSA add-back for CA, retirement
        # exclusions for IL / NY, SS taxability fraction, etc.).
        state_result = state_tax(
            regime=state_regime,
            filing_status=filing_status,
            wages_box1=final_kwargs.get("wages", 0.0),
            interest=final_kwargs.get("interest", 0.0),
            ordinary_div=final_kwargs.get("ordinary_div", 0.0),
            qualified_div=final_kwargs.get("qualified_div", 0.0),
            ltcg=final_kwargs.get("ltcg", 0.0),
            pension=final_kwargs.get("pension", 0.0),
            pretax_withdrawal=final_kwargs.get("pretax_withdrawal", 0.0),
            roth_conversion=final_kwargs.get("roth_conversion", 0.0),
            social_security=final_kwargs.get("social_security", 0.0),
            ss_taxable_federal=tax_result["ss_taxable"],
            hsa_contrib=hsa_contrib,
            age_a=a_age,
            age_b=b_age,
            alive_a=alive_a,
            alive_b=alive_b,
        )
        state_income_tax = state_result["state_tax"]

        # ---- Healthcare costs ----
        # Three pieces, all separate from federal/state income tax:
        #   * Pre-Medicare: `health_pre65` (anyone alive AND <65)
        #   * Medicare base: B+D premium per Medicare-enrolled spouse
        #   * IRMAA surcharge: looked up on AGI from year T-2 (TC-11)
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

        # IRMAA AGI lookback: year T uses year T-2 AGI (the SSA rule).
        # `state.agi_lag_*` are the 1- and 2-year lagged AGIs maintained
        # at the bottom of each loop iteration. For early years the
        # lag is 0, which means no IRMAA hit until prior years have
        # accumulated (the same way SSA defaults a new Medicare
        # enrollee's IRMAA to the lowest tier in their first year).
        if cfg.irmaa_lookback_years <= 0:
            irmaa_agi = float(tax_result["agi"])
        elif cfg.irmaa_lookback_years == 1:
            irmaa_agi = state.prior_agi
        else:
            # 2-year lookback (the SSA-published rule). `agi_lag_2`
            # holds AGI[T-2] at the start of year T per the State
            # convention. Anything > 2 falls back to T-2 (the lag chain
            # is only kept depth-2 to avoid slow-rolling out memory).
            irmaa_agi = state.agi_lag_2
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
        state.roth -= withdraws["roth"]
        state.taxable -= withdraws["taxable"]
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
            # Working years: subtract employee-side FICA. FICA is NOT
            # included in `federal` (it's a separate withholding line),
            # so we deduct it explicitly from cash inflow here.
            cash_inflow = (
                wages_box1
                + extra_interest + extra_qdiv + extra_ltcg
                + gross_cash_in
                - fica_total
            )
        else:
            cash_inflow = pension_income + ssn_income + gross_cash_in

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
                    pretax_withdrawal=final_kwargs.get("pretax_withdrawal", 0.0),
                    roth_conversion=final_kwargs.get("roth_conversion", 0.0),
                    social_security=final_kwargs.get("social_security", 0.0),
                    ss_taxable_federal=tax_result["ss_taxable"],
                    hsa_contrib=hsa_contrib,
                    age_a=a_age,
                    age_b=b_age,
                    alive_a=alive_a,
                    alive_b=alive_b,
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
        state.spouse_a_pretax *= 1 + asset_loc.pretax.annual_return(eq_r, bd_r)
        state.spouse_b_pretax *= 1 + asset_loc.pretax.annual_return(eq_r, bd_r)
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
            state.pension_balance += pension_annual_credit(
                spouse_a_salary if a_working else 0.0,
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
                "ssn": ssn_income,
                "rmd": rmd_total,
                "rmd_a": a_rmd,
                "rmd_b": b_rmd,
                "roth_conversion": conv,
                "roth_conversion_a": conv_a,
                "roth_conversion_b": conv_b,
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
                "employer_match_a": a_employer_match,
                "employer_match_b": b_employer_match,
                "fica": fica_total,
                "fica_oasdi": fica_combined["oasdi"],
                "fica_medicare": fica_combined["medicare"],
                "fica_additional_medicare": fica_combined["additional_medicare"],
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
            }
        )
    return pd.DataFrame(rows)
