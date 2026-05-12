"""Default scenario inputs.

These are the user-editable household-specific values: spouse ages,
retire ages, salaries, contribution rates, Roth-401(k) splits, starting
balances, Social Security amounts, etc. They're separated from `Config`
so users can edit just the "about me" data without touching the
simulation knobs (and so the same `Inputs` can drive different `Config`
strategies side-by-side).

Defaults reflect a typical dual-income married-filing-jointly couple turning
50 in 2026:
  - Combined household income ~$170k (mid-career professional + a lower-
    earning spouse). ~70th percentile of MFJ households per Census 2023 ACS.
  - Combined 401(k) balance $375k. Mean 401(k) balance for 50-somethings
    is $220-250k per Vanguard/Fidelity 2024 reporting; this assumes
    somewhat-better-than-mean savers, typical of tax-optimizer users.
  - Modest IRA, HSA, and taxable brokerage balances.
  - No private-sector pension (rare today; set if you have one).
  - SS estimates are SSA Quick-Calculator PIA figures at FRA for those
    career earnings.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

# Sentinel value used to detect "field was never touched by the caller".
# We use a non-default float (the historical default) and emit a warning
# only when a user explicitly sets `annual_expenses` to something else.
# See `Inputs.__post_init__` for the deprecation gating.
_ANNUAL_EXPENSES_LEGACY_DEFAULT = 85_000.0


@dataclass
class StartingBalances:
    spouse_a_pretax_401k: float = 225_000.0
    spouse_b_pretax_401k: float = 150_000.0
    spouse_a_roth_ira: float = 40_000.0
    spouse_b_roth_ira: float = 0.0
    spouse_a_pretax_ira: float = 0.0
    spouse_b_pretax_ira: float = 35_000.0
    pension_balance: float = 0.0
    hsa: float = 18_000.0
    taxable_brokerage: float = 80_000.0

    # NOTE: the simulator pools all Roth dollars into a single bucket
    # (state.roth) regardless of which spouse owns them. Pre-tax IRAs are
    # pooled per-spouse with that spouse's pre-tax 401(k) (so RMDs stay
    # tied to the right age). The per-account split is preserved here for
    # documentation and reporting only.

    @property
    def total_excl_real_estate(self) -> float:
        return (
            self.spouse_a_pretax_401k
            + self.spouse_b_pretax_401k
            + self.spouse_a_roth_ira
            + self.spouse_b_roth_ira
            + self.spouse_a_pretax_ira
            + self.spouse_b_pretax_ira
            + self.pension_balance
            + self.hsa
            + self.taxable_brokerage
        )


@dataclass
class CurrentIncome:
    spouse_a_gross: float = 95_000.0
    spouse_b_gross: float = 70_000.0
    spouse_a_bonus: float = 5_000.0
    interest: float = 500.0
    capital_gains: float = 1_000.0
    dividends: float = 2_000.0


@dataclass
class CurrentContrib:
    """Household contribution targets that aren't already on `Inputs` itself.

    The "main" 401(k) deferral percentages live on `Inputs` directly
    (`spouse_a_total_contrib_pct`, `spouse_a_roth_401k_pct`, ditto B)
    because they're optimizer decision variables. This dataclass is
    the home for the household contribution choices that aren't
    optimized over.

    Today that's just `hsa_family` (annual HSA family-coverage target,
    capped at the IRS limit at runtime). The simulator pre-tax-deducts
    this from wages and adds it to the HSA balance during working
    years; once both spouses hit Medicare eligibility (65) it's
    suppressed automatically.
    """

    hsa_family: float = 8_550.0


@dataclass
class PensionInputs:
    """Cash-balance pension inputs (BP-RAP-calibrated).

    Required fields (legacy):

      * ``balance_today``    — current cash-balance account value.
      * ``monthly_at_nrd``   — projected monthly annuity at NRD.
      * ``start_age``        — Normal Retirement Date (default 65).

    BP RAP-specific fields (default values preserve pre-v6.3
    behavior — the simulator computes pay credits using the top
    6%/11% tier when no age / service info is provided):

      * ``years_of_service_today`` — used to select the pay-credit
        tier each year; the model advances it +1 every working
        year. Set to ``None`` to keep the legacy 6%/11% behavior.
      * ``pre_2016_participant`` — controls the minimum-interest
        floor (5% pre-2016, 2% post-2016).
      * ``interest_rate`` — annual interest-credit rate; defaults
        to the participant's floor. Override to model a specific
        30-yr-Treasury assumption.
      * ``irs_comp_limit_today`` — IRS §401(a)(17) cap in today's
        dollars (defaults to 2025's $350k). Indexed forward with
        wage growth.
    """

    balance_today: float = 0.0
    monthly_at_nrd: float = 0.0
    start_age: int = 65  # NRD: age Spouse A's pension annuity begins.
    years_of_service_today: int | None = None
    # Default `False` (post-2016 / 2% floor) so the legacy 4.8% interest
    # default doesn't get floored to 5%. BP RAP participants who were
    # eligible before January 1, 2016 should flip this to `True`.
    pre_2016_participant: bool = False
    interest_rate: float | None = None
    irs_comp_limit_today: float = 350_000.0

    @property
    def annual_at_nrd(self) -> float:
        return self.monthly_at_nrd * 12


@dataclass
class SocialSecurity:
    """Household Social Security inputs.

    `monthly_spouse_a` / `monthly_spouse_b` are the **PIA at FRA** in
    today's dollars (the standard SSA-quoted estimate). The simulator
    applies (a) an actuarial scaling factor based on actual claim age
    relative to FRA, and (b) an annual COLA per `Config.ss_cola_rate`.

    Per-spouse claim ages live in `start_age_a` / `start_age_b`. The
    legacy single `start_age` knob is kept for backward compat: when
    `start_age_a` / `start_age_b` are `None`, both spouses claim at
    `start_age`. Setting either per-spouse value overrides the legacy
    field for that spouse only.

    `fra_a` / `fra_b` default to 67 (correct for anyone born 1960 or
    later, which covers anyone hitting retirement age in this model's
    typical horizon). For older birth cohorts, set explicitly.
    """

    monthly_spouse_a: float = 2_700.0  # PIA at FRA, today's dollars.
    monthly_spouse_b: float = 2_200.0  # PIA at FRA, today's dollars.
    start_age: int = 70  # Legacy fallback when start_age_a/b are None.
    start_age_a: int | None = None
    start_age_b: int | None = None
    fra_a: int = 67  # Full Retirement Age, Spouse A.
    fra_b: int = 67  # Full Retirement Age, Spouse B.

    @property
    def effective_start_age_a(self) -> int:
        return self.start_age_a if self.start_age_a is not None else self.start_age

    @property
    def effective_start_age_b(self) -> int:
        return self.start_age_b if self.start_age_b is not None else self.start_age


def claim_age_factor(claim_age: int, fra: int) -> float:
    """SSA actuarial scaling on FRA-PIA for `claim_age` ∈ [62, 70].

    Below 62 the spouse can't claim retirement benefits → 0.
    Past 70 there are no further delayed-retirement credits → caps at 70.
    Reduction (early): -5/9% per month for the first 36 months below FRA,
    then -5/12% per month thereafter (-30% at age 62 vs FRA 67).
    Increase (delayed): +8%/yr (= +2/3% per month) past FRA up to 70
    (+24% at age 70 vs FRA 67).
    """
    if claim_age < 62:
        return 0.0
    if claim_age > 70:
        claim_age = 70
    if claim_age == fra:
        return 1.0
    months = (claim_age - fra) * 12
    if months > 0:
        return 1.0 + (2.0 / 3.0) / 100.0 * months
    months = -months
    if months <= 36:
        return 1.0 - (5.0 / 9.0) / 100.0 * months
    return 1.0 - (5.0 / 9.0) / 100.0 * 36 - (5.0 / 12.0) / 100.0 * (months - 36)


@dataclass
class Inputs:
    """Container for all user-editable household scenario data.

    The first block (ages, retire ages, contribution rates, Roth-401(k)
    splits) lives directly on `Inputs` rather than nested under
    `contrib` because they're used independently throughout the
    simulator (timing, salary deferral math, optimizer decision
    variables) and grouping them top-level keeps the JSON scenario
    layout readable.
    """

    spouse_a_age_start: int = 50
    spouse_b_age_start: int = 50
    spouse_a_retire_age: int = 65
    spouse_b_retire_age: int = 65

    spouse_a_total_contrib_pct: float = 0.08
    spouse_b_total_contrib_pct: float = 0.06
    spouse_a_roth_401k_pct: float = 0.0
    spouse_b_roth_401k_pct: float = 0.0

    # Mega-backdoor Roth (after-tax 401(k) → in-plan Roth conversion).
    # Plan-dependent: requires the employer's 401(k) to allow both
    # after-tax (non-Roth, non-traditional) contributions AND
    # in-plan Roth conversions or in-service rollovers. Most plans
    # do NOT support this, but for those that do, it's the single
    # most powerful Roth-savings lever (~$40-50k/yr/spouse extra).
    #
    #   * `*_after_tax_401k_pct` — fraction of salary routed to
    #     after-tax 401(k) (then converted same-day to Roth). Capped
    #     each year by the §415(c) overall annual additions limit:
    #     §415(c) cap - elective deferrals - employer match.
    #   * `*_mega_backdoor_enabled` — boolean toggle (defaults False
    #     since most plans don't support it; setting True without
    #     plan support won't change the math but is logically wrong).
    #
    # After-tax dollars are NOT pre-tax (no Box 1 reduction) and NOT
    # FICA-exempt (already in wages_box1). They draw from after-tax
    # paycheck cash → Roth balance, with no income tax on the
    # contribution (already taxed) or the conversion (basis = 100%
    # of contribution, the same-day conversion has no earnings).
    spouse_a_after_tax_401k_pct: float = 0.0
    spouse_b_after_tax_401k_pct: float = 0.0
    spouse_a_mega_backdoor_enabled: bool = False
    spouse_b_mega_backdoor_enabled: bool = False

    # IRA contribution paths (Traditional, Roth, backdoor Roth).
    # Each spouse can contribute up to `IRA_CONTRIBUTION_LIMIT` (+50
    # catch-up) split across these three lines per year. The simulator
    # enforces the cap and the MAGI phase-out for direct Roth.
    #
    #   * `*_traditional_ira_contrib` — Traditional IRA contribution
    #     dollars/year. Treated here as **non-deductible** by default
    #     (the typical case for households with a workplace 401(k);
    #     deductibility phase-out is not modeled). The contribution
    #     adds to the Roth pretax basis only if you also do a
    #     same-year backdoor — otherwise it's a stranded after-tax
    #     IRA balance the simulator parks in `state.spouse_*_pretax`
    #     for simplicity (a modeling approximation).
    #   * `*_roth_ira_contrib` — direct Roth IRA contribution.
    #     Subject to MAGI phase-out (zeroed out above the limit).
    #   * `*_backdoor_roth` — boolean toggle. When True, the simulator
    #     contributes the full IRA cap as a non-deductible Traditional
    #     IRA, then immediately converts to Roth (the "backdoor"). If
    #     pretax-IRA balance is non-zero, the **pro-rata rule** taxes
    #     the conversion proportionally — the simulator models this.
    #     Income-uncapped (the whole point of the backdoor).
    #
    # Eligibility: requires earned income (own or spousal). The
    # simulator gates on "either spouse working AND alive".
    spouse_a_traditional_ira_contrib: float = 0.0
    spouse_b_traditional_ira_contrib: float = 0.0
    spouse_a_roth_ira_contrib: float = 0.0
    spouse_b_roth_ira_contrib: float = 0.0
    spouse_a_backdoor_roth: bool = False
    spouse_b_backdoor_roth: bool = False

    # Employer 401(k) match. Default 0 = no match (backward compat).
    # Modeled as: matched_dollars = salary
    #           * min(employee_pct, employer_match_max_pct)
    #           * employer_match_rate
    # i.e. "rate" is the fraction of the employee's deferral that the
    # employer matches, and "max_pct" caps the match at that fraction
    # of salary. The classic "100% on first 6%" plan is rate=1.0,
    # max_pct=0.06; a "50% on first 6%" plan (= 3% of pay) is rate=0.5,
    # max_pct=0.06. Regardless of the employee's Roth-vs-Traditional
    # election, the match always lands in the pre-tax bucket (IRS
    # rule), and it does NOT count against the elective-deferral cap.
    spouse_a_employer_match_rate: float = 0.0
    spouse_a_employer_match_max_pct: float = 0.0
    spouse_b_employer_match_rate: float = 0.0
    spouse_b_employer_match_max_pct: float = 0.0

    starting: StartingBalances = field(default_factory=StartingBalances)
    income: CurrentIncome = field(default_factory=CurrentIncome)
    contrib: CurrentContrib = field(default_factory=CurrentContrib)
    pension: PensionInputs = field(default_factory=PensionInputs)
    ss: SocialSecurity = field(default_factory=SocialSecurity)

    # DEPRECATED (v7): kept as a dataclass field for backward compatibility,
    # but the simulator does NOT read it. The simulator's spending base
    # comes from `cfg.resolved_spending()`, which prefers
    # `cfg.spending.base_spending` and falls back to
    # `cfg.annual_expenses_today`. A non-default value here only emits a
    # one-shot DeprecationWarning pointing users at the correct knob.
    annual_expenses: float = _ANNUAL_EXPENSES_LEGACY_DEFAULT

    def __post_init__(self) -> None:
        if self.annual_expenses != _ANNUAL_EXPENSES_LEGACY_DEFAULT:
            warnings.warn(
                "Inputs.annual_expenses is deprecated and ignored by the "
                "simulator. Set Config.annual_expenses_today (used when "
                "Config.spending is None) or Config.spending.base_spending "
                "(the SpendingProfile path) instead. This field will be "
                "removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )
