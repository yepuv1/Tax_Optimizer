"""Cash-balance pension projector — calibrated to the **BP Retirement
Accumulation Plan (RAP)**.

Modeling fidelity check: the formulae here match the "How the plan
works → Pay credits → Pay Credit Formula" table on page 9 of the
BP RAP August-2023 Summary Plan Description. The rate tiers, the
quarter-SSWB kink, the 4.8% / 5% interest floors, and the IRS
§401(a)(17) compensation limit are all implemented. Where the SPD
applies a rule monthly (e.g. pay credits on each month's eligible
earnings, with a $3,337.50 / mo kink in 2023), this module applies
the equivalent annual aggregate; the two only differ for users with
heavily front-/back-loaded annual incentive payments inside a
single calendar year (rare, and conservative when it matters).

**Pay-credit tiers (BP RAP page 9):**

  ┌─────────────────────┬────┬───────────┬──────────┬──────────┐
  │ Years of Service    │    │  Age      │ Low band │ High band│
  ├─────────────────────┼────┼───────────┼──────────┼──────────┤
  │ <10                 │ AND│  <40      │   4%     │   7%     │
  │ 10 to <20           │ OR │  40 to <50│   5%     │   9%     │
  │ 20+                 │ OR │  50+      │   6%     │  11%     │
  └─────────────────────┴────┴───────────┴──────────┴──────────┘

Rate selection: **whichever tier (age OR service) gives the higher
percentage wins.** A 42-year-old with 8 YoS is in the 5%/9% tier (age
tier dominates). A 35-year-old with 21 YoS is in the 6%/11% tier
(service tier dominates).

**Threshold (kink):** one-quarter of the *annual* Social Security
Wage Base. The 2025 SSWB is $184,500, so ¼ = $46,125.

**IRS §401(a)(17) compensation limit:** $350,000 in 2025
(inflation-indexed). The SPD applies this cap to eligible earnings
before computing pay credits, so a $500k earner only accrues pay
credits on the first $350k.

**Interest credit:** monthly, equal to `max(30-yr Treasury Bond
average from 4 months ago, floor)`. Floor = 5% for participants
eligible pre-2016, 2% for post-2016. Our annual approximation uses
the configured rate directly with the appropriate floor.

Modeling scope (F11): the projector and the simulator's pension
column track **spouse A only**. Multi-pension households (both
spouses with cash-balance plans) are not modeled. The recommended
workaround is to enter the *combined* monthly-at-NRD as a single
spouse-A pension and accept the SS-taxability / state-pension-
exclusion code paths will key off spouse A's age.
"""

from __future__ import annotations

PENSION_INTEREST: float = 0.048
"""Default annual interest credit (4.8%). Used by the simulator when
`PensionInputs.interest_rate` is not overridden. Reasonable estimate
for long-run 30-yr Treasury average, but a v6.1+ scenario can set
`pension.interest_rate` directly to model a specific rate path.
"""

PENSION_BASE_YEAR: int = 2025
"""`PENSION_QTR_SSWB` and `IRS_COMP_LIMIT` are anchored to this year."""

PENSION_QTR_SSWB: float = 184_500.0 / 4
"""Quarter of the Social-Security wage base (2025: $184,500/4 = $46,125).
Indexed forward with wage growth inside `project_pension_balance`."""

IRS_COMP_LIMIT: float = 350_000.0
"""IRS §401(a)(17) annual compensation limit (2025). The SPD caps
eligible earnings at this figure before computing pay credits.
Indexed forward with wage growth inside `project_pension_balance`."""

PENSION_FLOOR_PRE_2016: float = 0.05
"""Minimum annual interest-credit rate for participants eligible
pre-2016. Per BP RAP page 10."""

PENSION_FLOOR_POST_2016: float = 0.02
"""Minimum annual interest-credit rate for participants hired/rehired
on or after January 1, 2016. Per BP RAP page 10."""

# ---------------------------------------------------------------------------
# Top-band defaults (kept for pre-v6.3 callers that don't pass age / YoS).
# Equal to the BP RAP top tier (the historical defaults shipped pre-v6.3).
# ---------------------------------------------------------------------------

PENSION_LOW_RATE: float = 0.06
PENSION_HIGH_RATE: float = 0.11


def pay_credit_rates(
    age: int | None = None,
    years_of_service: int | None = None,
) -> tuple[float, float]:
    """Return ``(low_rate, high_rate)`` for the participant.

    Implements the "whichever provides the greater percentage" rule
    from the BP RAP SPD: a participant is placed in the higher of
    their age tier and service tier.

    Either argument may be ``None``, in which case the rule defaults
    to the legacy top-band rates (6% / 11%) so callers from the
    pre-v6.3 era preserve their previous behavior.
    """
    if age is None and years_of_service is None:
        return PENSION_LOW_RATE, PENSION_HIGH_RATE
    a = age if age is not None else 0
    yos = years_of_service if years_of_service is not None else 0
    # Map each axis to a tier index (0 = base, 1 = mid, 2 = top).
    age_tier = 0 if a < 40 else (1 if a < 50 else 2)
    yos_tier = 0 if yos < 10 else (1 if yos < 20 else 2)
    tier = max(age_tier, yos_tier)
    if tier == 2:
        return 0.06, 0.11
    if tier == 1:
        return 0.05, 0.09
    return 0.04, 0.07


def pension_annual_credit(
    annual_eligible_earnings: float,
    *,
    age: int | None = None,
    years_of_service: int | None = None,
    qtr_sswb: float | None = None,
    comp_limit: float | None = None,
) -> float:
    """Annual pay credit added to the cash-balance account.

    Faithful to BP RAP pay-credit rules:

      * Eligible earnings are capped at the IRS §401(a)(17)
        compensation limit (``comp_limit``; defaults to
        ``IRS_COMP_LIMIT``).
      * The low-rate band applies to the first ``qtr_sswb`` of
        capped earnings; the high-rate band applies above. Defaults
        to ``PENSION_QTR_SSWB``.
      * Tier (4/7, 5/9, 6/11) is selected by ``pay_credit_rates``
        using the participant's age and years of service. When both
        are unset, the top tier is used for backward compat.

    All math is in annual dollars (the SPD operates monthly with the
    annual SSWB-quarter kink expressed as $3,337.50/mo in 2023; the
    annual aggregate matches monthly exactly for level income and
    is conservative for back-loaded bonuses).
    """
    earnings = max(0.0, annual_eligible_earnings)
    cap = comp_limit if comp_limit is not None else IRS_COMP_LIMIT
    if cap > 0:
        earnings = min(earnings, cap)
    kink = qtr_sswb if qtr_sswb is not None else PENSION_QTR_SSWB
    low_rate, high_rate = pay_credit_rates(age=age, years_of_service=years_of_service)
    low_credit = min(earnings, kink) * low_rate
    high_credit = max(0.0, earnings - kink) * high_rate
    return low_credit + high_credit


def effective_interest_rate(
    requested_rate: float | None = None,
    *,
    pre_2016_participant: bool = True,
) -> float:
    """Return the per-year interest-credit rate to apply.

    BP RAP guarantees a minimum-rate floor:

      * Pre-2016 participants: ``PENSION_FLOOR_PRE_2016`` (5%).
      * Post-2016 participants: ``PENSION_FLOOR_POST_2016`` (2%).

    When ``requested_rate`` is supplied, it's only applied if it
    meets the participant's floor. ``None`` means "use my floor"
    (most conservative for projections that don't want to bet on a
    specific Treasury path).
    """
    floor = PENSION_FLOOR_PRE_2016 if pre_2016_participant else PENSION_FLOOR_POST_2016
    if requested_rate is None:
        return floor
    return max(requested_rate, floor)


def project_pension_balance(
    start_balance: float,
    start_earnings: float,
    years_to_retire: int,
    wage_growth: float = 0.025,
    *,
    start_age: int | None = None,
    years_of_service_today: int | None = None,
    interest_rate: float | None = None,
    pre_2016_participant: bool = False,
    comp_limit_today: float | None = None,
    retire_age: int | None = None,
) -> float:
    """Forward-project a cash-balance account to retirement.

    Each year:

      * Apply the interest credit to the prior balance.
      * If the participant is still working (``start_age + offset <
        retire_age``), add a pay credit computed against the current
        SSWB-quarter kink and IRS comp-limit (both indexed forward
        with ``wage_growth``). Once the participant has retired, only
        interest credits continue accruing until ``years_to_retire``
        elapses (matching the BP RAP rule: pay credits stop at
        termination, interest credits continue until benefit payment).
      * Tier is recomputed each year using ``start_age + year`` and
        ``years_of_service_today + year`` (so a participant crossing
        age 50 or 20 YoS during the projection switches to the
        higher tier).

    Pre-v6.3 callers that don't pass ``start_age`` / ``years_of_service_today``
    still get the legacy top-tier behavior. Pre-v6.3 callers that
    don't pass ``retire_age`` get the old "credits all the way to
    NRD" behavior — set ``retire_age`` to stop pay credits at
    actual retirement (the simulator passes this automatically).
    """
    # `interest_rate=None` keeps backward compat with pre-v6.3 callers:
    # falls back to `PENSION_INTEREST` (4.8%) before applying the
    # participant's floor.
    requested = interest_rate if interest_rate is not None else PENSION_INTEREST
    rate = effective_interest_rate(
        requested, pre_2016_participant=pre_2016_participant
    )
    bal = start_balance
    earn = start_earnings
    kink = PENSION_QTR_SSWB
    cap = comp_limit_today if comp_limit_today is not None else IRS_COMP_LIMIT
    for offset in range(years_to_retire):
        bal *= 1 + rate
        age_now = (start_age + offset) if start_age is not None else None
        yos_now = (
            (years_of_service_today + offset) if years_of_service_today is not None else None
        )
        is_working = (
            retire_age is None
            or age_now is None
            or age_now < retire_age
        )
        if is_working:
            bal += pension_annual_credit(
                earn,
                age=age_now,
                years_of_service=yos_now,
                qtr_sswb=kink,
                comp_limit=cap,
            )
        earn *= 1 + wage_growth
        kink *= 1 + wage_growth
        cap *= 1 + wage_growth
    return bal
