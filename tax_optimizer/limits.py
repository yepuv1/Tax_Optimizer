"""IRS contribution limits and helpers.

The simulator caps user-supplied deferral percentages against these
limits so an aggressive `spouse_a_total_contrib_pct = 0.30` doesn't
silently route $90k/yr through a vehicle that's actually capped at
~$30k. Numbers are 2025/2026 published values; not auto-indexed
forward (a year-by-year COLA table would be a future enhancement).

These constants are intentionally module-level (and not on `Config`)
because they're rules-of-the-road, not user-tunable strategy knobs.
If Congress changes them, edit here.
"""

from __future__ import annotations

# 401(k) elective deferral cap and age-50+ catch-up. SECURE 2.0's
# enhanced 60-63 catch-up ($11,250 vs $7,500) is intentionally not
# modeled -- four extra years at +$3,750/yr is small change in a
# 30-year horizon and the eligibility test is annoying.
ELECTIVE_DEFERRAL_LIMIT: float = 23_500.0
ELECTIVE_DEFERRAL_CATCH_UP_50: float = 7_500.0
ELECTIVE_DEFERRAL_CATCH_UP_AGE: int = 50

# 401(k) §415(c) overall annual additions limit. Combines employee
# elective deferrals + employer match + after-tax (mega-backdoor)
# contributions. 2026 nominal. Catch-up ($7,500 50+) is NOT included
# in §415(c); it sits on top per IRS rules.
SECTION_415C_LIMIT: float = 70_000.0

# Traditional / Roth IRA contribution limit and 50+ catch-up.
# Applies separately per spouse. The "backdoor Roth IRA" path uses
# the same dollar cap (it's an after-tax IRA contribution + same-day
# Roth conversion).
IRA_CONTRIBUTION_LIMIT: float = 7_000.0
IRA_CATCH_UP_50: float = 1_000.0
IRA_CATCH_UP_AGE: int = 50

# Direct Roth IRA contribution MAGI phase-outs (2025/2026 nominal).
# Below `lo` ⇒ full contribution allowed.
# Between `lo` and `hi` ⇒ pro-rata phaseout.
# Above `hi` ⇒ direct contribution disallowed (backdoor still works).
ROTH_IRA_PHASEOUT_LO_MFJ: float = 236_000.0
ROTH_IRA_PHASEOUT_HI_MFJ: float = 246_000.0
ROTH_IRA_PHASEOUT_LO_SINGLE: float = 150_000.0
ROTH_IRA_PHASEOUT_HI_SINGLE: float = 165_000.0

# HSA family-coverage cap and 55+ catch-up. Catch-up applies per
# spouse, but our model pools one HSA balance, so we apply one
# catch-up if *either* spouse is 55+. Medicare enrollment (age 65+)
# disqualifies further HSA contributions.
HSA_FAMILY_LIMIT: float = 8_550.0
HSA_SELF_LIMIT: float = 4_300.0  # 2025 self-only HDHP limit.
HSA_CATCH_UP_55: float = 1_000.0
HSA_CATCH_UP_AGE: int = 55
HSA_INELIGIBLE_AGE: int = 65  # Medicare enrollment ends HSA contributions.


def elective_deferral_cap(age: int) -> float:
    """401(k) elective deferral cap including age-50+ catch-up."""
    cap = ELECTIVE_DEFERRAL_LIMIT
    if age >= ELECTIVE_DEFERRAL_CATCH_UP_AGE:
        cap += ELECTIVE_DEFERRAL_CATCH_UP_50
    return cap


def ira_contribution_cap(age: int) -> float:
    """Traditional / Roth IRA annual contribution cap including 50+ catch-up.

    Applies separately to each spouse. Spousal IRA contributions
    (non-working spouse using working spouse's earned income) get the
    same cap. The MAGI phase-out for direct Roth IRA is enforced
    separately by `roth_ira_phaseout_factor()`.
    """
    cap = IRA_CONTRIBUTION_LIMIT
    if age >= IRA_CATCH_UP_AGE:
        cap += IRA_CATCH_UP_50
    return cap


def roth_ira_phaseout_factor(magi: float, filing_status: str) -> float:
    """Return the fraction of the IRA cap a direct Roth contribution
    is allowed under the IRS MAGI phase-out.

    Below the lower bound ⇒ 1.0 (full contribution).
    Above the upper bound ⇒ 0.0 (no direct contribution; the backdoor
    is still legal, modeled separately).
    Linear pro-rata in between.
    """
    if filing_status == "single":
        lo, hi = ROTH_IRA_PHASEOUT_LO_SINGLE, ROTH_IRA_PHASEOUT_HI_SINGLE
    else:
        lo, hi = ROTH_IRA_PHASEOUT_LO_MFJ, ROTH_IRA_PHASEOUT_HI_MFJ
    if magi <= lo:
        return 1.0
    if magi >= hi:
        return 0.0
    return (hi - magi) / (hi - lo)


def hsa_family_cap(
    age_a: int,
    age_b: int,
    *,
    either_working: bool,
    b_alive: bool = True,
) -> float:
    """Annual HSA contribution cap given both spouses' ages and work status.

    The returned cap models the IRS rules:

      * **Family coverage (both spouses HDHP-eligible)** — full
        family limit, plus a single $1k catch-up if either spouse
        is 55+.
      * **Self-only coverage (one spouse on Medicare, other still
        working under HDHP)** — downshifts to the self-only limit
        for the working spouse, because the Medicare-enrolled
        spouse is no longer HDHP-eligible. The 55+ catch-up still
        applies to the HSA-eligible spouse.
      * **Both 65+** — zero (both on Medicare → no HDHP enrollment).
      * **Nobody working** — zero (no earned income / no plan).

    Pre-v6.2 the model kept the full family limit until BOTH
    spouses hit 65, overstating capacity by ~$4.3k/year in staggered-
    Medicare scenarios.

    ``b_alive`` lets a single-filer household (or a widowed survivor)
    drop spouse B out of the HDHP-eligibility calculation entirely:
    when False, B is treated as not-HSA-eligible regardless of age,
    which collapses the cap to self-only for spouse A. Defaults to
    True for back-compat with all existing callers.
    """
    if not either_working:
        return 0.0
    a_hsa_eligible = age_a < HSA_INELIGIBLE_AGE
    b_hsa_eligible = b_alive and age_b < HSA_INELIGIBLE_AGE
    if not a_hsa_eligible and not b_hsa_eligible:
        return 0.0
    if a_hsa_eligible and b_hsa_eligible:
        # Both HDHP-eligible — family coverage with one catch-up if
        # either is 55+.
        cap = HSA_FAMILY_LIMIT
        if age_a >= HSA_CATCH_UP_AGE or age_b >= HSA_CATCH_UP_AGE:
            cap += HSA_CATCH_UP_55
        return cap
    # Exactly one spouse is Medicare-eligible: downshift to self-only
    # for the still-HSA-eligible spouse. The Medicare-enrolled spouse
    # cannot be on HDHP, so family coverage no longer applies.
    cap = HSA_SELF_LIMIT
    # Catch-up only adds for the HSA-eligible spouse if they're 55+.
    eligible_age = age_a if a_hsa_eligible else age_b
    if eligible_age >= HSA_CATCH_UP_AGE:
        cap += HSA_CATCH_UP_55
    return cap
