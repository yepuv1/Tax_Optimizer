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

# HSA family-coverage cap and 55+ catch-up. Catch-up applies per
# spouse, but our model pools one HSA balance, so we apply one
# catch-up if *either* spouse is 55+. Medicare enrollment (age 65+)
# disqualifies further HSA contributions.
HSA_FAMILY_LIMIT: float = 8_550.0
HSA_CATCH_UP_55: float = 1_000.0
HSA_CATCH_UP_AGE: int = 55
HSA_INELIGIBLE_AGE: int = 65  # Medicare enrollment ends HSA contributions.


def elective_deferral_cap(age: int) -> float:
    """401(k) elective deferral cap including age-50+ catch-up."""
    cap = ELECTIVE_DEFERRAL_LIMIT
    if age >= ELECTIVE_DEFERRAL_CATCH_UP_AGE:
        cap += ELECTIVE_DEFERRAL_CATCH_UP_50
    return cap


def hsa_family_cap(age_a: int, age_b: int, *, either_working: bool) -> float:
    """Annual HSA family-coverage cap given both spouses' ages.

    Returns 0 once both spouses have hit the Medicare-eligibility
    cliff (no HDHP, no contributions) or no spouse is working.
    Catch-up adds once when either spouse is 55+.
    """
    if not either_working:
        return 0.0
    if age_a >= HSA_INELIGIBLE_AGE and age_b >= HSA_INELIGIBLE_AGE:
        return 0.0
    cap = HSA_FAMILY_LIMIT
    if age_a >= HSA_CATCH_UP_AGE or age_b >= HSA_CATCH_UP_AGE:
        cap += HSA_CATCH_UP_55
    return cap
