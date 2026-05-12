"""IRMAA Medicare-premium-surcharge engine.

The simulator drives this function with the right lookback AGI (typically
MAGI from 2 years prior, per the SSA rule) via ``cfg.irmaa_lookback_years``
(default 2). Pass any precomputed MAGI you like — this module just looks
up the right tier and dollar amounts; it does NOT compute the lookback.

Hold-harmless (Part B premium freeze for current beneficiaries when SS
COLA is small) is not modeled; the population that hits IRMAA is well
above the hold-harmless income line, so the error is negligible.
"""

from __future__ import annotations

from .regimes import TaxRegime

MEDICARE_ELIGIBLE_AGE: int = 65


def irmaa_annual_surcharge(
    magi: float,
    n_enrolled: int,
    *,
    regime: TaxRegime,
    filing_status: str = "mfj",
) -> dict:
    """Annual IRMAA surcharge for `n_enrolled` Medicare beneficiaries.

    Looks up the tier table from the active `regime` so a regime change
    that adjusts IRMAA thresholds (e.g. an inflation-only adjustment vs
    a deliberate political compression) flows through automatically.
    """
    tiers = regime.irmaa_tiers(filing_status)
    if n_enrolled <= 0 or magi <= tiers[0][0]:
        return {"partB": 0.0, "partD": 0.0, "total": 0.0, "tier": 0}
    for tier_idx, (cap, partB, partD) in enumerate(tiers):
        if magi <= cap:
            annual = (partB + partD) * 12 * n_enrolled
            return {
                "partB": partB * 12 * n_enrolled,
                "partD": partD * 12 * n_enrolled,
                "total": annual,
                "tier": tier_idx,
            }
    _cap, partB, partD = tiers[-1]
    annual = (partB + partD) * 12 * n_enrolled
    return {
        "partB": partB * 12 * n_enrolled,
        "partD": partD * 12 * n_enrolled,
        "total": annual,
        "tier": len(tiers) - 1,
    }
