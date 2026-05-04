"""IRMAA Medicare-premium-surcharge engine.

Lookback note: real IRMAA uses MAGI from 2 years prior. We approximate
with current-year AGI to keep the simulator tractable; the error is small
unless income changes sharply between consecutive years.
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
