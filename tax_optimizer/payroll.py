"""Employee-side payroll tax (FICA) on W-2 wages.

FICA is computed on **gross** W-2 wages (Box 3, "Social Security wages"),
which means it's NOT reduced by traditional 401(k) elective deferrals.
HSA contributions through a cafeteria plan ARE FICA-exempt — but the
simulator routes HSA dollars through wages_box1 only (see `simulator.py`
B1 wiring), and the Box-3 wages used here remain gross. This is the
common-case approximation; cafeteria-plan HSAs reduce FICA wages by
the contribution amount in real life.

FICA is purely a cash-flow cost. It does NOT affect federal income tax
(it's collected separately) and is NOT a Box-1 wage reducer.

Three components:
    OASDI       : 6.2% to a per-individual wage base ($176,100 in 2026,
                  CPI-W-indexed annually by SSA).
    Medicare    : 1.45% on all wages, no cap.
    Add'l       : 0.9% on wages above $200,000 (statutory threshold,
    Medicare      NOT indexed since 2013, applied per W-2 — the joint
                  $250k MFJ test is reconciled at filing time and is
                  approximated here at the per-spouse level).

The wage base is published by SSA each year. We index from the 2026
nominal forward at `cfg.inflation` to keep real-FICA roughly constant
across a 30-year horizon; if Congress freezes the base (rare in modern
practice), set `wage_base` directly when calling `fica_employee`.
"""

from __future__ import annotations

OASDI_WAGE_BASE_2026: float = 176_100.0
OASDI_RATE: float = 0.062
MEDICARE_RATE: float = 0.0145
ADDITIONAL_MEDICARE_RATE: float = 0.009

# Per-W-2 (employee-side) threshold for the 0.9% additional Medicare
# surtax. Statutory since 2013; not inflation-indexed in real life.
ADDITIONAL_MEDICARE_THRESHOLD: float = 200_000.0


def fica_employee(
    wages: float,
    *,
    wage_base: float = OASDI_WAGE_BASE_2026,
    addl_threshold: float = ADDITIONAL_MEDICARE_THRESHOLD,
) -> dict[str, float]:
    """OASDI + Medicare + Additional Medicare on a single W-2's wages.

    Returns a dict with each component plus `total`. Pass each spouse's
    wages separately (each spouse has their own OASDI wage base and
    their own additional-Medicare threshold per W-2).
    """
    if wages <= 0:
        return {
            "oasdi": 0.0,
            "medicare": 0.0,
            "additional_medicare": 0.0,
            "total": 0.0,
        }
    oasdi = min(wages, wage_base) * OASDI_RATE
    medicare = wages * MEDICARE_RATE
    addl = max(0.0, wages - addl_threshold) * ADDITIONAL_MEDICARE_RATE
    return {
        "oasdi": oasdi,
        "medicare": medicare,
        "additional_medicare": addl,
        "total": oasdi + medicare + addl,
    }
