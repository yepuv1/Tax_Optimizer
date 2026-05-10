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
    Add'l       : 0.9% on wages above the statutory threshold (Form
    Medicare      8959). Per-W-2 employer withholding triggers at
                  $200k regardless of filing status, but the **actual
                  filing-time liability** uses the filing-status
                  threshold:
                    * single: $200,000
                    * MFJ:    $250,000 on combined wages
                  Use `fica_household(...)` to get the correct joint
                  liability; `fica_employee(...)` returns the
                  per-W-2 calculation that matches employer
                  withholding (useful for over/under-withholding
                  modeling).

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

# Per-W-2 employer-withholding threshold for the 0.9% Additional Medicare
# surtax. Statutory since 2013; not inflation-indexed in real life.
ADDITIONAL_MEDICARE_THRESHOLD: float = 200_000.0

# Filing-time (Form 8959) thresholds — what actually determines the
# household's Additional Medicare liability.
ADDITIONAL_MEDICARE_THRESHOLD_MFJ: float = 250_000.0
ADDITIONAL_MEDICARE_THRESHOLD_SINGLE: float = 200_000.0


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


def fica_household(
    wages_a: float,
    wages_b: float,
    *,
    filing_status: str,
    wage_base: float = OASDI_WAGE_BASE_2026,
) -> dict[str, float]:
    """Household-level FICA, with **Form 8959** Additional Medicare
    reconciliation done at the household combined-wages level.

    OASDI and base Medicare are per-W-2 (each spouse has their own wage
    base for OASDI, base Medicare is uncapped). Additional Medicare is
    the only piece that reconciles at filing time; this function applies
    the correct filing-status threshold ($250k MFJ vs $200k single) to
    combined wages, fixing the overstatement in earlier versions where
    a couple with two $180k W-2s would skip Additional Medicare entirely
    (each under $200k) when in fact their joint $360k crosses the $250k
    MFJ floor.
    """
    a = fica_employee(wages_a, wage_base=wage_base)
    b = fica_employee(wages_b, wage_base=wage_base)
    combined = max(0.0, wages_a) + max(0.0, wages_b)
    if filing_status == "mfj":
        threshold = ADDITIONAL_MEDICARE_THRESHOLD_MFJ
    else:
        threshold = ADDITIONAL_MEDICARE_THRESHOLD_SINGLE
    addl_household = max(0.0, combined - threshold) * ADDITIONAL_MEDICARE_RATE
    oasdi = a["oasdi"] + b["oasdi"]
    medicare = a["medicare"] + b["medicare"]
    return {
        "oasdi": oasdi,
        "medicare": medicare,
        "additional_medicare": addl_household,
        "total": oasdi + medicare + addl_household,
    }
