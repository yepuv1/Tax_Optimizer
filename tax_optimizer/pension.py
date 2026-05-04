"""Cash-balance pension projector.

Generic cash-balance formula: 6% credit on the first quarter of the
Social-Security wage base, 11% above. 4.8% annual interest credit. At
`pension_start_age` the projected cash balance converts to a fixed
annuity scaled by realized vs expected balance.
"""

from __future__ import annotations

PENSION_INTEREST: float = 0.048
PENSION_QTR_SSWB: float = 184_500.0 / 4
PENSION_LOW_RATE: float = 0.06
PENSION_HIGH_RATE: float = 0.11


def pension_annual_credit(annual_eligible_earnings: float) -> float:
    """Annual credit added to the cash-balance account for a given salary."""
    monthly_e = annual_eligible_earnings / 12.0
    low_credit = min(monthly_e, PENSION_QTR_SSWB) * PENSION_LOW_RATE
    high_credit = max(0.0, monthly_e - PENSION_QTR_SSWB) * PENSION_HIGH_RATE
    return (low_credit + high_credit) * 12


def project_pension_balance(
    start_balance: float,
    start_earnings: float,
    years_to_retire: int,
    wage_growth: float = 0.025,
) -> float:
    """Forward-project a cash-balance account to retirement."""
    bal, earn = start_balance, start_earnings
    for _ in range(years_to_retire):
        bal = bal * (1 + PENSION_INTEREST) + pension_annual_credit(earn)
        earn *= 1 + wage_growth
    return bal
