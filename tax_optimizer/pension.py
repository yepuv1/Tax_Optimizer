"""Cash-balance pension projector.

Generic cash-balance formula: 6% credit on the first quarter of the
Social-Security wage base, 11% above. 4.8% annual interest credit. At
`pension_start_age` the projected cash balance converts to a fixed
annuity scaled by realized vs expected balance.

The kink (`PENSION_QTR_SSWB`) is anchored to the 2025 SS wage base
($184,500). In real life this base indexes with average annual
wages (NAWI), so for multi-decade projections we should grow it
along with assumed wage growth. `pension_annual_credit_at_year`
exposes that path — call it with the offset from 2025 and your
wage-growth assumption to keep the kink real-dollar-equivalent.
"""

from __future__ import annotations

PENSION_INTEREST: float = 0.048
PENSION_BASE_YEAR: int = 2025  # `PENSION_QTR_SSWB` is anchored to this year.
PENSION_QTR_SSWB: float = 184_500.0 / 4
PENSION_LOW_RATE: float = 0.06
PENSION_HIGH_RATE: float = 0.11


def pension_annual_credit(
    annual_eligible_earnings: float, *, qtr_sswb: float | None = None
) -> float:
    """Annual credit added to the cash-balance account for a given salary.

    Works in **annual** dollars throughout: `PENSION_QTR_SSWB` is one
    quarter of the *annual* Social Security wage base, so the low-rate
    tier covers earnings up to that figure and the high-rate tier
    covers earnings above it. (Pre-v6.2 the function divided salary by
    12 before comparing to the annual threshold, which kept the high
    band dormant for any salary below ~$553k and systematically
    understated pension accrual by 30-40% for typical incomes.)

    `qtr_sswb` lets callers override the kink for a given year — used
    by `project_pension_balance` to index the kink forward with wage
    growth (F18).
    """
    earnings = max(0.0, annual_eligible_earnings)
    kink = qtr_sswb if qtr_sswb is not None else PENSION_QTR_SSWB
    low_credit = min(earnings, kink) * PENSION_LOW_RATE
    high_credit = max(0.0, earnings - kink) * PENSION_HIGH_RATE
    return low_credit + high_credit


def project_pension_balance(
    start_balance: float,
    start_earnings: float,
    years_to_retire: int,
    wage_growth: float = 0.025,
) -> float:
    """Forward-project a cash-balance account to retirement.

    Indexes the SSWB kink forward with `wage_growth` each year (F18).
    The SSWB tracks the National Average Wage Index (NAWI) in real
    life, which is well-approximated by `wage_growth` for projection
    purposes. Pre-v6.2 the kink was frozen at 2025's $184.5k/4, which
    silently pushed more earnings into the high band each decade.
    """
    bal, earn, kink = start_balance, start_earnings, PENSION_QTR_SSWB
    for _ in range(years_to_retire):
        bal = bal * (1 + PENSION_INTEREST) + pension_annual_credit(
            earn, qtr_sswb=kink
        )
        earn *= 1 + wage_growth
        kink *= 1 + wage_growth
    return bal
