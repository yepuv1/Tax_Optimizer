"""Required Minimum Distribution engine.

RMDs are computed per spouse, per account (not on a combined balance).
Each spouse's RMD is the prior-year balance of their own pretax accounts
divided by the IRS Uniform Lifetime divisor for that spouse's age, and
must be taken from their own account.
"""

from __future__ import annotations

UNIFORM_LIFETIME: dict[int, float] = {
    72: 27.4, 73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0, 79: 21.1,
    80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0, 86: 15.2, 87: 14.4,
    88: 13.7, 89: 12.9, 90: 12.2, 91: 11.5, 92: 10.8, 93: 10.1, 94: 9.5, 95: 8.9,
    96: 8.4, 97: 7.8, 98: 7.3, 99: 6.8, 100: 6.4, 101: 6.0, 102: 5.6, 103: 5.2,
    104: 4.9, 105: 4.6, 106: 4.3, 107: 4.1, 108: 3.9, 109: 3.7, 110: 3.5,
}


def rmd_amount(balance: float, age: int, rmd_start_age: int = 75) -> float:
    """Required minimum distribution for a single account given current age."""
    if age < rmd_start_age or balance <= 0:
        return 0.0
    keys = [k for k in UNIFORM_LIFETIME if k <= age]
    divisor = UNIFORM_LIFETIME.get(age, UNIFORM_LIFETIME[max(keys)] if keys else 1.0)
    return balance / divisor
