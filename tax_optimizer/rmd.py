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
    """Required minimum distribution for a single account given current age.

    Returns 0.0 when `age` falls below either `rmd_start_age` or the
    minimum age in `UNIFORM_LIFETIME` (72). The second guard is
    defensive: `Config.__post_init__` already rejects
    `rmd_start_age < 72`, but a stray caller could still pass a low
    age here, and we never want the prior fallback divisor of 1.0
    (which returned the entire balance as an RMD).
    """
    if age < rmd_start_age or balance <= 0:
        return 0.0
    min_table_age = min(UNIFORM_LIFETIME)
    if age < min_table_age:
        return 0.0
    if age in UNIFORM_LIFETIME:
        divisor = UNIFORM_LIFETIME[age]
    else:
        # Beyond the published table (>=110): pin to the top entry.
        divisor = UNIFORM_LIFETIME[max(UNIFORM_LIFETIME)]
    return balance / divisor
