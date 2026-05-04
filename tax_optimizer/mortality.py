"""Mortality / widow's-penalty modeling.

When one spouse dies, the survivor switches from MFJ to single-filer
status, which compresses tax brackets, halves the standard deduction,
lowers the NIIT and IRMAA thresholds, and changes the SS taxation
provisional-income table. The survivor also transitions from drawing both
spouses' SS benefits to drawing only the larger of the two (the survivor
benefit rule under the SSA), and the pension annuity scales by the
joint-and-survivor election fraction.

This module exposes a `Mortality` dataclass that the simulator queries
each year to determine filing status and post-death cash flows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FilingStatus = Literal["mfj", "single"]


@dataclass
class Mortality:
    """Spouse-specific mortality model.

    `year_of_death_a` / `year_of_death_b` are simulation year offsets
    (0 = simulation start, so `year_of_death_a=20` means Spouse A dies
    at the start of the 21st simulation year). `None` = both survive
    to horizon (the deterministic default — backward compatible).

    `pension_survivor_pct` is the survivor election on the cash-balance
    pension annuity. 0.50 = 50% joint-and-survivor (typical default),
    1.00 = 100% J&S (survivor receives the full annuity, but the
    primary-life payment is correspondingly lower; we don't model the
    pre-death reduction here).

    `ss_survivor_keeps_higher` is True per SSA rules: when one spouse
    dies, the survivor continues to receive their own benefit OR a
    survivor benefit equal to the deceased's benefit, whichever is
    larger. They never collect both. We model this by keeping the
    larger of `ssn_monthly_a` and `ssn_monthly_b` post-death and
    zeroing the other.
    """

    year_of_death_a: int | None = None
    year_of_death_b: int | None = None
    pension_survivor_pct: float = 0.50
    ss_survivor_keeps_higher: bool = True

    def alive_a(self, year_offset: int) -> bool:
        return self.year_of_death_a is None or year_offset < self.year_of_death_a

    def alive_b(self, year_offset: int) -> bool:
        return self.year_of_death_b is None or year_offset < self.year_of_death_b

    def both_alive(self, year_offset: int) -> bool:
        return self.alive_a(year_offset) and self.alive_b(year_offset)

    def filing_status(self, year_offset: int) -> FilingStatus:
        return "mfj" if self.both_alive(year_offset) else "single"

    def survivor_label(self, year_offset: int) -> str | None:
        """Return 'a', 'b', or None (both alive or both dead)."""
        a, b = self.alive_a(year_offset), self.alive_b(year_offset)
        if a and b:
            return None
        if a and not b:
            return "a"
        if b and not a:
            return "b"
        return "neither"
