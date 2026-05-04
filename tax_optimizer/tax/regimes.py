"""Tax-regime definitions.

A `TaxRegime` bundles all the bracket / threshold / deduction tables the
federal-tax engine needs, keyed by filing status. This makes it trivial
to swap regimes for stress-testing legislative change.

Three regimes are bundled:

  * `TCJA_EXTENDED` — 2026 brackets assuming TCJA is extended past its
    scheduled 2025 sunset. This is what the optimizer used historically.

  * `PRE_TCJA_2017` — 2017 brackets, untouched by TCJA. Useful for
    "what if Congress had repealed TCJA outright" stress tests.

  * `SUNSET_2026` — TCJA actually expires after 2025. Brackets revert
    to the pre-TCJA structure but with bracket widths inflation-adjusted
    forward (~1.30x the 2017 nominals per JCT projections), and the
    standard deduction roughly halves.

Bracket-table format: list of (lo, hi, rate) tuples where the bracket
applies to taxable income in `[lo, hi)`. Use `math.inf` for the top
bracket's upper bound.

IRMAA-tier-table format: list of (magi_upper_bound, partB_monthly,
partD_monthly) tuples. Surcharges are per enrollee per month.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

Bracket = tuple[float, float, float]
IRMAATier = tuple[float, float, float]


@dataclass(frozen=True)
class TaxRegime:
    """All bracket / threshold / deduction data for a given tax regime.

    The simulator passes the active regime + filing status into
    `federal_tax` and `irmaa_annual_surcharge` each year. Switching
    regimes mid-simulation is supported via the `Config` knobs
    `regime_change_year_offset` + `regime_change_target`.
    """

    name: str

    ord_brackets_mfj: Sequence[Bracket]
    ord_brackets_single: Sequence[Bracket]
    ltcg_brackets_mfj: Sequence[Bracket]
    ltcg_brackets_single: Sequence[Bracket]

    std_deduction_mfj: float
    std_deduction_single: float

    niit_threshold_mfj: float
    niit_threshold_single: float
    niit_rate: float

    irmaa_tiers_mfj: Sequence[IRMAATier]
    irmaa_tiers_single: Sequence[IRMAATier]

    # Provisional-income thresholds for SS taxability (Section 86 IRC).
    # Stable across regimes since 1984; included here for symmetry.
    ss_provisional_mfj: tuple[float, float] = (32_000.0, 44_000.0)
    ss_provisional_single: tuple[float, float] = (25_000.0, 34_000.0)

    def ord_brackets(self, filing_status: str) -> Sequence[Bracket]:
        return self.ord_brackets_mfj if filing_status == "mfj" else self.ord_brackets_single

    def ltcg_brackets(self, filing_status: str) -> Sequence[Bracket]:
        return self.ltcg_brackets_mfj if filing_status == "mfj" else self.ltcg_brackets_single

    def std_deduction(self, filing_status: str) -> float:
        return self.std_deduction_mfj if filing_status == "mfj" else self.std_deduction_single

    def niit_threshold(self, filing_status: str) -> float:
        return self.niit_threshold_mfj if filing_status == "mfj" else self.niit_threshold_single

    def irmaa_tiers(self, filing_status: str) -> Sequence[IRMAATier]:
        return self.irmaa_tiers_mfj if filing_status == "mfj" else self.irmaa_tiers_single

    def ss_provisional(self, filing_status: str) -> tuple[float, float]:
        return self.ss_provisional_mfj if filing_status == "mfj" else self.ss_provisional_single


# ---------------------------------------------------------------------------
# IRMAA tier tables (CMS 2026 published / projected; per-enrollee monthly)
# Single-filer thresholds are roughly half of MFJ for tiers 1-3, but the
# top tier kicks in at $500k single vs $750k MFJ.
# ---------------------------------------------------------------------------
_IRMAA_2026_MFJ: tuple[IRMAATier, ...] = (
    (212_000.0,   0.00,   0.00),
    (266_000.0,  74.00,  13.70),
    (334_000.0, 185.00,  35.30),
    (400_000.0, 295.00,  57.00),
    (750_000.0, 406.00,  78.60),
    (math.inf,  443.90,  85.80),
)

_IRMAA_2026_SINGLE: tuple[IRMAATier, ...] = (
    (106_000.0,   0.00,   0.00),
    (133_000.0,  74.00,  13.70),
    (167_000.0, 185.00,  35.30),
    (200_000.0, 295.00,  57.00),
    (500_000.0, 406.00,  78.60),
    (math.inf,  443.90,  85.80),
)


# ---------------------------------------------------------------------------
# TCJA_EXTENDED  (2026 brackets, assuming TCJA is extended)
# ---------------------------------------------------------------------------
TCJA_EXTENDED: TaxRegime = TaxRegime(
    name="TCJA_extended_2026",
    ord_brackets_mfj=[
        (0,         23_850,    0.10),
        (23_850,    96_950,    0.12),
        (96_950,   206_700,    0.22),
        (206_700,  394_600,    0.24),
        (394_600,  501_050,    0.32),
        (501_050,  751_600,    0.35),
        (751_600,  math.inf,   0.37),
    ],
    ord_brackets_single=[
        (0,         11_925,    0.10),
        (11_925,    48_475,    0.12),
        (48_475,   103_350,    0.22),
        (103_350,  197_300,    0.24),
        (197_300,  250_525,    0.32),
        (250_525,  626_350,    0.35),
        (626_350,  math.inf,   0.37),
    ],
    ltcg_brackets_mfj=[
        (0,        96_700,    0.00),
        (96_700,  600_050,    0.15),
        (600_050, math.inf,   0.20),
    ],
    ltcg_brackets_single=[
        (0,        48_350,    0.00),
        (48_350,  533_400,    0.15),
        (533_400, math.inf,   0.20),
    ],
    std_deduction_mfj=32_200.0,
    std_deduction_single=16_100.0,
    niit_threshold_mfj=250_000.0,
    niit_threshold_single=200_000.0,
    niit_rate=0.038,
    irmaa_tiers_mfj=_IRMAA_2026_MFJ,
    irmaa_tiers_single=_IRMAA_2026_SINGLE,
)


# ---------------------------------------------------------------------------
# PRE_TCJA_2017  (2017 brackets, no inflation adjustment)
# ---------------------------------------------------------------------------
PRE_TCJA_2017: TaxRegime = TaxRegime(
    name="pre_tcja_2017",
    ord_brackets_mfj=[
        (0,          18_650,   0.10),
        (18_650,     75_900,   0.15),
        (75_900,    153_100,   0.25),
        (153_100,   233_350,   0.28),
        (233_350,   416_700,   0.33),
        (416_700,   470_700,   0.35),
        (470_700,   math.inf,  0.396),
    ],
    ord_brackets_single=[
        (0,           9_325,   0.10),
        (9_325,      37_950,   0.15),
        (37_950,     91_900,   0.25),
        (91_900,    191_650,   0.28),
        (191_650,   416_700,   0.33),
        (416_700,   418_400,   0.35),
        (418_400,   math.inf,  0.396),
    ],
    ltcg_brackets_mfj=[
        (0,          75_900,   0.00),
        (75_900,    470_700,   0.15),
        (470_700,   math.inf,  0.20),
    ],
    ltcg_brackets_single=[
        (0,          37_950,   0.00),
        (37_950,    418_400,   0.15),
        (418_400,   math.inf,  0.20),
    ],
    std_deduction_mfj=12_700.0,
    std_deduction_single=6_350.0,
    niit_threshold_mfj=250_000.0,
    niit_threshold_single=200_000.0,
    niit_rate=0.038,
    irmaa_tiers_mfj=_IRMAA_2026_MFJ,
    irmaa_tiers_single=_IRMAA_2026_SINGLE,
)


# ---------------------------------------------------------------------------
# SUNSET_2026  (TCJA expires; pre-TCJA structure inflation-adjusted forward
# to 2026 nominal dollars per JCT extrapolation, ~1.30x 2017 widths.)
# ---------------------------------------------------------------------------
def _scale(brackets: Sequence[Bracket], factor: float) -> list[Bracket]:
    out: list[Bracket] = []
    for lo, hi, rate in brackets:
        new_lo = lo * factor
        new_hi = hi * factor if math.isfinite(hi) else math.inf
        out.append((new_lo, new_hi, rate))
    return out


_SUNSET_FACTOR = 1.30  # cumulative CPI 2017->2026 per BLS projection

SUNSET_2026: TaxRegime = TaxRegime(
    name="tcja_sunset_2026",
    ord_brackets_mfj=_scale(PRE_TCJA_2017.ord_brackets_mfj, _SUNSET_FACTOR),
    ord_brackets_single=_scale(PRE_TCJA_2017.ord_brackets_single, _SUNSET_FACTOR),
    ltcg_brackets_mfj=_scale(PRE_TCJA_2017.ltcg_brackets_mfj, _SUNSET_FACTOR),
    ltcg_brackets_single=_scale(PRE_TCJA_2017.ltcg_brackets_single, _SUNSET_FACTOR),
    std_deduction_mfj=PRE_TCJA_2017.std_deduction_mfj * _SUNSET_FACTOR,
    std_deduction_single=PRE_TCJA_2017.std_deduction_single * _SUNSET_FACTOR,
    niit_threshold_mfj=250_000.0,
    niit_threshold_single=200_000.0,
    niit_rate=0.038,
    irmaa_tiers_mfj=_IRMAA_2026_MFJ,
    irmaa_tiers_single=_IRMAA_2026_SINGLE,
)
