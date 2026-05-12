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
from dataclasses import dataclass, replace
from typing import Sequence

Bracket = tuple[float, float, float]
IRMAATier = tuple[float, float, float]


def _scale_brackets(brackets: Sequence[Bracket], factor: float) -> list[Bracket]:
    out: list[Bracket] = []
    for lo, hi, rate in brackets:
        new_lo = lo * factor
        new_hi = hi * factor if math.isfinite(hi) else math.inf
        out.append((new_lo, new_hi, rate))
    return out


def _scale_irmaa(tiers: Sequence[IRMAATier], factor: float) -> list[IRMAATier]:
    """Scale both the MAGI threshold AND the surcharge dollar amounts.

    In real life IRMAA MAGI thresholds index annually (CMS publishes new
    boundaries each year); the Part B / Part D surcharge dollars also
    grow each year, faster than CPI on average (medical inflation). For
    a clean projection we scale both at the same rate so real IRMAA stays
    roughly constant — under-estimating the medical-inflation premium
    growth, but avoiding the much-larger error of leaving the 2026
    surcharges flat-nominal across a 30-year horizon.
    """
    out: list[IRMAATier] = []
    for cap, partB, partD in tiers:
        new_cap = cap * factor if math.isfinite(cap) else math.inf
        out.append((new_cap, partB * factor, partD * factor))
    return out


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

    # Age-65+ additional standard deduction — IRC §63(f). Applied per
    # spouse 65+ on top of the base std deduction. The amounts differ
    # by filing status (MFJ uses the smaller per-spouse number; an
    # unmarried 65+ filer gets a larger amount).
    senior_std_deduction_mfj: float = 0.0
    senior_std_deduction_single: float = 0.0

    # 2025 OBBBA "senior bonus" — temporary additional $6,000/filer for
    # filers 65+ during tax years 2025–2028 (sunsets 2028 per current
    # statute). Stacks on top of `senior_std_deduction_*`. Set to 0 on
    # regimes where this bonus shouldn't apply (e.g. PRE_TCJA_2017,
    # which is a "what if" pre-OBBBA stress test).
    obbba_senior_bonus_per_filer: float = 0.0
    obbba_senior_bonus_start_year: int = 2025
    obbba_senior_bonus_end_year: int = 2028

    # ---- AMT (Alternative Minimum Tax, IRC §55) parameters ----
    # Set `amt_exemption_mfj = math.inf` to disable AMT for the regime.
    # `amt_std_deduction_addback=True` enables the pre-TCJA behavior of
    # adding the standard deduction back to AMTI (TCJA allowed std ded
    # in AMT through 2025; the sunset reverts this). LTCG/QDIV preserve
    # their preferential rates under AMT — only the ordinary portion of
    # AMTI gets the 26% / 28% AMT rate structure.
    amt_exemption_mfj: float = math.inf
    amt_exemption_single: float = math.inf
    amt_phaseout_start_mfj: float = math.inf
    amt_phaseout_start_single: float = math.inf
    amt_phaseout_rate: float = 0.25
    amt_28pct_threshold_mfj: float = math.inf
    amt_28pct_threshold_single: float = math.inf
    amt_rate_low: float = 0.26
    amt_rate_high: float = 0.28
    amt_std_deduction_addback: bool = False

    def ord_brackets(self, filing_status: str) -> Sequence[Bracket]:
        return self.ord_brackets_mfj if filing_status == "mfj" else self.ord_brackets_single

    def ltcg_brackets(self, filing_status: str) -> Sequence[Bracket]:
        return self.ltcg_brackets_mfj if filing_status == "mfj" else self.ltcg_brackets_single

    def std_deduction(self, filing_status: str) -> float:
        return self.std_deduction_mfj if filing_status == "mfj" else self.std_deduction_single

    def senior_std_deduction(self, filing_status: str) -> float:
        """Age-65+ additional std deduction PER FILER 65+ (§63(f))."""
        return (
            self.senior_std_deduction_mfj
            if filing_status == "mfj"
            else self.senior_std_deduction_single
        )

    def effective_std_deduction(
        self,
        filing_status: str,
        *,
        n_seniors_65plus: int = 0,
        calendar_year: int | None = None,
    ) -> float:
        """Total standard deduction including age-65+ add-ons and any
        time-limited "senior bonus" (e.g. 2025 OBBBA §70103, $6,000/filer
        65+ for tax years 2025–2028).

        Pass ``n_seniors_65plus = (alive_a and age_a >= 65)
        + (alive_b and age_b >= 65)``. ``calendar_year`` enables/disables
        the OBBBA bonus window; leave ``None`` to skip the bonus.
        """
        base = self.std_deduction(filing_status)
        senior = self.senior_std_deduction(filing_status) * max(0, n_seniors_65plus)
        bonus = 0.0
        if (
            self.obbba_senior_bonus_per_filer > 0
            and calendar_year is not None
            and self.obbba_senior_bonus_start_year
            <= calendar_year
            <= self.obbba_senior_bonus_end_year
            and n_seniors_65plus > 0
        ):
            bonus = self.obbba_senior_bonus_per_filer * n_seniors_65plus
        return base + senior + bonus

    def niit_threshold(self, filing_status: str) -> float:
        return self.niit_threshold_mfj if filing_status == "mfj" else self.niit_threshold_single

    def irmaa_tiers(self, filing_status: str) -> Sequence[IRMAATier]:
        return self.irmaa_tiers_mfj if filing_status == "mfj" else self.irmaa_tiers_single

    def ss_provisional(self, filing_status: str) -> tuple[float, float]:
        return self.ss_provisional_mfj if filing_status == "mfj" else self.ss_provisional_single

    def amt_exemption(self, filing_status: str) -> float:
        return (
            self.amt_exemption_mfj
            if filing_status == "mfj"
            else self.amt_exemption_single
        )

    def amt_phaseout_start(self, filing_status: str) -> float:
        return (
            self.amt_phaseout_start_mfj
            if filing_status == "mfj"
            else self.amt_phaseout_start_single
        )

    def amt_28pct_threshold(self, filing_status: str) -> float:
        return (
            self.amt_28pct_threshold_mfj
            if filing_status == "mfj"
            else self.amt_28pct_threshold_single
        )

    def inflated(self, factor: float) -> "TaxRegime":
        """Return a copy with widely-indexed thresholds scaled by `factor`.

        Scales: ordinary brackets, LTCG brackets, the standard deduction,
        and IRMAA tier thresholds + surcharge dollar amounts.

        Does NOT scale: the NIIT MAGI threshold (statutory $250k MFJ /
        $200k single, unindexed since 2013) or the SS-provisional-income
        thresholds ($32k/$44k MFJ, unindexed since 1993). Both are
        intentionally left flat-nominal to match the IRC.

        Use case: in a long-horizon simulation, real IRS practice is to
        index brackets annually to chained CPI. Leaving them at the
        regime's quoted-year nominals while income inflates produces a
        slow, stealth bracket-creep that biases conversion-sizing and
        all post-deduction tax math. This helper rebuilds the regime
        per year with `factor = (1 + bracket_indexing_rate) ** year_offset`.
        """
        if factor == 1.0:
            return self
        return replace(
            self,
            ord_brackets_mfj=_scale_brackets(self.ord_brackets_mfj, factor),
            ord_brackets_single=_scale_brackets(self.ord_brackets_single, factor),
            ltcg_brackets_mfj=_scale_brackets(self.ltcg_brackets_mfj, factor),
            ltcg_brackets_single=_scale_brackets(self.ltcg_brackets_single, factor),
            std_deduction_mfj=self.std_deduction_mfj * factor,
            std_deduction_single=self.std_deduction_single * factor,
            senior_std_deduction_mfj=self.senior_std_deduction_mfj * factor,
            senior_std_deduction_single=self.senior_std_deduction_single * factor,
            obbba_senior_bonus_per_filer=self.obbba_senior_bonus_per_filer * factor,
            irmaa_tiers_mfj=_scale_irmaa(self.irmaa_tiers_mfj, factor),
            irmaa_tiers_single=_scale_irmaa(self.irmaa_tiers_single, factor),
            amt_exemption_mfj=(
                self.amt_exemption_mfj * factor
                if math.isfinite(self.amt_exemption_mfj)
                else math.inf
            ),
            amt_exemption_single=(
                self.amt_exemption_single * factor
                if math.isfinite(self.amt_exemption_single)
                else math.inf
            ),
            amt_phaseout_start_mfj=(
                self.amt_phaseout_start_mfj * factor
                if math.isfinite(self.amt_phaseout_start_mfj)
                else math.inf
            ),
            amt_phaseout_start_single=(
                self.amt_phaseout_start_single * factor
                if math.isfinite(self.amt_phaseout_start_single)
                else math.inf
            ),
            amt_28pct_threshold_mfj=(
                self.amt_28pct_threshold_mfj * factor
                if math.isfinite(self.amt_28pct_threshold_mfj)
                else math.inf
            ),
            amt_28pct_threshold_single=(
                self.amt_28pct_threshold_single * factor
                if math.isfinite(self.amt_28pct_threshold_single)
                else math.inf
            ),
        )


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
    # §63(f) 65+ add-ons (2026 nominals): $1,600/spouse MFJ, $2,000/filer
    # single. Indexed forward by `cfg.bracket_indexing_rate` via inflated().
    senior_std_deduction_mfj=1_600.0,
    senior_std_deduction_single=2_000.0,
    # 2025 OBBBA §70103 senior bonus: +$6,000 per filer 65+ for tax
    # years 2025–2028 (sunsets at end of 2028 under current statute).
    obbba_senior_bonus_per_filer=6_000.0,
    niit_threshold_mfj=250_000.0,
    niit_threshold_single=200_000.0,
    niit_rate=0.038,
    irmaa_tiers_mfj=_IRMAA_2026_MFJ,
    irmaa_tiers_single=_IRMAA_2026_SINGLE,
    # TCJA-era AMT (2026 projected): very high exemption + 1.25M MFJ
    # phaseout start make AMT mostly dormant for retirees. Std ded is
    # allowed in AMT under TCJA.
    amt_exemption_mfj=137_000.0,
    amt_exemption_single=88_100.0,
    amt_phaseout_start_mfj=1_252_700.0,
    amt_phaseout_start_single=626_350.0,
    amt_28pct_threshold_mfj=239_100.0,
    amt_28pct_threshold_single=239_100.0,
    amt_std_deduction_addback=False,
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
    # 2017 §63(f) 65+ add-ons (pre-TCJA): $1,250/spouse MFJ, $1,550/filer
    # single. Sunset / inflated by `inflated()` if needed.
    senior_std_deduction_mfj=1_250.0,
    senior_std_deduction_single=1_550.0,
    # No OBBBA bonus under PRE_TCJA_2017 (pre-OBBBA stress test).
    obbba_senior_bonus_per_filer=0.0,
    niit_threshold_mfj=250_000.0,
    niit_threshold_single=200_000.0,
    niit_rate=0.038,
    irmaa_tiers_mfj=_IRMAA_2026_MFJ,
    irmaa_tiers_single=_IRMAA_2026_SINGLE,
    # 2017 AMT: lower exemption, phaseout starts at modest income,
    # std ded MUST be added back to AMTI (pre-TCJA AMT mechanics).
    amt_exemption_mfj=84_500.0,
    amt_exemption_single=54_300.0,
    amt_phaseout_start_mfj=160_900.0,
    amt_phaseout_start_single=120_700.0,
    amt_28pct_threshold_mfj=187_800.0,
    amt_28pct_threshold_single=187_800.0,
    amt_std_deduction_addback=True,
)


# ---------------------------------------------------------------------------
# SUNSET_2026  (TCJA expires; pre-TCJA structure inflation-adjusted forward
# to 2026 nominal dollars per JCT extrapolation, ~1.30x 2017 widths.)
# ---------------------------------------------------------------------------
_SUNSET_FACTOR = 1.30  # cumulative CPI 2017->2026 per BLS projection

SUNSET_2026: TaxRegime = TaxRegime(
    name="tcja_sunset_2026",
    ord_brackets_mfj=_scale_brackets(PRE_TCJA_2017.ord_brackets_mfj, _SUNSET_FACTOR),
    ord_brackets_single=_scale_brackets(PRE_TCJA_2017.ord_brackets_single, _SUNSET_FACTOR),
    ltcg_brackets_mfj=_scale_brackets(PRE_TCJA_2017.ltcg_brackets_mfj, _SUNSET_FACTOR),
    ltcg_brackets_single=_scale_brackets(PRE_TCJA_2017.ltcg_brackets_single, _SUNSET_FACTOR),
    std_deduction_mfj=PRE_TCJA_2017.std_deduction_mfj * _SUNSET_FACTOR,
    std_deduction_single=PRE_TCJA_2017.std_deduction_single * _SUNSET_FACTOR,
    # Pre-TCJA §63(f) inflated forward to 2026.
    senior_std_deduction_mfj=PRE_TCJA_2017.senior_std_deduction_mfj * _SUNSET_FACTOR,
    senior_std_deduction_single=PRE_TCJA_2017.senior_std_deduction_single * _SUNSET_FACTOR,
    # OBBBA senior bonus sunsets at end of 2028 — it persists into the
    # sunset regime through its window (2025–2028), then expires.
    obbba_senior_bonus_per_filer=6_000.0,
    niit_threshold_mfj=250_000.0,
    niit_threshold_single=200_000.0,
    niit_rate=0.038,
    irmaa_tiers_mfj=_IRMAA_2026_MFJ,
    irmaa_tiers_single=_IRMAA_2026_SINGLE,
    # AMT under sunset: pre-TCJA exemption + thresholds inflated forward
    # by _SUNSET_FACTOR. With phaseout starting at ~$209k MFJ, AMT can
    # bite during large Roth-conversion years — the entire point of
    # modeling it. Std ded add-back returns under pre-TCJA AMT mechanics.
    amt_exemption_mfj=PRE_TCJA_2017.amt_exemption_mfj * _SUNSET_FACTOR,
    amt_exemption_single=PRE_TCJA_2017.amt_exemption_single * _SUNSET_FACTOR,
    amt_phaseout_start_mfj=PRE_TCJA_2017.amt_phaseout_start_mfj * _SUNSET_FACTOR,
    amt_phaseout_start_single=PRE_TCJA_2017.amt_phaseout_start_single * _SUNSET_FACTOR,
    amt_28pct_threshold_mfj=PRE_TCJA_2017.amt_28pct_threshold_mfj * _SUNSET_FACTOR,
    amt_28pct_threshold_single=PRE_TCJA_2017.amt_28pct_threshold_single * _SUNSET_FACTOR,
    amt_std_deduction_addback=True,
)
