"""State income tax engine.

Federal-only optimization understates tax burden materially in CA / NY /
OR / NJ / MA / HI (and slightly in CT / IL / MD), and the *direction* of
the bias is not uniform across strategies — which means leaving state
tax out of the model can flip the optimal Roth-conversion plan, the
optimal claim-age decision, and the optimal "stay vs move" framing.

We don't try to model all 50 states. The bundled regimes cover the
~15 states whose absence-or-presence dominates the optimization
answer for a typical user; everything else folds into ``STATELESS``
(no income tax) which is also correct for FL / TX / WA / NV / TN /
NH / AK / SD / WY.

Bundled presets (2024 nominals; index forward via
``cfg.bracket_indexing_rate`` like the federal regime):

  * ``STATELESS``   — flat 0%; correct for FL / TX / WA / NV / TN /
                      NH / AK / SD / WY
  * ``CA``          — progressive brackets, LTCG at ordinary rate,
                      SS exempt, HSA NOT deductible at state level
  * ``NY``          — progressive brackets, LTCG at ordinary rate,
                      SS exempt, $20k per-filer retirement-income
                      exclusion at 59½+
  * ``IL``          — flat 4.95%, **all retirement income exempt**
                      (huge difference vs CA/NY for retirees)
  * ``MA``          — flat 5%, SS exempt, IRA/pension taxed normally
                      (the 9% surtax > $1M is intentionally not modeled)

States with material differences not in this list (OR / NJ / HI / MD)
can be added by following the ``CA`` / ``NY`` shape; the simulator
plumbing is regime-agnostic.

Custom states: pass any ``StateTaxRegime`` instance to
``cfg.state_regime``. The simulator inflation-indexes the brackets
each year using the same ``bracket_indexing_rate`` as federal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

from .regimes import Bracket, _scale_brackets


@dataclass(frozen=True)
class StateTaxRegime:
    """All bracket / threshold / deduction data for a state regime.

    For flat-tax states (IL, MA), pass a single bracket like
    ``[(0, math.inf, 0.0495)]``. For ``STATELESS`` use the empty
    bracket list — `state_tax` returns 0 immediately.
    """

    name: str

    # Ordinary-income brackets. Empty = no state income tax.
    ord_brackets_mfj: Sequence[Bracket] = ()
    ord_brackets_single: Sequence[Bracket] = ()

    std_deduction_mfj: float = 0.0
    std_deduction_single: float = 0.0

    # LTCG bracket structure. ``None`` ⇒ no preferential state-LTCG
    # rate, so LTCG is taxed as ordinary income (CA, NY, MA, IL all
    # work this way; the federal-style preferential rate is the
    # exception, not the rule, at the state level).
    ltcg_brackets_mfj: Sequence[Bracket] | None = None
    ltcg_brackets_single: Sequence[Bracket] | None = None

    # SS taxability at state level. Most states fully exempt SS;
    # `0.0` ⇒ exempt (CA, NY, IL, MA, et al). Set `0.85` to mimic
    # the federal max-taxable rate for the small set of states
    # that do tax SS (CT, MN, NE, ND, RI, WV — partially in some
    # of these).
    ss_taxable_fraction: float = 0.0

    # Retirement-income exclusion. Two flavors:
    #   * `retirement_full_exclusion=True`  — IL: pension, IRA, Roth
    #     conversion, and SS are all fully excluded from state tax.
    #   * `retirement_exclusion_per_filer=X` — NY: each filer (per
    #     spouse) excludes up to X dollars/yr of pension + IRA + Roth
    #     conversion income at age `min_age` and above.
    retirement_full_exclusion: bool = False
    retirement_exclusion_per_filer: float = 0.0
    retirement_exclusion_min_age: int = 59

    # HSA conformity. CA and NJ do NOT conform to federal HSA — HSA
    # contributions are NOT pre-tax-deductible at the state level
    # (and HSA earnings are state-taxable too, which we don't model).
    # Default `True` ⇒ HSA contribs reduce state taxable income too.
    hsa_deductible: bool = True

    # State Disability Insurance (SDI / VPDI). Employee withholding;
    # cash-flow cost only, doesn't affect state taxable income.
    # CA: 1.1% uncapped (SB 951 removed the wage cap effective 2024).
    # Other bundled states have no meaningful SDI. Set ``sdi_rate=0``
    # to disable (default).
    sdi_rate: float = 0.0
    sdi_wage_cap: float = math.inf

    # Optional per-year SDI rate schedule (calendar year → rate). When
    # provided, the simulator looks up the rate by `cfg.start_year +
    # year_offset` and falls back to the closest earlier year (or to
    # ``sdi_rate`` if the calendar year predates the schedule). CA's
    # EDD-published schedule diverges from the static 1.1% default
    # by ±10 bps annually (1.1% / 1.2% / 0.9% for 2024 / 2025 / 2026
    # — see CA EDD DB-1101) so this matters for households who care
    # about a specific year's withholding.
    sdi_rate_schedule: dict[int, float] | None = None

    def ord_brackets(self, filing_status: str) -> Sequence[Bracket]:
        return (
            self.ord_brackets_mfj
            if filing_status == "mfj"
            else self.ord_brackets_single
        )

    def ltcg_brackets(self, filing_status: str) -> Sequence[Bracket] | None:
        return (
            self.ltcg_brackets_mfj
            if filing_status == "mfj"
            else self.ltcg_brackets_single
        )

    def std_deduction(self, filing_status: str) -> float:
        return (
            self.std_deduction_mfj
            if filing_status == "mfj"
            else self.std_deduction_single
        )

    def has_tax(self) -> bool:
        return bool(self.ord_brackets_mfj or self.ord_brackets_single)

    def effective_sdi_rate(self, calendar_year: int) -> float:
        """Resolve the SDI rate for a given calendar year.

        When ``sdi_rate_schedule`` is set, picks the schedule entry for
        ``calendar_year`` (or the largest year ≤ ``calendar_year`` if
        the exact year is missing — i.e. the published rate "carries
        forward" until a future year publishes its own value). Falls
        back to the static ``sdi_rate`` if the calendar year predates
        every scheduled year.
        """
        if not self.sdi_rate_schedule:
            return self.sdi_rate
        eligible = [y for y in self.sdi_rate_schedule if y <= calendar_year]
        if not eligible:
            return self.sdi_rate
        return self.sdi_rate_schedule[max(eligible)]

    def inflated(self, factor: float) -> "StateTaxRegime":
        """Mirror of `TaxRegime.inflated`. Scales brackets + std deduction
        + retirement exclusion. Many state thresholds index annually in
        real life (CA, OR), some don't (IL, MA flat rates have no
        thresholds). We index everything for consistency."""
        if factor == 1.0 or not self.has_tax():
            return self
        return replace(
            self,
            ord_brackets_mfj=_scale_brackets(self.ord_brackets_mfj, factor),
            ord_brackets_single=_scale_brackets(
                self.ord_brackets_single, factor
            ),
            ltcg_brackets_mfj=(
                _scale_brackets(self.ltcg_brackets_mfj, factor)
                if self.ltcg_brackets_mfj
                else None
            ),
            ltcg_brackets_single=(
                _scale_brackets(self.ltcg_brackets_single, factor)
                if self.ltcg_brackets_single
                else None
            ),
            std_deduction_mfj=self.std_deduction_mfj * factor,
            std_deduction_single=self.std_deduction_single * factor,
            retirement_exclusion_per_filer=(
                self.retirement_exclusion_per_filer * factor
            ),
        )


# ---------------------------------------------------------------------------
# Tax computation
# ---------------------------------------------------------------------------


def _bracket_tax(amount: float, brackets: Sequence[Bracket]) -> float:
    tax = 0.0
    for lo, hi, rate in brackets:
        if amount <= lo:
            break
        tax += (min(amount, hi) - lo) * rate
    return tax


def state_tax(
    *,
    regime: StateTaxRegime,
    filing_status: str,
    wages_box1: float,
    interest: float,
    ordinary_div: float,
    qualified_div: float,
    ltcg: float,
    pension: float,
    pretax_withdrawal: float,
    roth_conversion: float,
    social_security: float,
    ss_taxable_federal: float,
    annuity_taxable: float = 0.0,
    hsa_contrib: float = 0.0,
    age_a: int = 0,
    age_b: int = 0,
    alive_a: bool = True,
    alive_b: bool = True,
    # Optional per-spouse distribution breakdowns. When provided, the
    # NY-style "per-filer" retirement exclusion is applied against each
    # spouse's OWN distributions (the correct §612(c)(3-a) treatment)
    # instead of being pooled. When omitted, falls back to the
    # combined-pool approximation for back-compat.
    pension_per_spouse: tuple[float, float] | None = None,
    pretax_per_spouse: tuple[float, float] | None = None,
    roth_conv_per_spouse: tuple[float, float] | None = None,
    annuity_per_spouse: tuple[float, float] | None = None,
) -> dict:
    """Compute state income tax for one year.

    Inputs use the same names as `federal.federal_tax`. Returns
    ``{state_tax, state_taxable_income, state_marginal}``.

    Mechanics that diverge from federal:

      * **HSA add-back.** If `regime.hsa_deductible=False` (CA, NJ),
        we add `hsa_contrib` back to ordinary income because the
        federal Box-1 wage figure already pre-deducted it. This is
        the only conformity adjustment we model — federal vs state
        AGI typically differs only at the margins for the regimes
        we ship.
      * **SS taxation.** Most states fully exempt SS. We multiply
        the federal `ss_taxable` figure by `regime.ss_taxable_fraction`
        and rebate the rest from ordinary income.
      * **Retirement exclusion.** IL fully exempts pension + IRA +
        Roth conversion + SS + annuity. NY exempts $20k per filer at
        59½+ against each spouse's own pension + IRA + Roth-conv +
        annuity income (per IRC §61(a)(9) annuity is gross income;
        NY Tax Law §612(c)(3-a) treats commercial annuity income
        identically to pension for the $20k retirement exclusion).
        Per-spouse exclusion is applied against the supplied
        ``pension_per_spouse`` / ``pretax_per_spouse`` /
        ``roth_conv_per_spouse`` / ``annuity_per_spouse`` tuples.
      * **Annuity income (`annuity_taxable`).** Treated as ordinary
        income at the state level (matches federal §61(a)(9) /
        §72(b) treatment). Eligible for the IL full exclusion and
        the NY per-filer exclusion.
      * **LTCG.** Most states tax LTCG as ordinary; if the regime
        provides explicit LTCG brackets, we honor them.
    """
    if not regime.has_tax():
        return {
            "state_tax": 0.0,
            "state_taxable_income": 0.0,
            "state_marginal": 0.0,
        }

    state_ss_taxable = ss_taxable_federal * regime.ss_taxable_fraction

    if regime.retirement_full_exclusion:
        # IL: pension + IRA / Roth-conv + annuity + SS all excluded
        # from state taxable income at any age. (IL allows this
        # without a minimum-age gate; the others we ship require
        # 59½+.) We exclude `state_ss_taxable` (the amount we'd have
        # added to ord_income), not `ss_taxable_federal`, so a
        # regime that combines `retirement_full_exclusion=True` with
        # a non-zero `ss_taxable_fraction` zeros out cleanly without
        # double-counting. Annuity income is included per IL
        # 35 ILCS 5/203(a)(2)(F): "any amount of distribution…from a
        # qualified retirement plan…or from an annuity contract."
        retirement_excluded = (
            pension
            + pretax_withdrawal
            + roth_conversion
            + annuity_taxable
            + state_ss_taxable
        )
    elif regime.retirement_exclusion_per_filer > 0:
        # NY-style: $X per spouse at age min_age+, applied per-spouse
        # against each spouse's own pension + IRA + Roth-conv +
        # annuity (NOT SS). Per NY Tax Law §612(c)(3-a) the exclusion
        # is granted to each filer against their own distributions —
        # a spouse with $0 of qualifying distributions cannot pass
        # their unused $20k to the higher-earner spouse. Commercial
        # annuity income is eligible (the statute lists "pensions and
        # annuities…received…from a private employer" as one of the
        # qualifying categories alongside IRA distributions).
        #
        # When per-spouse breakdowns are supplied, apply the cap
        # per-recipient and sum. When omitted, fall back to the
        # combined-pool approximation (used by older callers / tests).
        per = regime.retirement_exclusion_per_filer
        a_eligible = alive_a and age_a >= regime.retirement_exclusion_min_age
        b_eligible = alive_b and age_b >= regime.retirement_exclusion_min_age
        if (
            pension_per_spouse is not None
            or pretax_per_spouse is not None
            or roth_conv_per_spouse is not None
            or annuity_per_spouse is not None
        ):
            pa, pb = pension_per_spouse or (0.0, 0.0)
            wa, wb = pretax_per_spouse or (0.0, 0.0)
            ra, rb = roth_conv_per_spouse or (0.0, 0.0)
            aa, ab = annuity_per_spouse or (0.0, 0.0)
            a_pool = pa + wa + ra + aa if a_eligible else 0.0
            b_pool = pb + wb + rb + ab if b_eligible else 0.0
            retirement_excluded = min(per, a_pool) + min(per, b_pool)
        else:
            n_filers = int(a_eligible) + int(b_eligible)
            max_excl = per * n_filers
            retirement_pool = (
                pension + pretax_withdrawal + roth_conversion + annuity_taxable
            )
            retirement_excluded = min(max_excl, retirement_pool)
    else:
        retirement_excluded = 0.0

    # State HSA add-back (CA / NJ): if the state doesn't conform,
    # the federal wages_box1 figure has already pre-deducted HSA
    # contributions, so we add them back to recover the state-level
    # ordinary income.
    hsa_addback = 0.0 if regime.hsa_deductible else max(0.0, hsa_contrib)

    ord_income = (
        wages_box1
        + hsa_addback
        + interest
        + ordinary_div
        + pension
        + annuity_taxable
        + pretax_withdrawal
        + roth_conversion
        + state_ss_taxable
    )

    preferential = qualified_div + ltcg
    if regime.ltcg_brackets(filing_status) is None:
        # No preferential LTCG rate at state level → tax as ordinary.
        ord_income += preferential
        preferential = 0.0

    ord_income = max(0.0, ord_income - retirement_excluded)

    deduction = regime.std_deduction(filing_status)
    taxable_total = max(0.0, ord_income + preferential - deduction)
    taxable_ordinary = max(0.0, taxable_total - preferential)
    taxable_pref = max(0.0, taxable_total - taxable_ordinary)

    ord_tax = _bracket_tax(taxable_ordinary, regime.ord_brackets(filing_status))

    ltcg_tax = 0.0
    if regime.ltcg_brackets(filing_status) is not None:
        cursor = taxable_ordinary
        remaining = taxable_pref
        for lo, hi, rate in regime.ltcg_brackets(filing_status):
            if remaining <= 0:
                break
            room = max(0.0, hi - max(lo, cursor))
            slab = min(remaining, room)
            if slab > 0:
                ltcg_tax += slab * rate
                remaining -= slab
                cursor += slab

    total = ord_tax + ltcg_tax
    # State marginal rate at the top of the ordinary stack (the rate
    # the next dollar of pretax withdrawal or Roth conversion would
    # face). Mirror the federal helper (`tax/federal._marginal_rate`):
    # seed with the first slab's rate so that zero ordinary income
    # returns the first non-zero state rate (CA's 1%, NY's 4%, ...)
    # instead of 0.0, and use ``>=`` on the lower bound so an income
    # that lands *exactly* on a bracket boundary reports the rate the
    # next dollar would face. Pre-fix `taxable_ordinary == 0` returned
    # 0.0 regardless of state, which silently underestimated the
    # marginal cost of the *next* dollar in zero-income gap years
    # (e.g. an early-retirement bracket-fill conversion year).
    if regime.ord_brackets(filing_status):
        marginal = regime.ord_brackets(filing_status)[0][2]
    else:
        marginal = 0.0
    for lo, _hi, rate in regime.ord_brackets(filing_status):
        if taxable_ordinary >= lo:
            marginal = rate

    return {
        "state_tax": total,
        "state_taxable_income": taxable_total,
        "state_marginal": marginal,
    }


# ---------------------------------------------------------------------------
# Bundled regimes
# ---------------------------------------------------------------------------


STATELESS: StateTaxRegime = StateTaxRegime(name="stateless")


# California 2024 MFJ brackets (FTB published). The 1% Mental Health
# Services Tax (Prop 63) applies on income above $1M, raising the
# effective rate by 1% in those brackets.
_CA_MFJ: Sequence[Bracket] = [
    (0,           20_824,    0.0100),
    (20_824,      49_368,    0.0200),
    (49_368,      77_918,    0.0400),
    (77_918,     108_162,    0.0600),
    (108_162,    136_700,    0.0800),
    (136_700,    698_274,    0.0930),
    (698_274,    837_922,    0.1030),
    (837_922,  1_000_000,    0.1130),
    (1_000_000, 1_396_542,   0.1230),  # 11.3% + 1% MH surcharge
    (1_396_542, math.inf,    0.1330),  # 12.3% + 1% MH surcharge
]
_CA_SINGLE: Sequence[Bracket] = [
    (0,           10_412,    0.0100),
    (10_412,      24_684,    0.0200),
    (24_684,      38_959,    0.0400),
    (38_959,      54_081,    0.0600),
    (54_081,      68_350,    0.0800),
    (68_350,     349_137,    0.0930),
    (349_137,    418_961,    0.1030),
    (418_961,    698_271,    0.1130),
    (698_271,  1_000_000,    0.1230),  # 12.3% base top rate
    (1_000_000, math.inf,    0.1330),  # 12.3% + 1% MH surcharge
]

CA: StateTaxRegime = StateTaxRegime(
    name="CA",
    ord_brackets_mfj=_CA_MFJ,
    ord_brackets_single=_CA_SINGLE,
    std_deduction_mfj=11_080.0,
    std_deduction_single=5_540.0,
    # No preferential LTCG rate; LTCG taxed at ordinary state rate.
    ltcg_brackets_mfj=None,
    ltcg_brackets_single=None,
    # CA does not tax Social Security.
    ss_taxable_fraction=0.0,
    # CA does NOT conform to federal HSA — HSA contribs aren't
    # deductible at the state level.
    hsa_deductible=False,
    # CA SDI: 1.1% of all wages, uncapped since 2024 (SB 951). The
    # year-by-year schedule below tracks the EDD-published rates;
    # `sdi_rate` itself stays at 1.1% for back-compat with callers
    # that don't pass a calendar year.
    sdi_rate=0.011,
    sdi_wage_cap=math.inf,
    # CA EDD DB-1101 (and EDD's annual rate announcement letters):
    #   2024: 1.1% (post-SB-951, uncapped)
    #   2025: 1.2% (EDD 2024 announcement)
    #   2026: 0.9% (EDD 2025 announcement)
    # 2027+ defaults to the most recent published year.
    sdi_rate_schedule={
        2024: 0.011,
        2025: 0.012,
        2026: 0.009,
    },
)


# New York 2024 MFJ brackets.
_NY_MFJ: Sequence[Bracket] = [
    (0,            17_150,    0.0400),
    (17_150,       23_600,    0.0450),
    (23_600,       27_900,    0.0525),
    (27_900,      161_550,    0.0550),
    (161_550,     323_200,    0.0600),
    (323_200,   2_155_350,    0.0685),
    (2_155_350, 5_000_000,    0.0965),
    (5_000_000, 25_000_000,   0.1030),
    (25_000_000, math.inf,    0.1090),
]
_NY_SINGLE: Sequence[Bracket] = [
    (0,             8_500,    0.0400),
    (8_500,        11_700,    0.0450),
    (11_700,       13_900,    0.0525),
    (13_900,       80_650,    0.0550),
    (80_650,      215_400,    0.0600),
    (215_400,   1_077_550,    0.0685),
    (1_077_550, 5_000_000,    0.0965),
    (5_000_000, 25_000_000,   0.1030),
    (25_000_000, math.inf,    0.1090),
]

NY: StateTaxRegime = StateTaxRegime(
    name="NY",
    ord_brackets_mfj=_NY_MFJ,
    ord_brackets_single=_NY_SINGLE,
    std_deduction_mfj=16_050.0,
    std_deduction_single=8_000.0,
    ltcg_brackets_mfj=None,
    ltcg_brackets_single=None,
    ss_taxable_fraction=0.0,
    # NY exempts up to $20k/filer of pension + IRA + Roth-conv at 59½+.
    # Integer age 60 is the conservative approximation: someone who is 59
    # at year-start is only 59.5 mid-year, so they don't qualify for the
    # full year. Using 60 avoids granting the exclusion a year early.
    retirement_exclusion_per_filer=20_000.0,
    retirement_exclusion_min_age=60,
)


# Illinois — flat 4.95% on most income, but ALL retirement
# distributions (pension, IRA, 401k, Roth conversion, SS) exempt.
IL: StateTaxRegime = StateTaxRegime(
    name="IL",
    ord_brackets_mfj=[(0, math.inf, 0.0495)],
    ord_brackets_single=[(0, math.inf, 0.0495)],
    # IL doesn't really have a "standard deduction" — they use a
    # personal exemption ($2,775/filer 2024). Approximate as a
    # married-filing-jointly std deduction of 2 × that.
    std_deduction_mfj=5_550.0,
    std_deduction_single=2_775.0,
    ltcg_brackets_mfj=None,
    ltcg_brackets_single=None,
    ss_taxable_fraction=0.0,
    retirement_full_exclusion=True,
)


# Massachusetts — flat 5% on most income, SS exempt. (The 9% MA
# surtax on income > $1M is intentionally not modeled.)
MA: StateTaxRegime = StateTaxRegime(
    name="MA",
    ord_brackets_mfj=[(0, math.inf, 0.0500)],
    ord_brackets_single=[(0, math.inf, 0.0500)],
    # MA personal exemption $4,400/spouse approximated as std dn.
    std_deduction_mfj=8_800.0,
    std_deduction_single=4_400.0,
    ltcg_brackets_mfj=None,
    ltcg_brackets_single=None,
    ss_taxable_fraction=0.0,
)


_REGIMES: dict[str, StateTaxRegime] = {
    "stateless": STATELESS,
    "none": STATELESS,
    "ca": CA,
    "ny": NY,
    "il": IL,
    "ma": MA,
}


def lookup(name: str) -> StateTaxRegime:
    """Look up a bundled state regime by case-insensitive abbreviation."""
    key = name.strip().lower()
    if key not in _REGIMES:
        raise KeyError(
            f"Unknown state regime {name!r}. "
            f"Bundled: {sorted(_REGIMES)}. Pass a custom StateTaxRegime "
            f"to cfg.state_regime if your state isn't bundled."
        )
    return _REGIMES[key]
