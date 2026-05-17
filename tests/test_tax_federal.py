"""Tests for `tax_optimizer.tax.federal` and SS taxability helpers."""

from __future__ import annotations

import math

import pytest

from tax_optimizer.tax.federal import (
    _bracket_tax,
    _marginal_rate,
    amount_to_fill_bracket,
    federal_tax,
    social_security_taxable,
)
from tax_optimizer.tax.regimes import (
    PRE_TCJA_2017,
    SUNSET_2026,
    TCJA_EXTENDED,
)


# ---------------------------------------------------------------------------
# Bracket math (private, but the entire engine rests on it)
# ---------------------------------------------------------------------------


class TestBracketTax:
    """Two-bracket sanity, then the real MFJ table."""

    def test_zero_income_pays_zero(self) -> None:
        brackets = [(0, 100, 0.10), (100, math.inf, 0.20)]
        assert _bracket_tax(0, brackets) == 0.0

    def test_within_first_bracket(self) -> None:
        brackets = [(0, 100, 0.10), (100, math.inf, 0.20)]
        assert _bracket_tax(50, brackets) == pytest.approx(5.0)

    def test_at_first_bracket_top(self) -> None:
        brackets = [(0, 100, 0.10), (100, math.inf, 0.20)]
        assert _bracket_tax(100, brackets) == pytest.approx(10.0)

    def test_spans_two_brackets(self) -> None:
        brackets = [(0, 100, 0.10), (100, math.inf, 0.20)]
        # 100 * 10% + 50 * 20% = 20
        assert _bracket_tax(150, brackets) == pytest.approx(20.0)

    def test_full_mfj_2026_table_120k(self) -> None:
        # Hand calc against TCJA_EXTENDED MFJ ord brackets at $120k:
        #   23,850 * 10%        =  2,385.00
        #   (96,950-23,850)*12% =  8,772.00
        #   (120,000-96,950)*22%=  5,071.00
        # Total = 16,228.00
        tax = _bracket_tax(120_000, TCJA_EXTENDED.ord_brackets_mfj)
        assert tax == pytest.approx(16_228.00, abs=0.01)

    def test_marginal_rate_returns_top_active_rate(self) -> None:
        assert _marginal_rate(0, TCJA_EXTENDED.ord_brackets_mfj) == 0.10
        assert _marginal_rate(50_000, TCJA_EXTENDED.ord_brackets_mfj) == 0.12
        assert _marginal_rate(100_000, TCJA_EXTENDED.ord_brackets_mfj) == 0.22
        assert _marginal_rate(800_000, TCJA_EXTENDED.ord_brackets_mfj) == 0.37


# ---------------------------------------------------------------------------
# Social Security taxability (IRC §86)
# ---------------------------------------------------------------------------


class TestSocialSecurityTaxable:
    """The 0% / 50% / 85% thresholds are baked into MFJ at (32k, 44k)."""

    BASE1, BASE2 = 32_000.0, 44_000.0

    def test_no_benefits_returns_zero(self) -> None:
        out = social_security_taxable(
            provisional=100_000, ss_benefits=0, base1=self.BASE1, base2=self.BASE2
        )
        assert out == 0.0

    def test_below_base1_zero_taxable(self) -> None:
        # provisional well below 32k → none taxable
        out = social_security_taxable(
            provisional=20_000, ss_benefits=24_000, base1=self.BASE1, base2=self.BASE2
        )
        assert out == 0.0

    def test_between_base1_and_base2_50pct_tier(self) -> None:
        # provisional just inside 50% tier; taxable = min(0.5*(p-base1), 0.5*ss)
        out = social_security_taxable(
            provisional=38_000, ss_benefits=24_000, base1=self.BASE1, base2=self.BASE2
        )
        # 0.5 * (38_000 - 32_000) = 3_000  ;  0.5 * 24_000 = 12_000
        assert out == pytest.approx(3_000.0)

    def test_above_base2_capped_at_85pct(self) -> None:
        # high provisional → 85% of ss is the binding cap
        out = social_security_taxable(
            provisional=200_000, ss_benefits=24_000, base1=self.BASE1, base2=self.BASE2
        )
        assert out == pytest.approx(0.85 * 24_000)

    def test_just_above_base2_uses_two_tier_formula(self) -> None:
        # tier1 = min(0.5*(44-32), 0.5*ss) = min(6_000, 12_000) = 6_000
        # tier2 = 0.85 * (50_000 - 44_000) = 5_100
        # combined = 11_100; cap = 0.85*24_000 = 20_400 → not binding
        out = social_security_taxable(
            provisional=50_000, ss_benefits=24_000, base1=self.BASE1, base2=self.BASE2
        )
        assert out == pytest.approx(11_100.0)


# ---------------------------------------------------------------------------
# federal_tax: the single year-of-tax engine the simulator drives
# ---------------------------------------------------------------------------


class TestFederalTax:
    def test_zero_income_zero_tax(self) -> None:
        out = federal_tax(regime=TCJA_EXTENDED)
        assert out["tax"] == 0.0
        assert out["agi"] == 0.0
        assert out["taxable_income"] == 0.0
        assert out["marginal"] == 0.10
        assert out["ss_taxable"] == 0.0

    def test_wages_under_std_deduction_no_ord_tax(self) -> None:
        # MFJ standard deduction = 32,200; wages below that → 0 tax
        out = federal_tax(regime=TCJA_EXTENDED, wages=30_000)
        assert out["taxable_income"] == 0.0
        assert out["ordinary_tax"] == 0.0
        assert out["tax"] == 0.0
        assert out["marginal"] == 0.10

    def test_wages_above_std_deduction_dollar_check(self) -> None:
        # $100k wages, MFJ. AGI=100k; taxable=100k-32_200 = 67_800
        # Bracket math:
        #   23_850 * 10%        = 2_385
        #   (67_800-23_850)*12% = 5_274
        # Total ord_tax = 7_659; no LTCG, no NIIT
        out = federal_tax(regime=TCJA_EXTENDED, wages=100_000)
        assert out["taxable_income"] == pytest.approx(67_800.0)
        assert out["ordinary_tax"] == pytest.approx(7_659.0, abs=0.01)
        assert out["ltcg_tax"] == 0.0
        assert out["niit"] == 0.0
        assert out["tax"] == pytest.approx(7_659.0, abs=0.01)
        assert out["marginal"] == 0.12

    def test_filing_status_single_uses_single_tables(self) -> None:
        single = federal_tax(regime=TCJA_EXTENDED, filing_status="single", wages=80_000)
        mfj = federal_tax(regime=TCJA_EXTENDED, filing_status="mfj", wages=80_000)
        assert single["tax"] > mfj["tax"]
        # Single std deduction is half of MFJ:
        assert single["taxable_income"] > mfj["taxable_income"]

    def test_qualified_div_uses_ltcg_brackets(self) -> None:
        # MFJ, $50k wages + $30k qualified dividends
        # Ordinary taxable = max(0, (50_000+30_000) - 32_200 - pref_taxable)
        # The engine fills ord brackets first then layers LTCG above.
        out = federal_tax(regime=TCJA_EXTENDED, wages=50_000, qualified_div=30_000)
        # All preferential income should sit in the 0% LTCG bracket (top
        # of 0% MFJ = 96_700, so it's fully sheltered).
        assert out["ltcg_tax"] == 0.0
        assert out["agi"] == 80_000.0

    def test_high_ltcg_partially_taxed(self) -> None:
        # MFJ; force taxable_total ABOVE the 0% LTCG ceiling so some LTCG
        # spills into the 15% bracket.
        out = federal_tax(regime=TCJA_EXTENDED, wages=120_000, ltcg=50_000)
        assert out["ltcg_tax"] > 0.0
        # NIIT triggered: AGI=170k, threshold=250k → no NIIT yet
        assert out["niit"] == 0.0

    def test_niit_kicks_in_above_mfj_threshold(self) -> None:
        out = federal_tax(
            regime=TCJA_EXTENDED, wages=300_000, qualified_div=20_000
        )
        # AGI=320_000, investment_income=20_000, threshold=250_000
        # NIIT = 3.8% * min(20_000, 320_000-250_000) = 3.8% * 20_000 = 760
        assert out["niit"] == pytest.approx(760.0, abs=0.01)

    def test_explicit_deduction_override(self) -> None:
        # Pass deduction=0 → taxable_income = AGI; gives a different number
        # than the default standard deduction would.
        default = federal_tax(regime=TCJA_EXTENDED, wages=100_000)
        no_ded = federal_tax(regime=TCJA_EXTENDED, wages=100_000, deduction=0.0)
        assert no_ded["taxable_income"] > default["taxable_income"]
        assert no_ded["tax"] > default["tax"]

    def test_pretax_withdrawal_taxed_as_ordinary(self) -> None:
        a = federal_tax(regime=TCJA_EXTENDED, wages=50_000, pretax_withdrawal=20_000)
        b = federal_tax(regime=TCJA_EXTENDED, wages=70_000)
        # IRA withdrawal feeds the same ordinary line as wages → identical AGI / tax
        assert a["agi"] == b["agi"]
        assert a["tax"] == pytest.approx(b["tax"])

    def test_roth_conversion_taxed_as_ordinary(self) -> None:
        a = federal_tax(regime=TCJA_EXTENDED, wages=50_000, roth_conversion=20_000)
        b = federal_tax(regime=TCJA_EXTENDED, wages=70_000)
        assert a["agi"] == b["agi"]
        assert a["tax"] == pytest.approx(b["tax"])

    def test_ss_inclusion_in_provisional_drives_taxability(self) -> None:
        # Wages=40k, SS=24k → provisional=40k+12k=52k > 44k base2 → 85% tier
        out = federal_tax(regime=TCJA_EXTENDED, wages=40_000, social_security=24_000)
        # tier1 = min(0.5*12_000, 0.5*24_000) = 6_000
        # tier2 = 0.85 * (52_000 - 44_000) = 6_800
        # combined = 12_800; cap = 0.85*24_000 = 20_400 → not binding
        assert out["ss_taxable"] == pytest.approx(12_800.0)

    def test_tax_exempt_interest_added_to_provisional_only(self) -> None:
        # IRC §86(b)(2)(B): muni-bond interest counts toward SS
        # provisional but stays out of AGI / taxable income.
        # Reproduce the Pub 915 worksheet 1 line 3 add-back.
        #
        # Setup: $30k wages, $24k SS, no other income.
        # Without muni interest: provisional = 30k + 12k = 42k.
        #   42k is between 32k and 44k → tier1 only
        #   tier1 = min(0.5*(42-32), 0.5*24) = min(5, 12) = 5
        # With $5k muni interest: provisional = 30k + 5k + 12k = 47k.
        #   47k > 44k → tier2 fires
        #   tier1 = min(0.5*(44-32), 0.5*24) = 6
        #   tier2 = 0.85*(47-44) = 2.55
        #   combined = 8.55; cap = 0.85*24 = 20.4 → not binding
        no_muni = federal_tax(
            regime=TCJA_EXTENDED, wages=30_000, social_security=24_000
        )
        with_muni = federal_tax(
            regime=TCJA_EXTENDED,
            wages=30_000,
            social_security=24_000,
            tax_exempt_interest=5_000.0,
        )
        # Pre-fix `with_muni["ss_taxable"]` would have been the same
        # as `no_muni` because the kwarg didn't exist; the muni
        # would have to be passed via ``interest`` (which then
        # incorrectly federalized the muni). Post-fix the muni only
        # bumps the provisional line.
        assert no_muni["ss_taxable"] == pytest.approx(5_000.0)
        assert with_muni["ss_taxable"] == pytest.approx(8_550.0, abs=1.0)
        # AGI is NOT affected by the muni: federal exemption preserved.
        assert with_muni["agi"] == pytest.approx(
            no_muni["agi"] + (with_muni["ss_taxable"] - no_muni["ss_taxable"]),
            abs=1.0,
        )
        # The muni interest itself contributes $0 to ordinary AGI
        # (the only delta vs no_muni is the SS taxable bump above).


# ---------------------------------------------------------------------------
# Regime switching
# ---------------------------------------------------------------------------


class TestRegimes:
    def test_pre_tcja_higher_tax_than_tcja_at_mid_income(self) -> None:
        # Pre-TCJA brackets are wider but rates are higher in the middle.
        wages = 200_000
        tcja = federal_tax(regime=TCJA_EXTENDED, wages=wages)
        pre = federal_tax(regime=PRE_TCJA_2017, wages=wages)
        # Both regimes share the same NIIT threshold so MFJ comparison is clean.
        assert pre["tax"] > tcja["tax"]

    def test_sunset_brackets_are_inflated_pre_tcja(self) -> None:
        # The widths should be ~1.30x pre-TCJA. Verify std deduction matches
        # the documented multiplier exactly.
        assert SUNSET_2026.std_deduction_mfj == pytest.approx(
            PRE_TCJA_2017.std_deduction_mfj * 1.30
        )
        assert SUNSET_2026.std_deduction_single == pytest.approx(
            PRE_TCJA_2017.std_deduction_single * 1.30
        )

    def test_regime_helper_dispatch_by_filing_status(self) -> None:
        assert TCJA_EXTENDED.std_deduction("mfj") == 32_200.0
        assert TCJA_EXTENDED.std_deduction("single") == 16_100.0
        assert TCJA_EXTENDED.niit_threshold("mfj") == 250_000.0
        assert TCJA_EXTENDED.niit_threshold("single") == 200_000.0
        assert TCJA_EXTENDED.ss_provisional("mfj") == (32_000.0, 44_000.0)
        assert TCJA_EXTENDED.ss_provisional("single") == (25_000.0, 34_000.0)
        # Bracket dispatch returns the right list reference (not a copy).
        assert (
            TCJA_EXTENDED.ord_brackets("mfj") is TCJA_EXTENDED.ord_brackets_mfj
        )
        assert (
            TCJA_EXTENDED.ltcg_brackets("single")
            is TCJA_EXTENDED.ltcg_brackets_single
        )


# ---------------------------------------------------------------------------
# amount_to_fill_bracket
# ---------------------------------------------------------------------------


class TestAmountToFillBracket:
    def test_room_left_in_22pct_bracket(self) -> None:
        # MFJ 22% bracket runs (96_950, 206_700). At base_taxable=100_000
        # there's 106_700 of headroom up to the top of the 22% slab.
        room = amount_to_fill_bracket(
            base_taxable=100_000,
            target_top=0.22,
            brackets=TCJA_EXTENDED.ord_brackets_mfj,
        )
        assert room == pytest.approx(206_700.0 - 100_000.0)

    def test_already_above_target_top_returns_zero(self) -> None:
        room = amount_to_fill_bracket(
            base_taxable=300_000,
            target_top=0.22,
            brackets=TCJA_EXTENDED.ord_brackets_mfj,
        )
        assert room == 0.0

    def test_unknown_target_rate_returns_zero(self) -> None:
        room = amount_to_fill_bracket(
            base_taxable=10_000,
            target_top=0.99,  # not a real rate
            brackets=TCJA_EXTENDED.ord_brackets_mfj,
        )
        assert room == 0.0
