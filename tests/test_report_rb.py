"""Tests for Tier R-B improvements to `build_action_report`.

R-B added five pieces of behavior to the action-plan markdown:

  1. A "This year's concrete actions" sub-section at the top of §6
     with year-1 dollar amounts (deferrals, matches, mega-backdoor,
     backdoor IRA, HSA, conversion, expected tax bill).
  2. A widow's-penalty risk paragraph between the assumptions block
     and §2, rendered only when mortality fires inside the horizon
     and the two spouses die in different years.
  3. A TCJA-sunset (or any regime-change) paragraph quantifying the
     bracket / AGI / fed-tax jump at the regime boundary.
  4. A conditional `State tax` column in §7, active only when the
     state regime is not `STATELESS` *and* the retirement years
     actually accrue state tax.
  5. A conditional `Health $` column in §7 collapsing Medicare-base
     + pre-65 health − ACA-premium-credit into a single line.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from tax_optimizer import (
    Config,
    Inputs,
    Mortality,
    SocialSecurity,
    StartingBalances,
    SUNSET_2026,
    simulate,
)
from tax_optimizer.metrics import summarize
from tax_optimizer.report import (
    _sunset_paragraph,
    _this_year_actions,
    _widow_paragraph,
    build_action_report,
)
from tax_optimizer.results import StrategyResult
from tax_optimizer.tax.state import CA


def _sens_df_min() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "param": "spouse_a_roth_401k_pct",
            "low_value": 0.0, "high_value": 1.0,
            "delta_low": 1000.0, "delta_high": 0.0,
            "swing": 1000.0,
        },
        {
            "param": "inflation",
            "low_value": 0.01, "high_value": 0.04,
            "delta_low": 500.0, "delta_high": -500.0,
            "swing": 500.0,
        },
        {
            "param": "spouse_b_retire_age",
            "low_value": 62.0, "high_value": 67.0,
            "delta_low": -1000.0, "delta_high": 1200.0,
            "swing": 1200.0,
        },
    ])


def _results_with_winner(cfg: Config, inputs: Inputs) -> dict[str, StrategyResult]:
    """Build a 2-strategy results dict that's enough for `build_action_report`
    while exercising the lever-diff and TL;DR paths."""
    df = simulate(cfg, inputs)
    s0 = StrategyResult(cfg=cfg, inputs=inputs, df=df, summary=summarize(df))
    s3_inp = replace(inputs, spouse_a_roth_401k_pct=0.5)
    s3_df = simulate(cfg, s3_inp)
    s3 = StrategyResult(
        cfg=cfg, inputs=s3_inp, df=s3_df, summary=summarize(s3_df)
    )
    return {"S0_baseline": s0, "S3_optimizer": s3}


# =====================================================================
# R-B.1 — "This year's concrete actions"
# =====================================================================


class TestThisYearActions:
    def test_section_header_present(self) -> None:
        cfg, inp = Config(), Inputs()
        df = simulate(cfg, inp)
        lines = _this_year_actions(cfg, inp, df)
        text = "\n".join(lines)
        assert "This year's concrete actions" in text
        # The simulator starts in 2026 by default.
        assert "calendar year 2026" in text

    def test_lists_employee_deferral_when_nonzero(self) -> None:
        cfg, inp = Config(), Inputs()
        df = simulate(cfg, inp)
        text = "\n".join(_this_year_actions(cfg, inp, df))
        assert "401(k) employee deferral" in text

    def test_lists_mega_backdoor_only_when_enabled(self) -> None:
        cfg = Config()
        inp_off = Inputs(
            spouse_a_mega_backdoor_enabled=False,
            spouse_b_mega_backdoor_enabled=False,
        )
        inp_on = Inputs(
            spouse_a_mega_backdoor_enabled=True,
            spouse_b_mega_backdoor_enabled=True,
            spouse_a_after_tax_401k_pct=0.10,
            spouse_b_after_tax_401k_pct=0.10,
        )
        text_off = "\n".join(_this_year_actions(cfg, inp_off, simulate(cfg, inp_off)))
        text_on = "\n".join(_this_year_actions(cfg, inp_on, simulate(cfg, inp_on)))
        assert "Mega-backdoor Roth" not in text_off
        assert "Mega-backdoor Roth" in text_on

    def test_expected_tax_bill_line_present(self) -> None:
        cfg, inp = Config(), Inputs()
        df = simulate(cfg, inp)
        text = "\n".join(_this_year_actions(cfg, inp, df))
        assert "Expected year-1 tax bill" in text
        assert "federal $" in text
        assert "FICA $" in text


# =====================================================================
# R-B.2 — Widow's-penalty paragraph
# =====================================================================


class TestWidowParagraph:
    def test_empty_when_both_spouses_live_past_horizon(self) -> None:
        cfg = Config(
            mortality=Mortality(year_of_death_a=None, year_of_death_b=None),
        )
        inp = Inputs()
        df = simulate(cfg, inp)
        assert _widow_paragraph(cfg, inp, df) == []

    def test_empty_when_both_die_same_year(self) -> None:
        cfg = Config(
            mortality=Mortality(year_of_death_a=30, year_of_death_b=30),
        )
        inp = Inputs()
        df = simulate(cfg, inp)
        assert _widow_paragraph(cfg, inp, df) == []

    def test_renders_when_one_spouse_dies_mid_horizon(self) -> None:
        cfg = Config(
            horizon_age=95,
            mortality=Mortality(year_of_death_a=20, year_of_death_b=40, pension_survivor_pct=0.5),
        )
        inp = Inputs()
        df = simulate(cfg, inp)
        lines = _widow_paragraph(cfg, inp, df)
        text = "\n".join(lines)
        assert "Widow's-penalty risk" in text
        assert "Spouse A dies year 20" in text
        assert "Survivor (Spouse B)" in text

    def test_renders_b_dies_first_correctly(self) -> None:
        cfg = Config(
            horizon_age=95,
            mortality=Mortality(year_of_death_a=40, year_of_death_b=20, pension_survivor_pct=0.5),
        )
        inp = Inputs()
        df = simulate(cfg, inp)
        text = "\n".join(_widow_paragraph(cfg, inp, df))
        assert "Spouse B dies year 20" in text
        assert "Survivor (Spouse A)" in text


# =====================================================================
# R-B.3 — Regime-change (sunset) paragraph
# =====================================================================


class TestSunsetParagraph:
    def test_empty_when_no_regime_change_configured(self) -> None:
        cfg = Config(regime_change_year_offset=None, regime_change_target=None)
        inp = Inputs()
        df = simulate(cfg, inp)
        assert _sunset_paragraph(cfg, inp, df) == []

    def test_renders_when_regime_change_inside_horizon(self) -> None:
        cfg = Config(regime_change_year_offset=5, regime_change_target=SUNSET_2026)
        inp = Inputs()
        df = simulate(cfg, inp)
        text = "\n".join(_sunset_paragraph(cfg, inp, df))
        assert "Tax-regime change" in text
        assert "year 5" in text
        # Calendar year = start_year (2026) + 5
        assert "calendar 2031" in text
        # Should include the delta table headers
        assert "Last year of old regime" in text
        assert "First year of new regime" in text


# =====================================================================
# R-B.4 — Conditional State-tax column in §7
# =====================================================================


class TestStateTaxColumn:
    def test_absent_for_stateless_regime(self) -> None:
        cfg, inp = Config(), Inputs()
        results = _results_with_winner(cfg, inp)
        report = build_action_report(cfg, inp, results, _sens_df_min(), 1.0)
        section7 = report.split("## 7.")[1].split("## 8.")[0]
        assert "State tax" not in section7

    def test_present_when_state_regime_set(self) -> None:
        cfg = Config(state_regime=CA)
        inp = Inputs()
        results = _results_with_winner(cfg, inp)
        report = build_action_report(cfg, inp, results, _sens_df_min(), 1.0)
        section7 = report.split("## 7.")[1].split("## 8.")[0]
        assert "State tax" in section7


# =====================================================================
# R-B.5 — Conditional Healthcare column in §7
# =====================================================================


class TestHealthcareColumn:
    def test_absent_when_no_healthcare_lines(self) -> None:
        cfg = Config(
            medicare_base_b_d_premium=0,
            health_pre65_today=0,
            aca_enabled=False,
        )
        inp = Inputs()
        results = _results_with_winner(cfg, inp)
        report = build_action_report(cfg, inp, results, _sens_df_min(), 1.0)
        section7 = report.split("## 7.")[1].split("## 8.")[0]
        assert "Health $" not in section7

    def test_present_when_medicare_base_set(self) -> None:
        cfg = Config(medicare_base_b_d_premium=2500)
        inp = Inputs()
        results = _results_with_winner(cfg, inp)
        report = build_action_report(cfg, inp, results, _sens_df_min(), 1.0)
        section7 = report.split("## 7.")[1].split("## 8.")[0]
        assert "Health $" in section7

    def test_legend_extended_when_health_column_visible(self) -> None:
        cfg = Config(medicare_base_b_d_premium=2500)
        inp = Inputs()
        results = _results_with_winner(cfg, inp)
        report = build_action_report(cfg, inp, results, _sens_df_min(), 1.0)
        section7 = report.split("## 7.")[1].split("## 8.")[0]
        assert "Medicare base premium + pre-65 healthcare" in section7
