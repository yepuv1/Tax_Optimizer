"""Tests for Tier R-C improvements.

R-C added three pieces of behaviour:

  1. `_optimizer_rationale(cfg, inputs, w_cfg, w_inputs, w_df, w_summary)` —
     heuristic bullet points explaining *why* the recommended plan looks
     the way it does. Pattern checks:
       * Heir-rate framing (heir vs avg-retirement marginal)
       * Sunset front-loading (regime change configured + conversions chosen)
       * Bucket imbalance (pretax / Roth ratio > 5)
       * Mega-backdoor activity
       * Peak marginal sticker shock (>= 32%)
  2. `_heir_rate_sensitivity(w_cfg, w_df)` — sweeps `heir_marginal_rate`
     across {0.10, 0.22, 0.32, 0.37} (plus the current value if not in
     the set) and shows the after-tax NW per row.
  3. `compare_scenarios(scenarios, *, mc=None)` — public side-by-side
     diff of N independent scenarios, with optional MC overlay.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from tax_optimizer import (
    Config,
    Inputs,
    LognormalModel,
    Mortality,
    StartingBalances,
    SUNSET_2026,
    compare_scenarios,
    simulate,
)
from tax_optimizer.metrics import summarize
from tax_optimizer.report import (
    _heir_rate_sensitivity,
    _optimizer_rationale,
)
from tax_optimizer.tax.state import CA, NY


# =====================================================================
# R-C.1 — Optimizer rationale
# =====================================================================


class TestOptimizerRationale:
    def test_heading_present(self) -> None:
        cfg, inp = Config(), Inputs()
        df = simulate(cfg, inp)
        s = summarize(df, heir_marginal_rate=cfg.heir_marginal_rate)
        text = "\n".join(_optimizer_rationale(cfg, inp, cfg, inp, df, s))
        assert "Why the optimizer chose this" in text

    def test_heir_rate_pattern_fires_when_heir_meets_marginal(self) -> None:
        # heir_rate >= avg_retirement_marginal AND chose conversions.
        cfg = Config(
            heir_marginal_rate=0.32,
            roth_conversion_target_bracket=0.22,
        )
        inp = Inputs()
        df = simulate(cfg, inp)
        s = summarize(df, heir_marginal_rate=cfg.heir_marginal_rate)
        text = "\n".join(_optimizer_rationale(cfg, inp, cfg, inp, df, s))
        assert "Heir-rate framing" in text

    def test_no_conversion_pattern_fires_when_no_target(self) -> None:
        cfg = Config(
            heir_marginal_rate=0.10,
            roth_conversion_target_bracket=0.0,
        )
        inp = Inputs()
        df = simulate(cfg, inp)
        s = summarize(df, heir_marginal_rate=cfg.heir_marginal_rate)
        text = "\n".join(_optimizer_rationale(cfg, inp, cfg, inp, df, s))
        assert "No conversions chosen" in text

    def test_sunset_pattern_fires_when_regime_change_configured(self) -> None:
        cfg = Config(
            heir_marginal_rate=0.32,
            roth_conversion_target_bracket=0.22,
            regime_change_year_offset=5,
            regime_change_target=SUNSET_2026,
        )
        inp = Inputs()
        df = simulate(cfg, inp)
        s = summarize(df, heir_marginal_rate=cfg.heir_marginal_rate)
        text = "\n".join(_optimizer_rationale(cfg, inp, cfg, inp, df, s))
        assert "Sunset front-loading" in text

    def test_bucket_imbalance_pattern_fires_when_pretax_dominant(self) -> None:
        cfg = Config()
        inp = Inputs(
            starting=StartingBalances(
                spouse_a_pretax_401k=2_000_000,
                spouse_b_pretax_401k=500_000,
                spouse_a_roth_ira=10_000,
                taxable_brokerage=200_000,
            ),
        )
        df = simulate(cfg, inp)
        s = summarize(df, heir_marginal_rate=cfg.heir_marginal_rate)
        text = "\n".join(_optimizer_rationale(cfg, inp, cfg, inp, df, s))
        assert "Bucket imbalance" in text

    def test_renders_at_most_three_bullets(self) -> None:
        # Configure a household where many patterns could fire at once.
        cfg = Config(
            heir_marginal_rate=0.32,
            roth_conversion_target_bracket=0.32,
            regime_change_year_offset=5,
            regime_change_target=SUNSET_2026,
        )
        inp = Inputs(
            starting=StartingBalances(
                spouse_a_pretax_401k=3_000_000,
                spouse_a_roth_ira=5_000,
            ),
        )
        df = simulate(cfg, inp)
        s = summarize(df, heir_marginal_rate=cfg.heir_marginal_rate)
        out = _optimizer_rationale(cfg, inp, cfg, inp, df, s)
        # Section header + blank line + at most 3 bullet lines + blank line.
        bullet_lines = [line for line in out if line.startswith("- ")]
        assert 1 <= len(bullet_lines) <= 3


# =====================================================================
# R-C.2 — Heir-rate sensitivity
# =====================================================================


class TestHeirRateSensitivity:
    def test_renders_all_canonical_rates(self) -> None:
        cfg, inp = Config(), Inputs()
        df = simulate(cfg, inp)
        text = "\n".join(_heir_rate_sensitivity(cfg, df))
        for rate in ("10%", "22%", "32%", "37%"):
            assert rate in text

    def test_marks_current_assumption(self) -> None:
        cfg = Config(heir_marginal_rate=0.22)
        inp = Inputs()
        df = simulate(cfg, inp)
        text = "\n".join(_heir_rate_sensitivity(cfg, df))
        assert "current assumption" in text

    def test_includes_unusual_current_rate(self) -> None:
        # 0.27 is not in the canonical set; sweep should still include it.
        cfg = Config(heir_marginal_rate=0.27)
        inp = Inputs()
        df = simulate(cfg, inp)
        text = "\n".join(_heir_rate_sensitivity(cfg, df))
        assert "27%" in text

    def test_terminal_nw_strictly_decreases_with_higher_heir_rate(self) -> None:
        # Sanity: bigger heir haircut → smaller after-tax NW.
        cfg = Config(heir_marginal_rate=0.22)
        inp = Inputs()
        df = simulate(cfg, inp)
        from tax_optimizer.metrics import terminal_after_tax_nw
        nw_low = terminal_after_tax_nw(df, heir_marginal_rate=0.10)
        nw_mid = terminal_after_tax_nw(df, heir_marginal_rate=0.22)
        nw_high = terminal_after_tax_nw(df, heir_marginal_rate=0.37)
        assert nw_low > nw_mid > nw_high


# =====================================================================
# R-C.3 — compare_scenarios()
# =====================================================================


class TestCompareScenarios:
    def test_returns_markdown_for_single_scenario(self) -> None:
        cfg, inp = Config(), Inputs()
        text = compare_scenarios([("baseline", cfg, inp)])
        assert "# Scenario comparison" in text
        assert "baseline" in text
        # Outcome metrics block always present.
        assert "Terminal after-tax NW" in text

    def test_renders_each_scenario_as_a_column(self) -> None:
        cfg = Config()
        inp = Inputs()
        text = compare_scenarios([
            ("home", cfg, inp),
            ("CA",   replace(cfg, state_regime=CA), inp),
            ("NY",   replace(cfg, state_regime=NY), inp),
        ])
        for n in ("home", "CA", "NY"):
            assert n in text
        # Assumptions block surfaces the state regimes.
        assert "stateless" in text
        assert "CA" in text
        assert "NY" in text

    def test_bolds_best_value_per_metric(self) -> None:
        # CA and NY both add state tax → terminal NW lower.
        cfg = Config()
        inp = Inputs()
        text = compare_scenarios([
            ("stateless", cfg, inp),
            ("CA", replace(cfg, state_regime=CA), inp),
            ("NY", replace(cfg, state_regime=NY), inp),
        ])
        # Find the terminal NW row and confirm it has at least one bolded cell.
        for line in text.splitlines():
            if line.startswith("| Terminal after-tax NW"):
                assert "**" in line
                break
        else:
            raise AssertionError("Terminal NW row not present")

    def test_raises_on_duplicate_names(self) -> None:
        cfg, inp = Config(), Inputs()
        try:
            compare_scenarios([("a", cfg, inp), ("a", cfg, inp)])
        except ValueError as e:
            assert "unique" in str(e)
        else:
            raise AssertionError("duplicate names should have raised")

    def test_empty_input_returns_polite_message(self) -> None:
        text = compare_scenarios([])
        assert "no scenarios" in text.lower()

    def test_mc_block_renders_when_provided(self) -> None:
        from tax_optimizer import simulate_paths
        cfg = Config(market=LognormalModel(equity_bond_corr=0.10))
        inp = Inputs()
        mc = simulate_paths(cfg, inp, n_paths=20, seed=0, keep_paths=False)
        text = compare_scenarios(
            [("baseline", cfg, inp)],
            mc={"baseline": mc},
        )
        assert "Risk picture (Monte Carlo)" in text
        assert "P(success)" in text
        assert "CVaR(10%)" in text

    def test_mortality_summary_handles_none(self) -> None:
        cfg = Config(mortality=Mortality(year_of_death_a=None, year_of_death_b=None))
        inp = Inputs()
        text = compare_scenarios([("immortal", cfg, inp)])
        assert "∞" in text
